#!/usr/bin/env python3
"""LSTM Autoencoder for unplanned traffic incident anomaly detection (Module B)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from flowcast_config import (
    DATASET_PATH,
    FEATURE_MATRIX_PATH,
    LSTM_AE_PATH,
    MODELS_DIR,
    OCCUPANCY_COL,
    OUTPUTS_EDA_DIR,
    RANDOM_SEED,
    SEGMENT_ID_COL,
    SPEED_COL,
    THRESHOLD_JSON,
    TIMESTAMP_COL,
    VOLUME_COL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WINDOW_SIZE = 6
FEATURE_COLS = [SPEED_COL, VOLUME_COL, OCCUPANCY_COL]
DEVICE = torch.device("cpu")
THRESHOLD_MULTIPLIER = 2.5
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_feature_matrix() -> pd.DataFrame:
    """Build feature_matrix.parquet from traffic CSV when missing."""
    if FEATURE_MATRIX_PATH.exists():
        return pd.read_parquet(FEATURE_MATRIX_PATH)

    logger.warning("Feature matrix not found — building from traffic CSV.")
    if not DATASET_PATH.exists():
        from generate_demo_data import generate_traffic_csv

        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_traffic_csv().to_csv(DATASET_PATH, index=False)
        logger.info("Generated demo traffic CSV at %s", DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values([SEGMENT_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)
    df["is_event_related"] = df.get("event_type", "none").fillna("none").ne("none")

    FEATURE_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURE_MATRIX_PATH, index=False)
    logger.info("Saved feature matrix: %s (%s rows)", FEATURE_MATRIX_PATH, f"{len(df):,}")
    return df


def load_normal_traffic() -> pd.DataFrame:
    """Load feature matrix and keep only non-event (normal) traffic rows."""
    df = ensure_feature_matrix()
    if "is_event_related" not in df.columns:
        df["is_event_related"] = False
    normal = df[~df["is_event_related"].astype(bool)].copy()
    normal[TIMESTAMP_COL] = pd.to_datetime(normal[TIMESTAMP_COL])
    normal = normal.sort_values([SEGMENT_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)
    logger.info("Normal traffic rows: %s / %s", f"{len(normal):,}", f"{len(df):,}")
    return normal


def create_sliding_windows(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
    """
    Build sliding windows per segment.

    Returns:
        windows (n, window_size, 3), segment_ids (n,), segment list, fitted scaler.
    """
    if scaler is None:
        scaler = StandardScaler()

    raw_values = df[FEATURE_COLS].astype(float).values
    if fit_scaler:
        scaler.fit(raw_values)
    normalized = scaler.transform(raw_values)
    df = df.copy()
    for i, col in enumerate(FEATURE_COLS):
        df[f"_{col}"] = normalized[:, i]

    windows: list[np.ndarray] = []
    seg_ids: list[str] = []

    norm_cols = [f"_{c}" for c in FEATURE_COLS]
    for seg_id, group in df.groupby(SEGMENT_ID_COL, sort=False):
        values = group[norm_cols].values
        if len(values) < WINDOW_SIZE:
            continue
        for start in range(len(values) - WINDOW_SIZE + 1):
            windows.append(values[start : start + WINDOW_SIZE])
            seg_ids.append(str(seg_id))

    if not windows:
        raise ValueError("No sliding windows could be created — check data length per segment.")

    return np.stack(windows), np.array(seg_ids), sorted(df[SEGMENT_ID_COL].astype(str).unique()), scaler


class TrafficAutoencoder(nn.Module):
    """LSTM autoencoder: encoder 3→64→32→16, decoder reverse."""

    def __init__(self, input_dim: int = 3, window_size: int = WINDOW_SIZE) -> None:
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim

        self.encoder_lstm1 = nn.LSTM(
            input_dim, 64, num_layers=2, dropout=0.1, batch_first=True
        )
        self.encoder_lstm2 = nn.LSTM(64, 32, num_layers=1, batch_first=True)
        self.bottleneck = nn.Linear(32, 16)

        self.decoder_expand = nn.Linear(16, 32)
        self.decoder_lstm1 = nn.LSTM(32, 32, num_layers=1, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(32, 64, num_layers=2, dropout=0.1, batch_first=True)
        self.output_layer = nn.Linear(64, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder_lstm1(x)
        out, _ = self.encoder_lstm2(out)
        return self.bottleneck(out[:, -1, :])

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        dec_in = self.decoder_expand(latent).unsqueeze(1).repeat(1, self.window_size, 1)
        out, _ = self.decoder_lstm1(dec_in)
        out, _ = self.decoder_lstm2(out)
        return self.output_layer(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=(1, 2))


def train_autoencoder(
    train_windows: np.ndarray,
    val_windows: np.ndarray,
) -> tuple[TrafficAutoencoder, dict]:
    """Train LSTM autoencoder with early stopping on validation MSE."""
    model = TrafficAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_windows, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_tensor = torch.tensor(val_windows, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor.to(DEVICE))
            val_loss = criterion(val_recon, val_tensor.to(DEVICE)).item()

        avg_train = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": val_loss})
        logger.info("Epoch %3d | train_loss=%.6f | val_loss=%.6f", epoch, avg_train, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, EARLY_STOPPING_PATIENCE)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_val_loss": best_val_loss}


def compute_errors(model: TrafficAutoencoder, windows: np.ndarray) -> np.ndarray:
    """Per-window MSE reconstruction error."""
    model.eval()
    tensor = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        errors = model.reconstruction_error(tensor).cpu().numpy()
    return errors


def calibrate_thresholds(
    errors: np.ndarray,
    segment_ids: np.ndarray,
) -> dict:
    """Per-segment threshold = mean + 2.5 * std on normal validation data."""
    thresholds: dict[str, dict] = {}
    for seg in np.unique(segment_ids):
        seg_errors = errors[segment_ids == seg]
        mean = float(np.mean(seg_errors))
        std = float(np.std(seg_errors))
        thresholds[str(seg)] = {
            "mean": mean,
            "std": std,
            "threshold": mean + THRESHOLD_MULTIPLIER * std,
            "n_samples": int(len(seg_errors)),
        }

    global_mean = float(np.mean(errors))
    global_std = float(np.std(errors))
    return {
        "threshold_multiplier": THRESHOLD_MULTIPLIER,
        "global": {
            "mean": global_mean,
            "std": global_std,
            "threshold": global_mean + THRESHOLD_MULTIPLIER * global_std,
        },
        "segments": thresholds,
    }


def save_artifacts(
    model: TrafficAutoencoder,
    scaler: StandardScaler,
    thresholds: dict,
) -> None:
    """Persist model checkpoint, scaler, and per-segment thresholds."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "window_size": WINDOW_SIZE,
        "feature_cols": FEATURE_COLS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    torch.save(checkpoint, LSTM_AE_PATH)
    with open(THRESHOLD_JSON, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("Saved model → %s", LSTM_AE_PATH)
    logger.info("Saved thresholds → %s", THRESHOLD_JSON)


def load_artifacts() -> tuple[TrafficAutoencoder, StandardScaler, dict]:
    """Load trained model, scaler, and thresholds."""
    checkpoint = torch.load(LSTM_AE_PATH, map_location=DEVICE, weights_only=False)
    model = TrafficAutoencoder(
        input_dim=len(checkpoint["feature_cols"]),
        window_size=checkpoint["window_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])
    scaler.n_features_in_ = len(scaler.mean_)

    with open(THRESHOLD_JSON, encoding="utf-8") as f:
        thresholds = json.load(f)
    return model, scaler, thresholds


def _segment_threshold(thresholds: dict, segment_id: str) -> float:
    seg = thresholds.get("segments", {}).get(str(segment_id))
    if seg:
        return float(seg["threshold"])
    return float(thresholds["global"]["threshold"])


def detect_anomaly(segment_id: str, latest_30min_data: pd.DataFrame | np.ndarray) -> dict:
    """
    Detect anomaly for one segment from the latest window of traffic data.

    Args:
        segment_id: Road segment identifier.
        latest_30min_data: DataFrame with speed/volume/occupancy columns or
            array shaped (window_size, 3).

    Returns:
        Dict with is_anomaly, error, threshold, confidence.
    """
    model, scaler, thresholds = load_artifacts()

    if isinstance(latest_30min_data, pd.DataFrame):
        values = latest_30min_data[FEATURE_COLS].astype(float).values
    else:
        values = np.asarray(latest_30min_data, dtype=float)

    if values.shape[0] < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} timesteps, got {values.shape[0]}")
    window = values[-WINDOW_SIZE:]
    window_norm = scaler.transform(window)
    tensor = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        error = float(model.reconstruction_error(tensor).item())

    threshold = _segment_threshold(thresholds, segment_id)
    is_anomaly = error > threshold
    # Confidence: how far above/below threshold (sigmoid-like scale)
    margin = (error - threshold) / max(threshold, 1e-6)
    confidence = float(1 / (1 + np.exp(-margin)))

    return {
        "segment_id": str(segment_id),
        "is_anomaly": bool(is_anomaly),
        "error": round(error, 6),
        "threshold": round(threshold, 6),
        "confidence": round(confidence, 4),
    }


def run_realtime_monitor(
    data_stream: Iterator[pd.DataFrame],
    alert_callback: Callable[[str, dict], None],
    poll_interval_sec: float = 0.0,
) -> None:
    """
    Continuously monitor segments; fire alert after 2 consecutive anomalous windows.

    Args:
        data_stream: Iterator yielding DataFrames with segment_id, timestamp, features.
        alert_callback: Called with (segment_id, anomaly_dict) on confirmed alert.
        poll_interval_sec: Optional sleep between stream batches.
    """
    model, scaler, thresholds = load_artifacts()
    consecutive: dict[str, int] = {}

    for batch in data_stream:
        if batch.empty:
            continue
        batch = batch.copy()
        batch[TIMESTAMP_COL] = pd.to_datetime(batch[TIMESTAMP_COL])
        batch = batch.sort_values([SEGMENT_ID_COL, TIMESTAMP_COL])

        for seg_id, group in batch.groupby(SEGMENT_ID_COL):
            if len(group) < WINDOW_SIZE:
                continue
            values = group[FEATURE_COLS].astype(float).values[-WINDOW_SIZE:]
            window_norm = scaler.transform(values)
            tensor = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                error = float(model.reconstruction_error(tensor).item())

            threshold = _segment_threshold(thresholds, str(seg_id))
            result = {
                "segment_id": str(seg_id),
                "is_anomaly": error > threshold,
                "error": round(error, 6),
                "threshold": round(threshold, 6),
                "confidence": round(float(1 / (1 + np.exp(-(error - threshold) / max(threshold, 1e-6)))), 4),
            }

            key = str(seg_id)
            if result["is_anomaly"]:
                consecutive[key] = consecutive.get(key, 0) + 1
                if consecutive[key] >= 2:
                    alert_callback(key, result)
                    consecutive[key] = 0
            else:
                consecutive[key] = 0

        if poll_interval_sec > 0:
            time.sleep(poll_interval_sec)


def synthesize_incident_windows(
    normal_windows: np.ndarray,
    normal_seg_ids: np.ndarray,
    n_incidents: int = 200,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic incident windows by perturbing normal traffic patterns."""
    rng = np.random.default_rng(seed)
    n = min(n_incidents, len(normal_windows))
    idx = rng.choice(len(normal_windows), size=n, replace=False)
    incident_windows = normal_windows[idx].copy()
    incident_windows[:, :, 0] *= rng.uniform(0.2, 0.5, size=n)[:, None]
    incident_windows[:, :, 1] *= rng.uniform(1.3, 2.0, size=n)[:, None]
    incident_windows[:, :, 2] = np.clip(
        incident_windows[:, :, 2] * rng.uniform(1.2, 1.8, size=n)[:, None], 0, 3
    )
    all_windows = np.concatenate([normal_windows, incident_windows], axis=0)
    all_labels = np.concatenate(
        [np.zeros(len(normal_windows), dtype=int), np.ones(n, dtype=int)], axis=0
    )
    all_seg_ids = np.concatenate([normal_seg_ids, normal_seg_ids[idx]], axis=0)
    perm = rng.permutation(len(all_windows))
    return all_windows[perm], all_labels[perm], all_seg_ids[perm]


def evaluate_detector(
    model: TrafficAutoencoder,
    thresholds: dict,
    test_windows: np.ndarray,
    test_seg_ids: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Compute classification metrics and save EDA plots."""
    OUTPUTS_EDA_DIR.mkdir(parents=True, exist_ok=True)
    errors = compute_errors(model, test_windows)
    seg_thresholds = np.array([_segment_threshold(thresholds, s) for s in test_seg_ids])
    preds = (errors > seg_thresholds).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="binary", zero_division=0
    )
    try:
        roc_auc = float(roc_auc_score(test_labels, errors))
    except ValueError:
        roc_auc = float("nan")

    fpr = float(np.mean((preds == 1) & (test_labels == 0)))

    # Reconstruction error distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors[test_labels == 0], bins=40, alpha=0.6, label="Normal", color="steelblue")
    ax.hist(errors[test_labels == 1], bins=40, alpha=0.6, label="Incident", color="crimson")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Count")
    ax.set_title("LSTM Autoencoder — Reconstruction Error Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_EDA_DIR / "lstm_ae_error_distribution.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fpr_curve, tpr_curve, _ = roc_curve(test_labels, errors)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_curve, tpr_curve, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("LSTM Autoencoder — ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_EDA_DIR / "lstm_ae_roc_curve.png", dpi=150)
    plt.close(fig)

    # Segment-level error heatmap (top segments by volatility)
    seg_volatility: dict[str, float] = {}
    for seg in np.unique(test_seg_ids):
        seg_volatility[str(seg)] = float(np.std(errors[test_seg_ids == seg]))
    top_segs = sorted(seg_volatility, key=seg_volatility.get, reverse=True)[:15]
    heatmap = np.array([errors[test_seg_ids == s][:20] for s in top_segs])
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(top_segs)))
    ax.set_yticklabels(top_segs)
    ax.set_xlabel("Window index (sample)")
    ax.set_title("Segment-Level Reconstruction Error Heatmap (Top 15 Volatile)")
    fig.colorbar(im, ax=ax, label="MSE")
    fig.tight_layout()
    fig.savefig(OUTPUTS_EDA_DIR / "lstm_ae_segment_heatmap.png", dpi=150)
    plt.close(fig)

    metrics = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(roc_auc, 4),
        "false_positive_rate": round(fpr, 4),
        "n_test_windows": int(len(test_windows)),
        "n_incidents": int(test_labels.sum()),
    }
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def main() -> None:
    """Train LSTM autoencoder, calibrate thresholds, evaluate, and demo inference."""
    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_EDA_DIR.mkdir(parents=True, exist_ok=True)

    normal_df = load_normal_traffic()
    windows, seg_ids, _, scaler = create_sliding_windows(normal_df, fit_scaler=True)

    split_idx = int(len(windows) * 0.8)
    train_windows = windows[:split_idx]
    val_windows = windows[split_idx:]
    train_seg_ids = seg_ids[:split_idx]
    val_seg_ids = seg_ids[split_idx:]

    logger.info("Training windows: %s | Validation windows: %s", len(train_windows), len(val_windows))
    model, train_info = train_autoencoder(train_windows, val_windows)
    logger.info("Best validation loss: %.6f", train_info["best_val_loss"])

    val_errors = compute_errors(model, val_windows)
    thresholds = calibrate_thresholds(val_errors, val_seg_ids)
    save_artifacts(model, scaler, thresholds)

    # Evaluation with synthetic incident labels
    eval_windows, eval_labels, eval_seg_ids = synthesize_incident_windows(
        val_windows, val_seg_ids, n_incidents=min(300, max(1, len(val_windows) // 2))
    )

    metrics = evaluate_detector(model, thresholds, eval_windows, eval_seg_ids, eval_labels)

    # Demo: single-segment detection
    sample_seg = str(normal_df[SEGMENT_ID_COL].iloc[0])
    sample_data = normal_df[normal_df[SEGMENT_ID_COL] == sample_seg].tail(WINDOW_SIZE)
    demo_result = detect_anomaly(sample_seg, sample_data)
    logger.info("Demo detect_anomaly(%s): %s", sample_seg, demo_result)

    # Demo: realtime monitor on two timesteps
    alerts: list[tuple[str, dict]] = []

    def _on_alert(seg: str, info: dict) -> None:
        alerts.append((seg, info))
        logger.warning("ALERT segment=%s error=%.4f threshold=%.4f", seg, info["error"], info["threshold"])

    demo_stream: list[pd.DataFrame] = []
    for _seg, grp in normal_df.groupby(SEGMENT_ID_COL):
        grp = grp.sort_values(TIMESTAMP_COL)
        for i in range(WINDOW_SIZE, min(WINDOW_SIZE + 4, len(grp))):
            demo_stream.append(grp.iloc[i - WINDOW_SIZE : i].copy())
        break

    run_realtime_monitor(iter(demo_stream), _on_alert)

    print("\n=== LSTM Autoencoder Training Complete ===")
    print(f"Model:      {LSTM_AE_PATH}")
    print(f"Thresholds: {THRESHOLD_JSON}")
    print(f"Metrics:    {json.dumps(metrics, indent=2)}")
    print(f"Plots:      {OUTPUTS_EDA_DIR}/lstm_ae_*.png")


if __name__ == "__main__":
    main()
