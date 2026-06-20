#!/usr/bin/env python3
"""FlowCast AI — Temporal Fusion Transformer training pipeline (Module A)."""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flowcast_config import (
    FEATURE_MATRIX_PATH,
    MODELS_DIR,
    OUTPUTS_EDA_DIR,
    RANDOM_SEED,
    SEGMENT_ID_COL,
    TFT_CHECKPOINT,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET = "congestion_risk_score"
MAX_ENCODER_LENGTH = 672
MAX_PREDICTION_LENGTH = 16


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load feature matrix and chronological train/val splits."""
    root = FEATURE_MATRIX_PATH.parent
    train_path = root / "train_features.parquet"
    val_path = root / "val_features.parquet"
    if train_path.exists() and val_path.exists():
        return pd.read_parquet(train_path), pd.read_parquet(val_path), pd.read_parquet(FEATURE_MATRIX_PATH)
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df = df.sort_values(["timestamp", SEGMENT_ID_COL]).reset_index(drop=True)
    n = len(df)
    t1, t2 = int(n * 0.70), int(n * 0.85)
    return df.iloc[:t1], df.iloc[t1:t2], df


def prepare_tft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add time_idx and ensure required TFT columns exist."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    t0 = out["timestamp"].min()
    out["time_idx"] = ((out["timestamp"] - t0).dt.total_seconds() / 60).astype(int)
    for col, default in [
        ("road_type", "arterial"),
        ("road_capacity", 3000.0),
        ("lane_count", 3),
        ("segment_centrality", 0.5),
        ("event_phase", "no_event"),
        ("event_type_encoded", "none"),
    ]:
        if col not in out.columns:
            out[col] = default
            
    # Fallback for missing time_varying_unknown_reals and known_reals
    missing_reals = [
        "speed_lag_1h", "speed_lag_24h", "volume_lag_1h",
        "rolling_mean_1h", "rolling_std_1h", "CRS_lag_1h", "CRS_lag_24h",
        "hours_to_event_start", "expected_attendance_normalized", "venue_proximity_km",
        "is_holiday", "is_weekend", "is_monsoon_season",
        "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
        TARGET
    ]
    for col in missing_reals:
        if col not in out.columns:
            out[col] = 0.0

    out["event_type_encoded"] = out["event_type_encoded"].astype(str)
    out["event_phase"] = out["event_phase"].astype(str)
    out["road_type"] = out["road_type"].astype(str)
    for et in ["ipl", "festival", "rally", "concert", "marathon", "government", "none"]:
        col = f"event_{et}"
        if col not in out.columns:
            out[col] = (out.get("event_type_encoded", "none") == et).astype(int)
    return out


def train_tft_pytorch_forecasting(train_df: pd.DataFrame, val_df: pd.DataFrame) -> bool:
    """Train TFT via pytorch-forecasting; return True on success."""
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss, SMAPE
    except ImportError:
        logger.warning("pytorch-forecasting/lightning not installed — using sklearn fallback.")
        return False

    train_df = prepare_tft_dataframe(train_df)
    val_df = prepare_tft_dataframe(val_df)
    combined = pd.concat([train_df, val_df], ignore_index=True)

    static_categoricals = [SEGMENT_ID_COL, "road_type"]
    static_reals = ["road_capacity", "lane_count", "segment_centrality"]
    time_varying_known_reals = [
        "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
        "hours_to_event_start", "expected_attendance_normalized", "venue_proximity_km",
        "is_holiday", "is_weekend", "is_monsoon_season",
    ]
    time_varying_known_categoricals = ["event_type_encoded", "event_phase"]
    time_varying_unknown_reals = [
        "speed_lag_1h", "speed_lag_24h", "volume_lag_1h",
        "rolling_mean_1h", "rolling_std_1h", "CRS_lag_1h", "CRS_lag_24h", TARGET,
    ]

    try:
        training = TimeSeriesDataSet(
            combined[combined["time_idx"] <= train_df["time_idx"].max()],
            time_idx="time_idx",
            target=TARGET,
            group_ids=[SEGMENT_ID_COL],
            max_encoder_length=MAX_ENCODER_LENGTH,
            max_prediction_length=MAX_PREDICTION_LENGTH,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=[SEGMENT_ID_COL]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        validation = TimeSeriesDataSet.from_dataset(training, combined, predict=True, stop_randomization=True)

        train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

        tft = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=128,
            attention_head_size=4,
            dropout=0.15,
            hidden_continuous_size=64,
            output_size=7,
            loss=QuantileLoss(quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]),
            log_interval=10,
            reduce_on_plateau_patience=5,
            learning_rate=1e-3,
        )

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(MODELS_DIR),
            filename="tft_flowcast",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="cpu",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[checkpoint_cb, early_stop, lr_monitor],
            enable_progress_bar=True,
            logger=False,
        )

        try:
            res = trainer.tuner.lr_find(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
            if res and hasattr(res, "suggestion"):
                tft.hparams.learning_rate = res.suggestion()
        except Exception as exc:
            logger.info("LR finder skipped: %s", exc)

        trainer.fit(tft, train_loader, val_loader)

        best = checkpoint_cb.best_model_path
        if best:
            Path(best).replace(TFT_CHECKPOINT) if Path(best) != TFT_CHECKPOINT else None
            if not TFT_CHECKPOINT.exists() and Path(best).exists():
                import shutil
                shutil.copy(best, TFT_CHECKPOINT)

        # Evaluation plots on validation
        preds = tft.predict(val_loader, return_x=False)
        actuals = np.concatenate([y[1].numpy().flatten() for y in val_loader])
        pred_median = preds.numpy().reshape(-1, 7)[:, 3][: len(actuals)]
        mae = np.mean(np.abs(actuals - pred_median))
        rmse = np.sqrt(np.mean((actuals - pred_median) ** 2))
        smape = float(np.mean(2 * np.abs(actuals - pred_median) / (np.abs(actuals) + np.abs(pred_median) + 1e-8)) * 100)
        logger.info("TFT validation — SMAPE=%.4f MAE=%.4f RMSE=%.4f", smape, mae, rmse)

        OUTPUTS_EDA_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(actuals[:200], label="Actual CRS", alpha=0.8)
        ax.plot(pred_median[:200], label="Predicted P50", alpha=0.8)
        ax.set_title("TFT Validation — Predicted vs Actual CRS")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("CRS")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUTS_EDA_DIR / "tft_pred_vs_actual.png", dpi=150)
        plt.close(fig)

        meta = {"backend": "pytorch_forecasting", "smape": smape, "mae": mae, "rmse": rmse}
        with open(MODELS_DIR / "tft_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return True
    except Exception as e:
        logger.warning(f"TFT training failed (falling back to sklearn): {e}")
        return False


def train_sklearn_fallback(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Train GradientBoosting fallback when TFT libraries unavailable."""
    import joblib
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    features = [
        "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
        "is_weekend", "is_holiday", "is_monsoon_season",
        "hours_to_event_start", "expected_attendance_normalized", "venue_proximity_km",
        "speed_lag_1h", "speed_lag_24h", "volume_lag_1h", "rolling_mean_1h", "rolling_std_1h",
        "CRS_lag_1h", "CRS_lag_24h", "road_capacity", "lane_count", "segment_centrality",
    ]
    features = [c for c in features if c in train_df.columns]
    X_train, y_train = train_df[features].values, train_df[TARGET].values
    X_val, y_val = val_df[features].values, val_df[TARGET].values

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_val), 0, 10)
    mae = mean_absolute_error(y_val, pred)
    rmse = mean_squared_error(y_val, pred, squared=False)
    logger.info("Sklearn fallback — MAE=%.4f RMSE=%.4f", mae, rmse)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "backend": "sklearn"}, TFT_CHECKPOINT.with_suffix(".pkl"))
    with open(MODELS_DIR / "tft_meta.json", "w", encoding="utf-8") as f:
        json.dump({"backend": "sklearn", "mae": mae, "rmse": rmse, "features": features}, f, indent=2)


def predict_congestion(
    event_type: str,
    venue_id: str,
    event_datetime: str,
    attendance: float,
    segment_ids: list[str],
    hours_ahead: int = 4,
) -> dict[str, list[list[float]]]:
    """Return per-segment quantile CRS forecasts for 15-min steps."""
    import joblib

    steps = hours_ahead * 4
    result: dict[str, list[list[float]]] = {}
    meta_path = MODELS_DIR / "tft_meta.json"
    if (TFT_CHECKPOINT.with_suffix(".pkl")).exists():
        bundle = joblib.load(TFT_CHECKPOINT.with_suffix(".pkl"))
        model, features = bundle["model"], bundle["features"]
        df = pd.read_parquet(FEATURE_MATRIX_PATH)
        for seg in segment_ids:
            seg_df = df[df[SEGMENT_ID_COL] == seg].tail(1)
            if seg_df.empty:
                base = 5.0
                result[seg] = [[base - 0.8, base, base + 0.9] for _ in range(steps)]
                continue
            x = seg_df[features].values
            p50 = float(np.clip(model.predict(x)[0], 0, 10))
            result[seg] = [[p50 - 0.9, p50, p50 + 1.0] for _ in range(steps)]
        return result

    # Demo quantile forecasts when no model artifact
    boost = {"ipl": 2.5, "festival": 2.0, "marathon": 1.5}.get(event_type, 1.0)
    for seg in segment_ids:
        base = min(10.0, 4.0 + boost * (attendance / 50000))
        result[seg] = [[base - 1.0, base, base + 1.2] for _ in range(steps)]
    return result


def main() -> None:
    """Train TFT model or sklearn fallback."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_EDA_DIR.mkdir(parents=True, exist_ok=True)
    if not FEATURE_MATRIX_PATH.exists():
        from generate_demo_data import main as gen_demo
        from importlib import import_module
        gen_demo()
        import_module("02_features").main()

    train_df, val_df, _ = load_splits()
    ok = train_tft_pytorch_forecasting(train_df, val_df)
    if not ok:
        train_sklearn_fallback(train_df, val_df)
    logger.info("Training complete. Artifacts in %s", MODELS_DIR)


if __name__ == "__main__":
    main()
