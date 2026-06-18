"""NLP Incident Classification Engine — DistilBERT-inspired multilingual classifier."""
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

INCIDENT_CLASSES = [
    "accident_major", "accident_minor", "vehicle_breakdown",
    "waterlogging", "tree_fall", "road_closure", "vip_convoy",
    "protest_bandh", "congestion", "clear",
]

KEYWORD_MAP = {
    "accident_major": ["accident", "collision", "crash", "overturn", "fatal", "major accident"],
    "accident_minor": ["minor accident", "scratch", "small accident", "fender"],
    "vehicle_breakdown": ["breakdown", "break down", "stuck", "not moving", "off road", "gear",
                          "clutch", "starting problem", "mechanic", "bmtc", "bus off"],
    "waterlogging": ["water logging", "waterlogging", "flooded", "drainage", "underpass water"],
    "tree_fall": ["tree fall", "tree fallen", "tree down", "mar bidd", "ಮರ"],
    "road_closure": ["road closed", "closure", "barricade", "blocked", "diversion"],
    "vip_convoy": ["vip", "convoy", "cm office", "minister", "protocol"],
    "protest_bandh": ["protest", "bandh", "rally", "dharna", "strike"],
    "congestion": ["slow moving", "traffic jam", "congestion", "heavy traffic", "gridlock"],
}

KANNADA_PATTERNS = {
    "vehicle_breakdown": ["ನಿಂತ", "ಕೆಟ್ಟ", "ಬ್ರೇಕ್"],
    "tree_fall": ["ಮರ", "ಬಿದ್ದ"],
    "congestion": ["ನಿಧಾನ", "ಟ್ರಾಫಿಕ್"],
}


class NLPIncidentClassifier:
    """Multilingual incident classifier for English + Kannada transliterated text."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False

    def _clean_text(self, text: str) -> str:
        if not text or text == "NULL":
            return ""
        text = re.sub(r"\[LOCATION\]|\[PERSON\]|\[PHONE\]", " ", text)
        text = re.sub(r"[^\w\s\u0C80-\u0CFF]", " ", text.lower())
        return " ".join(text.split())

    def _keyword_classify(self, text: str) -> tuple[str, float]:
        text_lower = text.lower()
        scores = {}
        for cls, keywords in KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[cls] = score
        for cls, patterns in KANNADA_PATTERNS.items():
            for pat in patterns:
                if pat in text:
                    scores[cls] = scores.get(cls, 0) + 2

        if not scores or max(scores.values()) == 0:
            return "congestion", 0.55
        best = max(scores, key=scores.get)
        conf = min(0.95, 0.6 + scores[best] * 0.1)
        return best, conf

    def _extract_entities(self, text: str, address: str = "") -> dict:
        combined = f"{text} {address}".lower()
        roads = re.findall(
            r"(?:road|rd|highway|ring road|orr|main road|cross|circle|junction|flyover|layout)[\w\s]*",
            combined, re.I,
        )
        junctions = re.findall(r"[\w\s]+(?:junction|circle|cross|jn|flyover)", combined, re.I)
        severity_words = []
        for word in ["emergency", "major", "minor", "slow", "blocked", "closed", "heavy"]:
            if word in combined:
                severity_words.append(word)
        return {
            "extracted_road": roads[0].strip()[:100] if roads else None,
            "extracted_junction": junctions[0].strip()[:100] if junctions else None,
            "severity_words": ", ".join(severity_words) if severity_words else None,
        }

    def train(self, df) -> dict:
        texts, labels = [], []
        for _, row in df.iterrows():
            desc = self._clean_text(str(row.get("description", "")))
            if not desc:
                continue
            cls, _ = self._keyword_classify(desc)
            cause = str(row.get("event_cause", ""))
            if cause == "accident":
                cls = "accident_major" if "major" in desc else "accident_minor"
            elif cause in KEYWORD_MAP or cause.replace("_", "") in desc:
                cls_map = {
                    "vehicle_breakdown": "vehicle_breakdown",
                    "water_logging": "waterlogging",
                    "tree_fall": "tree_fall",
                    "congestion": "congestion",
                }
                cls = cls_map.get(cause, cls)
            texts.append(desc)
            labels.append(cls)

        if len(texts) < 50:
            return {"samples": len(texts), "accuracy_pct": 0, "note": "insufficient text data"}

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=500, random_state=42, C=1.0)),
        ])
        self.pipeline.fit(texts, labels)
        self.is_trained = True

        acc = float(self.pipeline.score(texts, labels) * 100)
        joblib.dump(self.pipeline, self.models_dir / "nlp_classifier.joblib")
        return {"samples": len(texts), "accuracy_pct": round(acc, 1), "classes": len(set(labels))}

    def load(self) -> bool:
        path = self.models_dir / "nlp_classifier.joblib"
        if path.exists():
            self.pipeline = joblib.load(path)
            self.is_trained = True
            return True
        return False

    def classify(self, text: str, address: str = "") -> dict:
        cleaned = self._clean_text(text)
        entities = self._extract_entities(cleaned, address)

        if self.is_trained and self.pipeline and cleaned:
            try:
                cls = self.pipeline.predict([cleaned])[0]
                proba = self.pipeline.predict_proba([cleaned])[0]
                conf = float(max(proba))
            except Exception:
                cls, conf = self._keyword_classify(cleaned)
        else:
            cls, conf = self._keyword_classify(cleaned or address)

        return {
            "classified_type": cls,
            "confidence": round(conf, 3),
            "raw_text": text[:500] if text else "",
            **entities,
        }
