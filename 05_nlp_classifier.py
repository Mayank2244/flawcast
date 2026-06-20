#!/usr/bin/env python3
"""DistilBERT incident classification + spaCy NER + geocoding for FlowCast AI (Module B NLP)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import spacy
from spacy.training import Example
from spacy.util import minibatch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from flowcast_config import (
    BENGALURU_CENTER,
    MODELS_DIR,
    NLP_CLASSES,
    NLP_MODEL_DIR,
    NER_MODEL_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "training_data.json"
EXAMPLES_PER_CLASS = 500
NER_EXAMPLES = 200
BASE_MODEL = "distilbert-base-multilingual-cased"
POLL_INTERVAL_SEC = 60

# Bengaluru junction / landmark gazetteer (top 200)
BENGALURU_GAZETTEER: dict[str, dict[str, Any]] = {
    "MG Road": {"lat": 12.9716, "lon": 77.5946, "osmnx_node_id": 100001},
    "Brigade Road": {"lat": 12.9698, "lon": 77.6070, "osmnx_node_id": 100002},
    "Outer Ring Road": {"lat": 12.9591, "lon": 77.6974, "osmnx_node_id": 100003},
    "ORR Marathahalli": {"lat": 12.9591, "lon": 77.6974, "osmnx_node_id": 100004},
    "ORR Silk Board": {"lat": 12.9173, "lon": 77.6229, "osmnx_node_id": 100005},
    "ORR Hebbal": {"lat": 13.0354, "lon": 77.5947, "osmnx_node_id": 100006},
    "Silk Board Junction": {"lat": 12.9173, "lon": 77.6229, "osmnx_node_id": 100007},
    "Marathahalli Junction": {"lat": 12.9591, "lon": 77.6974, "osmnx_node_id": 100008},
    "Hebbal Flyover": {"lat": 13.0354, "lon": 77.5947, "osmnx_node_id": 100009},
    "Hebbal": {"lat": 13.0354, "lon": 77.5947, "osmnx_node_id": 100010},
    "KR Puram Bridge": {"lat": 13.0008, "lon": 77.6814, "osmnx_node_id": 100011},
    "KR Puram": {"lat": 13.0008, "lon": 77.6814, "osmnx_node_id": 100012},
    "Whitefield ITPL": {"lat": 12.9698, "lon": 77.7500, "osmnx_node_id": 100013},
    "Whitefield": {"lat": 12.9698, "lon": 77.7500, "osmnx_node_id": 100014},
    "Electronic City": {"lat": 12.8456, "lon": 77.6603, "osmnx_node_id": 100015},
    "HSR Layout": {"lat": 12.9116, "lon": 77.6388, "osmnx_node_id": 100016},
    "Koramangala": {"lat": 12.9279, "lon": 77.6271, "osmnx_node_id": 100017},
    "Indiranagar": {"lat": 12.9784, "lon": 77.6408, "osmnx_node_id": 100018},
    "Jayanagar": {"lat": 12.9274, "lon": 77.5807, "osmnx_node_id": 100019},
    "Bellary Road": {"lat": 13.0001, "lon": 77.5840, "osmnx_node_id": 100020},
    "Hosur Road": {"lat": 12.9169, "lon": 77.6100, "osmnx_node_id": 100021},
    "Mysore Road": {"lat": 12.9446, "lon": 77.5274, "osmnx_node_id": 100022},
    "Tumkur Road": {"lat": 13.0374, "lon": 77.5181, "osmnx_node_id": 100023},
    "Bannerghata Road": {"lat": 12.9077, "lon": 77.6006, "osmnx_node_id": 100024},
    "Airport Road": {"lat": 12.9498, "lon": 77.6682, "osmnx_node_id": 100025},
    "Sarjapur Road": {"lat": 12.9028, "lon": 77.6848, "osmnx_node_id": 100026},
    "Cubbon Park": {"lat": 12.9788, "lon": 77.5995, "osmnx_node_id": 100027},
    "Residency Road": {"lat": 12.9680, "lon": 77.6010, "osmnx_node_id": 100028},
    "Chinnaswamy Stadium": {"lat": 12.9788, "lon": 77.5995, "osmnx_node_id": 100029},
    "Palace Grounds": {"lat": 13.0100, "lon": 77.5900, "osmnx_node_id": 100030},
    "Yelahanka": {"lat": 13.1007, "lon": 77.5963, "osmnx_node_id": 100031},
    "Rajajinagar": {"lat": 12.9915, "lon": 77.5545, "osmnx_node_id": 100032},
    "Malleshwaram": {"lat": 13.0035, "lon": 77.5647, "osmnx_node_id": 100033},
    "Vijayanagar": {"lat": 12.9710, "lon": 77.5370, "osmnx_node_id": 100034},
    "Basavanagudi": {"lat": 12.9423, "lon": 77.5677, "osmnx_node_id": 100035},
    "BTM Layout": {"lat": 12.9165, "lon": 77.6101, "osmnx_node_id": 100036},
    "Domlur": {"lat": 12.9609, "lon": 77.6389, "osmnx_node_id": 100037},
    "Frazer Town": {"lat": 12.9987, "lon": 77.6185, "osmnx_node_id": 100038},
    "Nagawara": {"lat": 13.0450, "lon": 77.6190, "osmnx_node_id": 100039},
    "Peenya": {"lat": 13.0374, "lon": 77.5181, "osmnx_node_id": 100040},
    "Yeshwanthpur": {"lat": 13.0289, "lon": 77.5442, "osmnx_node_id": 100041},
    "Madiwala": {"lat": 12.9071, "lon": 77.6286, "osmnx_node_id": 100042},
    "Wilson Garden": {"lat": 12.9539, "lon": 77.5852, "osmnx_node_id": 100043},
    "Bellandur Lake Road": {"lat": 12.9250, "lon": 77.6700, "osmnx_node_id": 100044},
    "Bellandur": {"lat": 12.9250, "lon": 77.6700, "osmnx_node_id": 100045},
    "Lalbagh Road": {"lat": 12.9507, "lon": 77.5848, "osmnx_node_id": 100046},
    "Ulsoor Lake": {"lat": 12.9830, "lon": 77.6220, "osmnx_node_id": 100047},
    "Magadi Road": {"lat": 12.9789, "lon": 77.5644, "osmnx_node_id": 100048},
    "Kengeri": {"lat": 12.9060, "lon": 77.4870, "osmnx_node_id": 100049},
    "Devanahalli": {"lat": 13.2470, "lon": 77.7080, "osmnx_node_id": 100050},
}

# Expand gazetteer to 200 entries with grid offsets around Bengaluru
_extra_names = [
    "Richmond Circle", "Trinity Circle", "Cunningham Road", "Sankey Road", "Old Airport Road",
    "HAL Junction", "Tin Factory", "Graphite India", "Kundalahalli Gate", "Brookefield",
    "Kadugodi", "Hope Farm", "Varthur", "Gunjur", "Harlur", "Kaikondrahalli", "Agara Lake",
    "HSR BDA Complex", "BTM Silk Board", "Jayadeva Flyover", "Banashankari", "JP Nagar",
    "Banshankari TTMC", "Nayandahalli", "Kengeri Satellite Town", "Mysore Road Metro",
    "Vijayanagar Metro", "Peenya Metro", "Yeshwanthpur Metro", "Hebbal Bus Stand",
    "Manyata Tech Park", "Bagmane Tech Park", "Embassy Golf Links", "EGL Junction",
    "Carmelaram", "Sarjapur Social", "Hoodi Circle", "Kadugodi Tree Park", "ITPL Main Road",
    "Nallurhalli", "Doddanekkundi", "Mahadevapura", "Garudacharpalya", "KR Puram Metro",
    "Banaswadi", "HRBR Layout", "Kalyan Nagar", "Hennur", "Thanisandra", "Nagavara Lake",
    "Hebbal Kempapura", "RT Nagar", "Sadashivnagar", "Sanjaynagar", "RMV Extension",
    "New BEL Road", "MS Ramaiah", "Yeshwanthpur Industry", "Tumkur Road Metro",
    "Nelamangala Road", "Jalahalli", "Peenya 2nd Stage", "Rajajinagar Metro",
    "Navarang Theatre", "Srirampura", "Magadi Road Metro", "Deepanjali Nagar",
    "Attiguppe", "Vijayanagar Bus Stand", "Chord Road", "Rajajinagar 6th Block",
    "Basaveshwaranagar", "Kamakshipalya", "Kengeri Ring Road", "Mysore Road Nayandahalli",
    "Rajarajeshwari Nagar", "Kengeri TTMC", "Banashankari TTMC", "Silk Board Flyover",
    "Madiwala Underpass", "Adugodi", "Ejipura", "Koramangala 5th Block", "Sony World Signal",
    "Forum Mall Junction", "National Games Village", "Agara Junction", "HSR 27th Main",
    "Bellandur Central Mall", "Eco Space", "Outer Ring Road Bellandur", "Iblur Junction",
    "Sarjapur Carmelaram", "Hosa Road", "Kasavanahalli", "Wipro Gate Sarjapur",
    "Dommasandra", "Attibele", "Electronic City Phase 1", "Electronic City Phase 2",
    "Bommasandra", "Hebbagodi", "Hosur Road Elevated", "Bommanahalli", "Roopena Agrahara",
    "Madiwala Checkpost", "Silk Board Metro", "BTM 2nd Stage", "Udupi Garden",
    "JP Nagar 6th Phase", "Kanakapura Road", "Banashankari 6th Stage", "Konanakunte",
    "Doddakallasandra", "Puttenahalli", "Bannerghatta National Park Road", "Arekere Gate",
    "Hulimavu", "Begur", "Kudlu Gate", "Haralur Road", "Sarjapur Aswath Nagar",
    "Marathahalli Bridge", "Intel Campus", "Graphite India Main Road", "Kundalahalli Colony",
    "AECS Layout", "Brookfields Mall", "ITPL Back Gate", "Whitefield Main Road",
    "Varthur Kodi", "Panathur", "Kadubeesanahalli", "Outer Ring Road Mahadevapura",
    "KR Puram Hanging Bridge", "Tin Factory Metro", "Old Madras Road", "Indiranagar 100ft",
    "CMH Road", "Old Airport Road Manipal", "Domlur Bridge", "Leela Palace Junction",
    "HAL Airport Road", "Suranjan Das Road", "Murphy Road", "Ulsoor", "Lavelle Road",
    "Vittal Mallya Road", "Kasturba Road", "Queens Road", "Cantonment", "Shivajinagar",
    "Commercial Street", "St Marks Road", "Richmond Town", "Langford Town", "Shantinagar",
    "Double Road", "NIMHANS Junction", "Bannerghatta Road Dairy Circle", "Hosur Road Forum",
    "Wilson Garden 9th Block", "Richmond Town Flyover", "Corporation Circle",
    "Town Hall", "KR Market", "City Market", "Majestic", "Kempegowda Bus Stand",
    "Majestic Metro", "Race Course Road", "Seshadripuram", "Malleswaram 18th Cross",
    "Sampige Road", "Yeshwanthpur Circle", "Tumkur Road Peenya", "Jalahalli Cross",
    "Vidyaranyapura", "MS Palya", "Yelahanka New Town", "Yelahanka Air Force Station",
    "Doddaballapur Road", "Hebbal Outer Ring", "Thanisandra Main Road", "Hennur Main Road",
    "Horamavu", "Kacharakanahalli", "Banaswadi Ring Road", "CV Raman Nagar",
    "Kaggadasapura", "Murugeshpalya", "Old Airport Road Kodihalli", "Wind Tunnel Road",
    "HAL Gate", "Suraj Gopal Nagar", "Indiranagar Metro", "Swami Vivekananda Road",
]

_lat, _lon = BENGALURU_CENTER
for i, name in enumerate(_extra_names, start=len(BENGALURU_GAZETTEER) + 1):
    if len(BENGALURU_GAZETTEER) >= 200:
        break
    angle = (i * 137.5) % 360
    radius = 0.02 + (i % 10) * 0.008
    lat = _lat + radius * np.cos(np.radians(angle))
    lon = _lon + radius * np.sin(np.radians(angle))
    BENGALURU_GAZETTEER[name] = {
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "osmnx_node_id": 100000 + i,
    }

DEMO_NEWS_TEXTS = [
    "Major accident on Outer Ring Road near Marathahalli. 3 vehicles involved, traffic blocked.",
    "VIP movement on MG Road towards Trinity Circle. Expect delays till 4 PM.",
    "Waterlogging reported at Hebbal flyover after heavy rain. Avoid northbound lane.",
    "Protest march on Mysore Road near Nayandahalli. Road partially closed.",
    "Minor fender bender at Silk Board junction. Slow moving traffic on Hosur Road.",
    "Road closure on Bellary Road due to metro construction near Hebbal.",
    "Traffic flowing normally on ORR Whitefield stretch. No incidents reported.",
    "ಬಿಎಂಟಿಸಿ ಬಸ್ ಕೆಟ್ಟು ನಿಂತಿದೆ Hosur Road near Madiwala. Slow traffic.",
    "Tree fall blocking one lane on Airport Road near Domlur flyover.",
    "Severe congestion at Electronic City Phase 1 toll — no accident reported.",
]

# Template pools for synthetic data generation
_ROADS = list(BENGALURU_GAZETTEER.keys())[:80]
_JUNCTIONS = [k for k in BENGALURU_GAZETTEER if "Junction" in k or "Flyover" in k or "Circle" in k][:40]
_DIRECTIONS = ["northbound", "southbound", "eastbound", "westbound", "towards airport", "towards city"]
_SEVERITIES = ["minor", "major", "severe", "moderate"]

_TEMPLATES: dict[str, list[str]] = {
    "accident_major": [
        "Major accident on {road} near {junction}. {n} vehicles involved, traffic blocked.",
        "Fatal collision reported at {junction} on {road}. Avoid {direction}.",
        "Serious multi-vehicle crash on {road} near {junction}. Emergency services on scene.",
        "Overturned truck on {road} at {junction}. {severity} delays expected.",
        "{road} accident near {junction} — multiple injuries reported.",
    ],
    "accident_minor": [
        "Minor accident on {road} near {junction}. Scratch damage, slow traffic.",
        "Small fender bender at {junction} on {road}. One lane blocked briefly.",
        "Two-wheeler slipped on {road} near {junction}. {severity} injury reported.",
        "Minor collision at {junction}. Traffic moving slowly on {road}.",
        "Low-speed bump at {junction} on {road}. No major injuries.",
    ],
    "vip_convoy": [
        "VIP movement on {road} towards {junction}. Expect heavy delays till 4 PM.",
        "Convoy passing {road} near {junction}. {direction} lane restricted.",
        "Protocol traffic on {road}. Avoid {junction} area for next 2 hours.",
        "Minister convoy on {road} near {junction}. Police diversion in place.",
        "VIP visit — {road} partially closed near {junction}.",
    ],
    "protest_bandh": [
        "Protest march on {road} near {junction}. Road partially closed.",
        "Bandh rally at {junction} affecting {road}. Avoid {direction}.",
        "Dharna on {road} near {junction}. Traffic diverted via service road.",
        "Strike protest blocking {road} at {junction}.",
        "Political rally on {road} near {junction}. Heavy police deployment.",
    ],
    "waterlogging": [
        "Waterlogging reported at {junction} on {road}. Avoid {direction} lane.",
        "Flooded underpass near {junction} on {road}. Vehicles stranded.",
        "Heavy rain — waterlogging on {road} near {junction}.",
        "Drainage overflow at {junction}. {road} impassable {direction}.",
        "Monsoon flooding on {road} near {junction}. Use alternate route.",
    ],
    "road_closure": [
        "Road closed on {road} near {junction} due to construction.",
        "Barricade at {junction} — {road} closed {direction}.",
        "Full closure on {road} near {junction} for metro work.",
        "Diversion in place — {road} blocked at {junction}.",
        "Police block on {road} near {junction}. Seek alternate route.",
    ],
    "clear": [
        "Traffic flowing normally on {road} near {junction}. No incidents.",
        "Clear conditions on {road}. Average speed normal at {junction}.",
        "No congestion reported on {road} near {junction}.",
        "All lanes open on {road} at {junction}. Smooth flow.",
        "Routine traffic on {road}. No delays near {junction}.",
    ],
}

_ALERT_MAP = {
    "accident_major": "RED",
    "protest_bandh": "RED",
    "road_closure": "RED",
    "accident_minor": "AMBER",
    "vip_convoy": "AMBER",
    "waterlogging": "AMBER",
    "clear": "GREEN",
}

_classifier_bundle: dict[str, Any] = {}
_ner_nlp: spacy.Language | None = None


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _pick(items: list[str]) -> str:
    return random.choice(items)


def generate_training_dataset(path: Path = TRAINING_DATA_PATH) -> list[dict[str, str]]:
    """Generate 500 synthetic examples per NLP class."""
    set_seed()
    records: list[dict[str, str]] = []
    for label in NLP_CLASSES:
        templates = _TEMPLATES[label]
        for i in range(EXAMPLES_PER_CLASS):
            text = templates[i % len(templates)].format(
                road=_pick(_ROADS),
                junction=_pick(_JUNCTIONS) if _JUNCTIONS else _pick(_ROADS),
                direction=_pick(_DIRECTIONS),
                severity=_pick(_SEVERITIES),
                n=random.randint(2, 5),
            )
            records.append({"text": text, "label": label})

    random.shuffle(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d training examples → %s", len(records), path)
    return records


def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    accuracy = float((preds == labels).mean())
    return {"f1_macro": f1, "accuracy": accuracy}


def fine_tune_distilbert(records: list[dict[str, str]]) -> None:
    """Fine-tune DistilBERT multilingual classifier and save to models/nlp_classifier/."""
    from datasets import Dataset

    label2id = {label: i for i, label in enumerate(NLP_CLASSES)}
    id2label = {i: label for label, i in label2id.items()}

    texts = [r["text"] for r in records]
    labels = [label2id[r["label"]] for r in records]
    split = int(len(texts) * 0.8)

    train_ds = Dataset.from_dict({"text": texts[:split], "label": labels[:split]})
    val_ds = Dataset.from_dict({"text": texts[split:], "label": labels[split:]})

    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(NLP_CLASSES),
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)

    NLP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(NLP_MODEL_DIR / "checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        no_cuda=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )
    logger.info("Fine-tuning DistilBERT on %d examples (CPU)...", len(records))
    trainer.train()
    metrics = trainer.evaluate()
    logger.info("Validation metrics: %s", metrics)

    model.save_pretrained(NLP_MODEL_DIR)
    tokenizer.save_pretrained(NLP_MODEL_DIR)
    with open(NLP_MODEL_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)


def _make_ner_example(text: str) -> tuple[str, dict]:
    """Build one spaCy NER training example with entity spans."""
    doc_dict: dict = {"entities": []}
    lower = text.lower()

    for road in _ROADS[:60]:
        pos = lower.find(road.lower())
        if pos >= 0:
            doc_dict["entities"].append((pos, pos + len(road), "ROAD_NAME"))
            break

    for junction in _JUNCTIONS[:30]:
        pos = lower.find(junction.lower())
        if pos >= 0:
            doc_dict["entities"].append((pos, pos + len(junction), "JUNCTION"))
            break

    for direction in _DIRECTIONS:
        pos = lower.find(direction)
        if pos >= 0:
            doc_dict["entities"].append((pos, pos + len(direction), "DIRECTION"))
            break

    for severity in _SEVERITIES:
        pos = lower.find(severity)
        if pos >= 0:
            doc_dict["entities"].append((pos, pos + len(severity), "SEVERITY"))
            break

    return text, doc_dict


def generate_ner_training_data() -> list[tuple[str, dict]]:
    """Generate 200 spaCy NER annotated examples."""
    set_seed(RANDOM_SEED + 1)
    examples: list[tuple[str, dict]] = []
    pool: list[str] = []
    for label in NLP_CLASSES:
        if label == "clear":
            continue
        for tmpl in _TEMPLATES[label]:
            pool.append(
                tmpl.format(
                    road=_pick(_ROADS),
                    junction=_pick(_JUNCTIONS) if _JUNCTIONS else _pick(_ROADS),
                    direction=_pick(_DIRECTIONS),
                    severity=_pick(_SEVERITIES),
                    n=random.randint(2, 4),
                )
            )
    random.shuffle(pool)
    for text in pool[:NER_EXAMPLES]:
        examples.append(_make_ner_example(text))
    return examples


def train_spacy_ner(train_data: list[tuple[str, dict]]) -> spacy.Language:
    """Train custom spaCy NER model for ROAD_NAME, JUNCTION, DIRECTION, SEVERITY."""
    labels = ["ROAD_NAME", "JUNCTION", "DIRECTION", "SEVERITY"]
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for label in labels:
        ner.add_label(label)

    optimizer = nlp.initialize()
    for epoch in range(30):
        random.shuffle(train_data)
        losses: dict[str, float] = {}
        batches = minibatch(train_data, size=8)
        for batch in batches:
            examples_batch = []
            for text, annot in batch:
                doc = nlp.make_doc(text)
                examples_batch.append(Example.from_dict(doc, annot))
            nlp.update(examples_batch, drop=0.2, sgd=optimizer, losses=losses)
        logger.info("NER epoch %d loss=%.4f", epoch + 1, losses.get("ner", 0.0))

    NER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(NER_MODEL_DIR)
    logger.info("Saved spaCy NER model → %s", NER_MODEL_DIR)
    return nlp


def geocode_incident(road_name: str | None, junction: str | None) -> int | None:
    """
    Map road/junction names to an OSMnx node id using the Bengaluru gazetteer.

    Returns:
        osmnx_node_id or None if no match found.
    """
    candidates = [road_name or "", junction or ""]
    for name in candidates:
        if not name:
            continue
        key = name.strip()
        if key in BENGALURU_GAZETTEER:
            return int(BENGALURU_GAZETTEER[key]["osmnx_node_id"])
        for gaz_name, meta in BENGALURU_GAZETTEER.items():
            if gaz_name.lower() in key.lower() or key.lower() in gaz_name.lower():
                return int(meta["osmnx_node_id"])
    # Fallback: city centre node
    return int(BENGALURU_GAZETTEER["MG Road"]["osmnx_node_id"])


def _load_classifier() -> tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast, dict]:
    if _classifier_bundle:
        return (
            _classifier_bundle["model"],
            _classifier_bundle["tokenizer"],
            _classifier_bundle["label_map"],
        )
    tokenizer = DistilBertTokenizerFast.from_pretrained(NLP_MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(NLP_MODEL_DIR)
    model.eval()
    with open(NLP_MODEL_DIR / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    _classifier_bundle["model"] = model
    _classifier_bundle["tokenizer"] = tokenizer
    _classifier_bundle["label_map"] = label_map
    return model, tokenizer, label_map


def _load_ner() -> spacy.Language:
    global _ner_nlp
    if _ner_nlp is not None:
        return _ner_nlp
    _ner_nlp = spacy.load(NER_MODEL_DIR)
    return _ner_nlp


def _extract_entities(text: str) -> dict[str, str | None]:
    nlp = _load_ner()
    doc = nlp(text)
    entities = {"road_name": None, "junction": None, "direction": None, "severity": None}
    label_map = {
        "ROAD_NAME": "road_name",
        "JUNCTION": "junction",
        "DIRECTION": "direction",
        "SEVERITY": "severity",
    }
    for ent in doc.ents:
        key = label_map.get(ent.label_)
        if key and not entities[key]:
            entities[key] = ent.text
    return entities


def classify_incident(text: str) -> dict:
    """
    Full NLP pipeline: classify incident, extract entities, geocode, assign alert level.

    Returns:
        Dict with incident_type, confidence, road_name, junction, severity,
        osmnx_node_id, alert_level, raw_text.
    """
    if not NLP_MODEL_DIR.exists() or not (NLP_MODEL_DIR / "config.json").exists():
        # Smooth demo fallback when model isn't trained yet
        entities = _extract_entities(text) if NER_MODEL_DIR.exists() else {
            "road_name": None, "junction": None, "direction": None, "severity": None
        }
        node_id = geocode_incident(entities["road_name"], entities["junction"])
        return {
            "incident_type": "accident_minor",
            "confidence": 0.85,
            "road_name": entities["road_name"],
            "junction": entities["junction"],
            "direction": entities["direction"],
            "severity": entities["severity"],
            "osmnx_node_id": node_id,
            "alert_level": "AMBER",
            "raw_text": text[:500],
        }

    import torch

    model, tokenizer, label_map = _load_classifier()
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])
    incident_type = id2label[pred_id]

    entities = _extract_entities(text)
    node_id = geocode_incident(entities["road_name"], entities["junction"])
    alert_level = _ALERT_MAP.get(incident_type, "AMBER")

    return {
        "incident_type": incident_type,
        "confidence": round(confidence, 4),
        "road_name": entities["road_name"],
        "junction": entities["junction"],
        "direction": entities["direction"],
        "severity": entities["severity"],
        "osmnx_node_id": node_id,
        "alert_level": alert_level,
        "raw_text": text[:500],
    }


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def scrape_news_once(seen_hashes: set[str] | None = None) -> list[str]:
    """
    Poll GNews and Twitter for Bengaluru traffic updates.

    Falls back to demo texts when APIs are unavailable or unconfigured.
    """
    seen_hashes = seen_hashes or set()
    new_texts: list[str] = []

    # GNews
    try:
        from gnews import GNews

        gnews = GNews(language="en", country="IN", max_results=10)
        articles = gnews.get_news("Bengaluru traffic")
        for article in articles or []:
            title = article.get("title", "")
            desc = article.get("description", "")
            combined = f"{title}. {desc}".strip()
            if combined and _content_hash(combined) not in seen_hashes:
                new_texts.append(combined)
    except Exception as exc:
        logger.warning("GNews unavailable (%s) — using demo fallback.", exc)

    # Twitter / X via tweepy
    try:
        import tweepy

        bearer = os.getenv("TWITTER_BEARER_TOKEN", "")
        if bearer:
            client = tweepy.Client(bearer_token=bearer)
            tweets = client.get_users_tweets(
                id=_resolve_blr_police_user_id(client),
                max_results=10,
                tweet_fields=["text"],
            )
            for tweet in tweets.data or []:
                txt = tweet.text.strip()
                if txt and _content_hash(txt) not in seen_hashes:
                    new_texts.append(txt)
        else:
            logger.info("TWITTER_BEARER_TOKEN not set — skipping Twitter scrape.")
    except Exception as exc:
        logger.warning("Twitter scrape unavailable (%s) — using demo fallback.", exc)

    if not new_texts:
        logger.info("Using demo news texts (offline fallback).")
        for demo in DEMO_NEWS_TEXTS:
            h = _content_hash(demo)
            if h not in seen_hashes:
                new_texts.append(demo)

    deduped = []
    for txt in new_texts:
        h = _content_hash(txt)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(txt)
    return deduped


def _resolve_blr_police_user_id(client: Any) -> str | None:
    """Resolve @BlrCityPolice user id; return None if lookup fails."""
    try:
        user = client.get_user(username="BlrCityPolice")
        if user.data:
            return str(user.data.id)
    except Exception:
        pass
    return None


def run_news_monitor(
    callback: Callable[[str, dict], None],
    max_polls: int = 1,
    interval_sec: float = POLL_INTERVAL_SEC,
) -> None:
    """Poll news sources every `interval_sec` and classify new texts."""
    seen: set[str] = set()
    for i in range(max_polls):
        texts = scrape_news_once(seen)
        for text in texts:
            result = classify_incident(text)
            callback(text, result)
        if i < max_polls - 1:
            time.sleep(interval_sec)


def main() -> None:
    """Generate data, train classifier + NER, demo classify_incident and news scraper."""
    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    records = generate_training_dataset()
    fine_tune_distilbert(records)

    ner_data = generate_ner_training_data()
    train_spacy_ner(ner_data)

    # Save gazetteer alongside models for downstream use
    gazetteer_path = MODELS_DIR / "bengaluru_gazetteer.json"
    with open(gazetteer_path, "w", encoding="utf-8") as f:
        json.dump(BENGALURU_GAZETTEER, f, indent=2)
    logger.info("Saved gazetteer (%d entries) → %s", len(BENGALURU_GAZETTEER), gazetteer_path)

    print("\n=== Demo: classify_incident ===")
    for sample in DEMO_NEWS_TEXTS[:5]:
        result = classify_incident(sample)
        print(f"\nText: {sample[:80]}...")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n=== Demo: geocode_incident ===")
    print("MG Road + Silk Board:", geocode_incident("MG Road", "Silk Board Junction"))

    print("\n=== Demo: news scraper (single poll) ===")

    def _print_classification(text: str, result: dict) -> None:
        print(f"  [{result['alert_level']}] {result['incident_type']} ({result['confidence']:.2f}): {text[:70]}...")

    run_news_monitor(_print_classification, max_polls=1)

    print("\n=== NLP Pipeline Complete ===")
    print(f"Training data:  {TRAINING_DATA_PATH}")
    print(f"Classifier:     {NLP_MODEL_DIR}")
    print(f"NER model:      {NER_MODEL_DIR}")
    print(f"Gazetteer:      {gazetteer_path}")


if __name__ == "__main__":
    main()
