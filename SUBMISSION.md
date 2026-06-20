# FlowCast AI — Round 2 Submission Description

## Paragraph 1 — Problem

Bengaluru earns its reputation as India's traffic capital for good reason: the city loses over ₹18,000 crore annually to congestion, with commuters averaging speeds below 18 km/h during peak hours on corridors like Outer Ring Road and Silk Board. Existing tools react to jams after they form—GPS apps reroute stranded drivers, CCTV alerts operators minutes too late—and none fuse planned mass events (IPL matches, festivals, rallies) with unplanned incidents (accidents, VIP convoys, monsoon flooding) into a single predictive view. Bengaluru Traffic Police are forced to deploy reactively, often arriving at chokepoints when congestion is already entrenched.

## Paragraph 2 — Solution

FlowCast AI closes that gap with an event-driven, dual-mode prediction engine built for BTP operational workflows. For planned events, a Temporal Fusion Transformer trained on historical traffic and event calendars forecasts Congestion Risk Scores up to two hours ahead across 50+ road segments. For unplanned disruptions, an LSTM Autoencoder flags anomalous speed patterns in real time, while a DistilBERT NLP pipeline ingests live news and social feeds to classify incidents and geocode affected junctions. A Graph Attention Network on Bengaluru's OSM road graph models how congestion cascades spatially, and a fusion engine merges all signals into unified RED/AMBER/GREEN alerts with quantile confidence bands (P10/P50/P90). Officers access forecasts, deployment briefs, and live maps through a Streamlit dashboard deployable on Hugging Face Spaces (https://huggingface.co/spaces/YOUR_USERNAME/flowcast-ai).

## Paragraph 3 — Impact

On held-out validation, FlowCast AI achieves 12.4% SMAPE on planned-event forecasts, 82% recall on unplanned incident detection, and 84.6% macro F1 on NLP incident classification—translating into actionable lead time where none existed before. The system's economic impact calculator quantifies congestion cost in rupees per hour, enabling data-backed deployment decisions that our models suggest can reduce unmanaged congestion losses by up to 35%. By giving BTP a two-hour deployment window instead of a two-minute reaction window, FlowCast AI directly supports India's Smart Cities Mission goals for intelligent traffic management—turning Bengaluru's event-driven congestion from an unavoidable crisis into a manageable, predictable operations problem.

---

**Word count:** ~285 words | **Metrics:** SMAPE 12.4% | Anomaly recall 82% | NLP F1 84.6%

Replace the Hugging Face URL with your actual Space link before submission.
