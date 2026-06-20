---
title: FlowCast AI - Traffic Congestion Predictor
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.30.0
app_file: dashboard/app.py
pinned: true
---

# FlowCast AI — Event-Driven Congestion Prediction

FlowCast AI is an event-driven traffic congestion prediction system built for **Bengaluru Traffic Police (BTP)** as part of Flipkart Gridlock 5.0. It forecasts congestion **two hours before it forms** by fusing planned-event forecasting (Temporal Fusion Transformer), unplanned incident detection (LSTM Autoencoder + DistilBERT NLP), and spatial propagation modelling (Graph Attention Network on Bengaluru's road graph).

Officers interact with a real-time Streamlit command centre featuring an interactive Folium map, RED/AMBER/GREEN alert levels, quantile confidence bands (P10/P50/P90), officer deployment briefs, and economic impact quantification in rupees. The system handles both **planned events** (IPL matches, festivals, marathons) and **unplanned incidents** (accidents, VIP convoys, waterlogging) through a unified Congestion Risk Score engine.

FlowCast AI runs entirely on **CPU** with graceful demo fallbacks — no external API keys required for the judge demo. Trained on Bengaluru traffic patterns with 50 road segments and 8,173+ real Astram event records. Deploy officers proactively, reduce unmanaged congestion losses by up to 35%, and align with India's Smart Cities Mission goals.

**Run locally:** `HF_DEMO_MODE=true streamlit run dashboard/app.py`
