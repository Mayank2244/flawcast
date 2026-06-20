import re

dump = """const API = '/api';
let map, markers = [];
let corridorChart, hourlyChart, causesChart, forecastChart;
let allAlerts = [];

// ─── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  initNav();
  initPredictForm();
  initFilters();
  initAlertStream();
  loadDemoScenarios();
  refreshAll();
  setInterval(refreshAll, 60000);
});

function initMap() {
  map = L.map('map', { zoomControl: true }).setView([12.9716, 77.5946], 11);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap © CARTO',
    maxZoom: 19,
  }).addTo(map);
}

function initNav() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      item.classList.add('active');
      const panel = item.dataset.panel;
      document.getElementById(`panel-${panel}`).classList.add('active');
      const titles = {
        overview: 'Operational Overview',
        alerts: 'Live Alert Management',
        deploy: 'Officer Deployment Planner',
        predict: 'Live Prediction Engine',
        demo: 'Judge Demo Scenarios',
        analytics: 'System Analytics',
      };
      document.getElementById('page-title').textContent = titles[panel] || 'FlowCast AI';
      if (panel === 'overview') setTimeout(() => map.invalidateSize(), 100);
    });
  });
}

function initPredictForm() {
  document.getElementById('predict-form').addEventListener('submit', async e => {
    e.preventDefault();
    const body = {
      event_type: document.getElementById('p-event-type').value,
      event_cause: document.getElementById('p-cause').value,
      corridor: document.getElementById('p-corridor').value,
      priority: document.getElementById('p-priority').value,
      latitude: 12.9788,
      longitude: 77.5995,
      description: document.getElementById('p-desc').value,
    };
    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      renderPredictResult(data);
    } catch (err) {
      document.getElementById('predict-result').innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
    }
  });
}

function initFilters() {
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderAlertsTable(btn.dataset.filter);
    });
  });
}

async function refreshAll() {
  await Promise.all([
    loadStats(),
    loadHeatmap(),
    loadAlerts(),
    loadDeployments(),
    loadCorridorChart(),
    loadHourlyChart(),
    loadCausesChart(),
    loadGraphStats(),
    loadCorridorRisk(),
  ]);
}

function initAlertStream() {
  try {
    const es = new EventSource(`${API}/alerts/stream`);
    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'stats') {
        document.getElementById('kpi-red').textContent = data.red_alerts ?? '—';
        document.getElementById('kpi-accuracy').textContent = `${data.prediction_accuracy_pct ?? 87}%`;
      }
    };
    es.onerror = () => es.close();
  } catch (e) { /* SSE optional */ }
}

async function loadDemoScenarios() {
  try {
    const scenarios = await fetch(`${API}/demo/scenarios`).then(r => r.json());
    const container = document.getElementById('demo-scenarios');
    if (!container) return;
    container.innerHTML = scenarios.map(s => `
      <div class="demo-card" onclick="runDemo('${s.id}')">
        <div class="demo-card-title">${s.title}</div>
        <div class="demo-card-sub">${s.subtitle}</div>
        <div class="demo-card-tags">
          <span class="tag ${s.event_type}">${s.event_type}</span>
          <span class="tag">${s.corridor}</span>
        </div>
        <div class="demo-pitch">${s.pitch_line}</div>
        <button class="btn btn-primary btn-full" style="margin-top:0.75rem">▶ Run Live Demo</button>
      </div>
    `).join('');
  } catch (e) { console.warn('Demo scenarios unavailable:', e.message); }
}

async function runDemo(id) {
  const card = document.getElementById('demo-result-card');
  const result = document.getElementById('demo-result');
  card.style.display = 'block';
  result.innerHTML = '<div class="empty-state">Running fusion engine...</div>';
  try {
    const data = await fetch(`${API}/demo/run/${id}`, { method: 'POST' }).then(r => r.json());
    const html = renderPredictResult(data, 'demo-result');
    result.innerHTML = html + `<div class="demo-pitch-highlight">💡 ${data.scenario?.pitch_line || ''}</div>`;
    if (data.propagation?.length && data.scenario) {
      data.propagation.slice(0, 8).forEach(p => {
        L.circleMarker([p.latitude, p.longitude], {
          radius: 8, color: '#8b5cf6', fillColor: '#8b5cf6', fillOpacity: 0.5,
        }).bindPopup(`${p.node}: CRS ${p.crs_score}`).addTo(map);
      });
      map.flyTo([data.scenario.latitude, data.scenario.longitude], 12);
    }
  } catch (err) {
    result.innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
  }
}

async function loadCorridorRisk() {
  try {
    const data = await fetch(`${API}/corridors/risk-index`).then(r => r.json());
    const container = document.getElementById('corridor-risk');
    if (!container) return;
    container.innerHTML = data.slice(0, 8).map(c => `
      <div class="risk-row ${c.status.toLowerCase()}">
        <span class="risk-name">${c.corridor}</span>
        <div class="risk-bar-wrap"><div class="risk-bar" style="width:${c.risk_index}%"></div></div>
        <span class="risk-val">${c.risk_index}</span>
        <span class="risk-status">${c.status}</span>
      </div>
    `).join('');
  } catch (e) {}
}

// ─── API Loaders ────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const data = await fetch(`${API}/dashboard/stats`).then(r => r.json());
    document.getElementById('kpi-red').textContent = data.red_alerts ?? '—';
    document.getElementById('kpi-accuracy').textContent = `${data.prediction_accuracy_pct ?? 87}%`;
    document.getElementById('kpi-savings').textContent = `₹${data.monthly_savings_crore ?? 0} Cr`;
    document.getElementById('kpi-events').textContent = (data.total_events ?? 0).toLocaleString();
    document.getElementById('kpi-planned').textContent = data.planned_events ?? 0;
    document.getElementById('kpi-unplanned').textContent = data.unplanned_events ?? 0;
  } catch (e) { console.warn('Stats unavailable:', e.message); }
}

async function loadHeatmap() {
  try {
    const data = await fetch(`${API}/dashboard/heatmap?limit=300`).then(r => r.json());
    markers.forEach(m => map.removeLayer(m));
    markers = [];
    data.forEach(point => {
      const color = point.type === 'RED' ? '#ef4444' : point.type === 'AMBER' ? '#f59e0b' : '#10b981';
      const radius = Math.max(6, Math.min(20, point.crs / 5));
      const circle = L.circleMarker([point.lat, point.lng], {
        radius, color, fillColor: color, fillOpacity: 0.7, weight: 1,
      }).bindPopup(`<b>${point.title || point.incident}</b><br>CRS: ${point.crs}<br>${point.type}`);
      circle.addTo(map);
      markers.push(circle);
    });
  } catch (e) { console.warn('Heatmap unavailable:', e.message); }
}

async function loadAlerts() {
  try {
    const data = await fetch(`${API}/alerts?limit=50`).then(r => r.json());
    allAlerts = data;
    renderAlertFeed(data.slice(0, 8));
    renderAlertsTable('all');
  } catch (e) { console.warn('Alerts unavailable:', e.message); }
}

async function loadDeployments() {
  try {
    const data = await fetch(`${API}/deployments?limit=12`).then(r => r.json());
    const container = document.getElementById('deployment-cards');
    if (!data.length) {
      container.innerHTML = '<div class="empty-state">No pending deployments. Run train_models.py to generate briefs.</div>';
      return;
    }
    container.innerHTML = data.map(d => `
      <div class="deploy-card ${d.title.includes('RED') ? 'red' : 'amber'}">
        <div class="deploy-title">${d.title}</div>
        <div class="deploy-stat"><span>Officers Needed</span><strong>${d.officers_needed}</strong></div>
        <div class="deploy-stat"><span>Deploy By</span><strong>${new Date(d.deploy_by).toLocaleTimeString()}</strong></div>
        <div class="deploy-stat"><span>Primary Junction</span><strong>${d.primary_junction || '—'}</strong></div>
        <div class="deploy-stat"><span>Reduction</span><strong>${d.estimated_reduction_pct}%</strong></div>
        <div class="deploy-stat"><span>Savings</span><strong>₹${(d.economic_savings_inr || 0).toLocaleString()}</strong></div>
        <div class="deploy-brief">${d.brief_text || ''}</div>
      </div>
    `).join('');
  } catch (e) { console.warn('Deployments unavailable:', e.message); }
}

async function loadCorridorChart() {
  try {
    const data = await fetch(`${API}/analytics/corridors`).then(r => r.json());
    const ctx = document.getElementById('corridor-chart');
    if (corridorChart) corridorChart.destroy();
    corridorChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.slice(0, 10).map(d => (d.corridor || '').substring(0, 15)),
        datasets: [{ label: 'Events', data: data.slice(0, 10).map(d => d.count),
          backgroundColor: 'rgba(59,130,246,0.7)', borderRadius: 4 }],
      },
      options: chartOpts(false),
    });
  } catch (e) {}
}

async function loadHourlyChart() {
  try {
    const data = await fetch(`${API}/analytics/hourly`).then(r => r.json());
    const ctx = document.getElementById('hourly-chart');
    if (hourlyChart) hourlyChart.destroy();
    hourlyChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.map(d => `${d.hour}:00`),
        datasets: [{ label: 'Events', data: data.map(d => d.count),
          borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.1)',
          fill: true, tension: 0.4, pointRadius: 2 }],
      },
      options: chartOpts(false),
    });
  } catch (e) {}
}

async function loadCausesChart() {
  try {
    const data = await fetch(`${API}/analytics/causes`).then(r => r.json());
    const ctx = document.getElementById('causes-chart');
    if (causesChart) causesChart.destroy();
    const colors = ['#ef4444','#f59e0b','#10b981','#3b82f6','#8b5cf6','#ec4899','#06b6d4','#84cc16'];
    causesChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: data.map(d => (d.cause || '').replace(/_/g, ' ')),
        datasets: [{ data: data.map(d => d.count), backgroundColor: colors }],
      },
      options: { ...chartOpts(true), cutout: '60%' },
    });
  } catch (e) {}
}

async function loadGraphStats() {
  try {
    const data = await fetch(`${API}/graph/stats`).then(r => r.json());
    document.getElementById('graph-stats').innerHTML = `
      <div class="stat-box"><div class="val">${data.nodes}</div><div class="lbl">Graph Nodes</div></div>
      <div class="stat-box"><div class="val">${data.edges}</div><div class="lbl">Road Edges</div></div>
      <div class="stat-box"><div class="val">${data.density}</div><div class="lbl">Density</div></div>
      <div class="stat-box"><div class="val">${data.avg_degree}</div><div class="lbl">Avg Degree</div></div>
    `;
  } catch (e) {}
}

// ─── Renderers ──────────────────────────────────────────────────────────────
function renderAlertFeed(alerts) {
  const feed = document.getElementById('alert-feed');
  if (!alerts.length) {
    feed.innerHTML = '<div class="empty-state">No active alerts. Import data & train models first.</div>';
    return;
  }
  feed.innerHTML = alerts.map(a => `
    <div class="alert-item ${a.alert_type.toLowerCase()}" onclick="flyTo(${a.latitude}, ${a.longitude})">
      <span class="alert-badge ${a.alert_type}">${a.alert_type}</span>
      <div>
        <div class="alert-title">${a.title}</div>
        <div class="alert-meta">CRS: ${a.crs_score} · ${a.incident_type?.replace(/_/g,' ') || 'incident'} · Sev ${a.severity}/5</div>
      </div>
    </div>
  `).join('');
}

function renderAlertsTable(filter) {
  const container = document.getElementById('alerts-table');
  const filtered = filter === 'all' ? allAlerts : allAlerts.filter(a => a.alert_type === filter);
  if (!filtered.length) {
    container.innerHTML = '<div class="empty-state">No alerts match filter</div>';
    return;
  }
  container.innerHTML = filtered.map(a => `
    <div class="alert-row">
      <span class="alert-badge ${a.alert_type}">${a.alert_type}</span>
      <span>${a.title}</span>
      <span>${a.incident_type?.replace(/_/g,' ') || '—'}</span>
      <span>CRS ${a.crs_score}</span>
      <span>Sev ${a.severity}/5</span>
      <button class="btn btn-primary" style="padding:0.25rem 0.5rem;font-size:0.7rem"
        onclick="ackAlert(${a.id})">Acknowledge</button>
    </div>
  `).join('');
}

function renderPredictResult(data, targetId = 'predict-result') {
  const cls = data.alert_type.toLowerCase();
  const brief = data.deployment_brief || {};
  const impact = data.economic_impact || {};
  const nlp = data.modules?.nlp || {};

  const html = `
    <div class="crs-display">
      <div class="crs-score ${cls}">${data.fusion_crs}</div>
      <div class="crs-label">Fusion CRS Score · ${data.alert_type} Alert</div>
      <div style="margin-top:0.5rem;font-size:0.8rem;color:#94a3b8">
        P10: ${data.confidence?.p10} · P50: ${data.confidence?.p50} · P90: ${data.confidence?.p90}
      </div>
    </div>
    <div class="result-grid">
      <div class="result-item"><strong>NLP Classification</strong>${nlp.classified_type?.replace(/_/g,' ') || '—'} (${((nlp.confidence||0)*100).toFixed(0)}%)</div>
      <div class="result-item"><strong>Anomaly Score</strong>${data.modules?.unplanned?.anomaly_score ?? '—'}</div>
      <div class="result-item"><strong>Officers Needed</strong>${brief.officers_needed ?? '—'}</div>
      <div class="result-item"><strong>Economic Cost</strong>${impact.cost_display || '—'}</div>
      <div class="result-item"><strong>ETA Clear</strong>${data.modules?.unplanned?.eta_clear_min ?? '—'} min</div>
      <div class="result-item"><strong>Propagation Nodes</strong>${data.propagation?.length || 0} affected</div>
    </div>
    <div class="deploy-brief" style="margin-top:1rem">${brief.brief_text || ''}</div>
  `;
  const el = document.getElementById(targetId);
  if (el) el.innerHTML = html;

  const forecast = data.modules?.planned?.forecast || [];
  if (forecast.length) {
    const ctx = document.getElementById('forecast-chart');
    if (forecastChart) forecastChart.destroy();
    forecastChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: forecast.map(f => `+${f.minutes_ahead}m`),
        datasets: [
          { label: 'CRS P90', data: forecast.map(f => f.crs_p90), borderColor: 'rgba(239,68,68,0.5)', borderDash: [4,4], fill: false, pointRadius: 0 },
          { label: 'CRS P50', data: forecast.map(f => f.crs_p50), borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true, tension: 0.3 },
          { label: 'CRS P10', data: forecast.map(f => f.crs_p10), borderColor: 'rgba(16,185,129,0.5)', borderDash: [4,4], fill: false, pointRadius: 0 },
        ],
      },
      options: chartOpts(false),
    });
  }
  return html;
}

function chartOpts(legend) {
  return {
    responsive: true,
    plugins: {
      legend: { display: legend, labels: { color: '#94a3b8', font: { size: 11 } } },
    },
    scales: {
      x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#1e293b' } },
      y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#1e293b' } },
    },
  };
}

function flyTo(lat, lng) {
  map.flyTo([lat, lng], 14, { duration: 1 });
}

async function ackAlert(id) {
  await fetch(`${API}/alerts/${id}/acknowledge`, { method: 'PATCH' });
  loadAlerts();
}
"""

with open("frontend/static/js/dashboard.js", "w") as f:
    f.write(dump)

