-- FlowCast AI — MySQL Schema for Flipkart Gridlock 5.0
-- Import this file in MySQL Workbench: File → Run SQL Script

CREATE DATABASE IF NOT EXISTS flowcast_ai
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE flowcast_ai;

-- ─── Core event data (from Astram dataset) ───────────────────────────────────
CREATE TABLE IF NOT EXISTS events (
    id                  VARCHAR(20) PRIMARY KEY,
    event_type          ENUM('planned', 'unplanned') NOT NULL,
    latitude            DECIMAL(10, 7) NOT NULL,
    longitude           DECIMAL(10, 7) NOT NULL,
    end_latitude        DECIMAL(10, 7),
    end_longitude       DECIMAL(10, 7),
    address             TEXT,
    end_address         TEXT,
    event_cause         VARCHAR(50),
    requires_road_closure BOOLEAN DEFAULT FALSE,
    start_datetime      DATETIME NOT NULL,
    end_datetime        DATETIME,
    status              VARCHAR(20),
    authenticated       VARCHAR(10),
    description         TEXT,
    veh_type            VARCHAR(50),
    corridor            VARCHAR(100),
    priority            ENUM('High', 'Low', 'Medium') DEFAULT 'Low',
    police_station      VARCHAR(100),
    zone                VARCHAR(100),
    junction            VARCHAR(150),
    created_date        DATETIME,
    resolved_datetime   DATETIME,
    duration_minutes    INT GENERATED ALWAYS AS (
        CASE WHEN end_datetime IS NOT NULL AND start_datetime IS NOT NULL
             THEN TIMESTAMPDIFF(MINUTE, start_datetime, end_datetime)
             WHEN resolved_datetime IS NOT NULL
             THEN TIMESTAMPDIFF(MINUTE, start_datetime, resolved_datetime)
             ELSE NULL END
    ) STORED,
    INDEX idx_event_type (event_type),
    INDEX idx_event_cause (event_cause),
    INDEX idx_corridor (corridor),
    INDEX idx_start (start_datetime),
    INDEX idx_status (status),
    INDEX idx_location (latitude, longitude)
);

-- ─── Road segments / corridors ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS road_segments (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    segment_name    VARCHAR(150) NOT NULL UNIQUE,
    corridor        VARCHAR(100),
    center_lat      DECIMAL(10, 7),
    center_lng      DECIMAL(10, 7),
    capacity        INT DEFAULT 1000,
    lane_count      TINYINT DEFAULT 2,
    speed_limit     TINYINT DEFAULT 40,
    betweenness     DECIMAL(8, 6) DEFAULT 0,
    zone            VARCHAR(100),
    INDEX idx_corridor_seg (corridor)
);

-- ─── BTP police stations ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS police_stations (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    latitude        DECIMAL(10, 7) NOT NULL,
    longitude       DECIMAL(10, 7) NOT NULL,
    zone            VARCHAR(100),
    officer_count   INT DEFAULT 12,
    INDEX idx_zone (zone)
);

-- ─── Congestion forecasts (Module A — Planned) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS congestion_forecasts (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    segment_id      INT NOT NULL,
    forecast_time   DATETIME NOT NULL,
    target_time     DATETIME NOT NULL,
    crs_score       DECIMAL(5, 2) NOT NULL COMMENT 'Congestion Risk Score 0-100',
    crs_p10         DECIMAL(5, 2),
    crs_p50         DECIMAL(5, 2),
    crs_p90         DECIMAL(5, 2),
    event_id        VARCHAR(20),
    source          ENUM('planned_tft', 'unplanned_anomaly', 'nlp', 'fusion') DEFAULT 'planned_tft',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (segment_id) REFERENCES road_segments(id),
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE SET NULL,
    INDEX idx_target (target_time),
    INDEX idx_segment (segment_id)
);

-- ─── Live alerts (Module B — Unplanned) ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    alert_type      ENUM('RED', 'AMBER', 'GREEN') NOT NULL,
    severity        TINYINT NOT NULL COMMENT '1-5',
    incident_type   VARCHAR(50),
    title           VARCHAR(255) NOT NULL,
    description     TEXT,
    latitude        DECIMAL(10, 7) NOT NULL,
    longitude       DECIMAL(10, 7) NOT NULL,
    affected_radius_km DECIMAL(4, 2) DEFAULT 2.0,
    crs_score       DECIMAL(5, 2),
    eta_clear_min   INT,
    source          ENUM('sensor_anomaly', 'nlp', 'fusion', 'manual') DEFAULT 'fusion',
    event_id        VARCHAR(20),
    status          ENUM('active', 'acknowledged', 'resolved') DEFAULT 'active',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at     TIMESTAMP NULL,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE SET NULL,
    INDEX idx_status (status),
    INDEX idx_created (created_at),
    INDEX idx_alert_type (alert_type)
);

-- ─── Officer deployment recommendations ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS deployment_briefs (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    alert_id        BIGINT,
    event_id        VARCHAR(20),
    title           VARCHAR(255) NOT NULL,
    officers_needed INT NOT NULL,
    deploy_by       DATETIME NOT NULL,
    primary_junction VARCHAR(150),
    secondary_junction VARCHAR(150),
    station_id      INT,
    estimated_reduction_pct DECIMAL(5, 2),
    economic_savings_inr  DECIMAL(15, 2),
    brief_text      TEXT,
    status          ENUM('pending', 'deployed', 'completed') DEFAULT 'pending',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (alert_id) REFERENCES alerts(id) ON DELETE SET NULL,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE SET NULL,
    FOREIGN KEY (station_id) REFERENCES police_stations(id) ON DELETE SET NULL
);

-- ─── Economic impact log ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS economic_impact (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    event_id        VARCHAR(20),
    alert_id        BIGINT,
    date            DATE NOT NULL,
    affected_vehicles INT,
    delay_minutes   INT,
    cost_inr        DECIMAL(15, 2),
    prevented_cost_inr DECIMAL(15, 2) DEFAULT 0,
    notes           TEXT,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE SET NULL,
    FOREIGN KEY (alert_id) REFERENCES alerts(id) ON DELETE SET NULL,
    INDEX idx_date (date)
);

-- ─── NLP incident classifications ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nlp_incidents (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    raw_text        TEXT NOT NULL,
    classified_type VARCHAR(50),
    confidence      DECIMAL(5, 4),
    extracted_road  VARCHAR(150),
    extracted_junction VARCHAR(150),
    severity_words  VARCHAR(100),
    latitude        DECIMAL(10, 7),
    longitude       DECIMAL(10, 7),
    event_id        VARCHAR(20),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE SET NULL
);

-- ─── Audit log (PDPB compliance) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    action          VARCHAR(100) NOT NULL,
    entity_type     VARCHAR(50),
    entity_id       VARCHAR(50),
    details         JSON,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_action (action),
    INDEX idx_created_audit (created_at)
);

-- ─── Model metrics (accuracy tracker) ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_metrics (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL,
    metric_name     VARCHAR(50) NOT NULL,
    metric_value    DECIMAL(8, 4),
    evaluated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
