"""
database.py — Motor de persistencia SQLite para analítica y aforo vehicular.

Almacena y consulta eventos históricos de cruce de vehículos para auditoría,
métricas horarias, detección de horas pico y reporteo ejecutivo para toma de decisiones.
"""

import sqlite3
import datetime
import random
import os
import uuid
from typing import Optional, Any
from pathlib import Path

import auth_service

DB_PATH = Path(__file__).parent / "traffic_history.db"

def get_connection() -> sqlite3.Connection:
    """Obtiene una conexión a la base de datos con row_factory como sqlite3.Row."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Inicializa la estructura de tablas e índices si no existen."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                camera_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                vehicle_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                dwell_time REAL DEFAULT 0.0
            )
        """)
        
        # Índices para acelerar consultas analíticas
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON traffic_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_cam_time ON traffic_events(camera_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON traffic_events(vehicle_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_direction ON traffic_events(direction)")

        # Migración dinámica de columna entity_type en traffic_events si no existe
        cursor.execute("PRAGMA table_info(traffic_events)")
        event_cols = [r["name"] for r in cursor.fetchall()]
        if "entity_type" not in event_cols:
            cursor.execute("ALTER TABLE traffic_events ADD COLUMN entity_type TEXT DEFAULT 'VEHICLE'")
            cursor.execute("UPDATE traffic_events SET entity_type = 'PEDESTRIAN' WHERE vehicle_type = 'Peaton' OR vehicle_type = 'Persona'")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_entity ON traffic_events(entity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_entity_time ON traffic_events(entity_type, timestamp)")

        # Migración dinámica de columna line_id en traffic_events si no existe
        if "line_id" not in event_cols:
            cursor.execute("ALTER TABLE traffic_events ADD COLUMN line_id TEXT DEFAULT 'line_1'")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_line ON traffic_events(line_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_line_time ON traffic_events(line_id, timestamp)")

        # ── TABLA DE AVISTAMIENTOS EN CÁMARA (MÉTRICA 1: VISTOS EN FOV) ─────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS camera_sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                camera_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                vehicle_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                crossed INTEGER DEFAULT 0,
                line_id TEXT DEFAULT NULL
            )
        """)
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_sightings_unique ON camera_sightings(camera_id, track_id, date(timestamp))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_date ON camera_sightings(date(timestamp))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_cam_date ON camera_sightings(camera_id, date(timestamp))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_entity_date ON camera_sightings(entity_type, date(timestamp))")

        # ── TABLA DE UBICACIONES / SITIOS MULTI-PANTALLA ─────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                address TEXT,
                novastar_ip TEXT DEFAULT '192.168.1.140',
                novastar_port INTEGER DEFAULT 8001,
                shelly_ip TEXT DEFAULT '192.168.1.150',
                shelly_channel INTEGER DEFAULT 0,
                screen_area_m2 REAL DEFAULT 32.0,
                cost_per_kwh REAL DEFAULT 3.85,
                is_active INTEGER DEFAULT 1
            )
        """)

        # Migración dinámica de columnas geográficas y de orientación solar si no existen
        cursor.execute("PRAGMA table_info(locations)")
        loc_cols = [r["name"] for r in cursor.fetchall()]
        if "latitude" not in loc_cols:
            cursor.execute("ALTER TABLE locations ADD COLUMN latitude REAL DEFAULT 20.6736")
        if "longitude" not in loc_cols:
            cursor.execute("ALTER TABLE locations ADD COLUMN longitude REAL DEFAULT -103.3855")
        if "screen_orientation_deg" not in loc_cols:
            cursor.execute("ALTER TABLE locations ADD COLUMN screen_orientation_deg REAL DEFAULT 270.0")
        if "min_night_brightness" not in loc_cols:
            cursor.execute("ALTER TABLE locations ADD COLUMN min_night_brightness INTEGER DEFAULT 25")
        if "max_day_brightness" not in loc_cols:
            cursor.execute("ALTER TABLE locations ADD COLUMN max_day_brightness INTEGER DEFAULT 95")
        if "auto_brightness_enabled" not in loc_cols:
            cursor.execute("ALTER TABLE locations ADD COLUMN auto_brightness_enabled INTEGER DEFAULT 1")

        # ── TABLA DE CAMPAÑAS PUBLICITARIAS Y PAUTAS ─────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                client TEXT NOT NULL,
                location_id TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                start_hour INTEGER DEFAULT 6,
                end_hour INTEGER DEFAULT 23,
                spot_seconds INTEGER DEFAULT 15,
                spots_per_hour INTEGER DEFAULT 12,
                status TEXT DEFAULT 'Activa',
                FOREIGN KEY (location_id) REFERENCES locations(id)
            )
        """)

        # Migración de columna client_id en campaigns
        cursor.execute("PRAGMA table_info(campaigns)")
        camp_cols = [r["name"] for r in cursor.fetchall()]
        if "client_id" not in camp_cols:
            cursor.execute("ALTER TABLE campaigns ADD COLUMN client_id TEXT")

        # ── TABLA DE VIDEOS Y SPOTS PUBLICITARIOS (DETALLE POR VIDEO) ─────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS campaign_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER NOT NULL,
                client_id TEXT NOT NULL,
                video_name TEXT NOT NULL,
                title TEXT NOT NULL,
                duration_seconds INTEGER DEFAULT 15,
                resolution TEXT DEFAULT '1920x1080',
                fps INTEGER DEFAULT 60,
                plays_today INTEGER DEFAULT 0,
                total_plays INTEGER DEFAULT 0,
                total_impressions INTEGER DEFAULT 0,
                kwh_consumed REAL DEFAULT 0.0,
                status TEXT DEFAULT 'En rotación',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
                FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_campaign ON campaign_videos(campaign_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_client ON campaign_videos(client_id)")

        # ── TABLA DE REPORTES DE SINCRONIZACIÓN EDGE ─────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edge_sync_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                location_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                date_reported TEXT NOT NULL,
                total_vehicles INTEGER NOT NULL,
                total_in INTEGER NOT NULL,
                total_out INTEGER NOT NULL,
                peak_hour TEXT,
                raw_summary TEXT
            )
        """)

        # ── TABLA DE CLIENTES Y MARCAS PUBLICITARIAS ─────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                contact_email TEXT,
                logo_url TEXT,
                kwh_rate_cfe REAL DEFAULT 3.85,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migración dinámica de columna kwh_rate_cfe en clients
        cursor.execute("PRAGMA table_info(clients)")
        client_cols = [r["name"] for r in cursor.fetchall()]
        if "kwh_rate_cfe" not in client_cols:
            cursor.execute("ALTER TABLE clients ADD COLUMN kwh_rate_cfe REAL DEFAULT 3.85")

        # ── TABLA DE ASOCIACIÓN CLIENTE ↔ PANTALLAS / UBICACIONES ────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_locations (
                client_id TEXT NOT NULL,
                location_id TEXT NOT NULL,
                custom_kwh_rate REAL DEFAULT NULL,
                PRIMARY KEY (client_id, location_id),
                FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE,
                FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE CASCADE
            )
        """)

        # Migración dinámica de custom_kwh_rate en client_locations
        cursor.execute("PRAGMA table_info(client_locations)")
        cl_cols = [r["name"] for r in cursor.fetchall()]
        if "custom_kwh_rate" not in cl_cols:
            cursor.execute("ALTER TABLE client_locations ADD COLUMN custom_kwh_rate REAL DEFAULT NULL")

        # ── TABLA DE CONTROLADORES Y DISPOSITIVOS (TB40 / VX600 PRO / SHELLY) ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hardware_devices (
                id TEXT PRIMARY KEY,
                location_id TEXT NOT NULL,
                name TEXT NOT NULL,
                device_type TEXT NOT NULL,   -- 'TB40', 'VX600_PRO', 'HYBRID', 'SHELLY_PRO'
                ip_address TEXT NOT NULL,
                port INTEGER DEFAULT 8001,
                dual_screen_enabled INTEGER DEFAULT 0,
                active_preset TEXT DEFAULT 'preset_mirror_1tb40',
                input_source_1 TEXT DEFAULT 'TB40_MASTER',
                input_source_2 TEXT DEFAULT 'TB40_SLAVE',
                status TEXT DEFAULT 'online',
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE CASCADE
            )
        """)

        # ── TABLA DE LOGS Y CONEXIONES/DESCONEXIONES DE CÁMARAS ───────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS camera_connection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                camera_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT,
                details TEXT,
                duration_offline_sec REAL DEFAULT 0.0,
                duration_online_sec REAL DEFAULT 0.0
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cam_conn_time ON camera_connection_logs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cam_conn_cam_time ON camera_connection_logs(camera_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cam_conn_event ON camera_connection_logs(event_type)")

        # ── TABLA DE USUARIOS Y ACCESOS UNIFICADOS ────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                nombre TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                rol TEXT NOT NULL DEFAULT 'cliente',
                cliente_id TEXT,
                activo INTEGER DEFAULT 1,
                creado_en DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cliente_id) REFERENCES clients(id) ON DELETE SET NULL
            )
        """)
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")

        # ── TABLA DE INVITACIONES POR CORREO ─────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS invitations (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                nombre TEXT NOT NULL,
                rol TEXT NOT NULL DEFAULT 'cliente',
                cliente_id TEXT,
                token TEXT UNIQUE NOT NULL,
                estado TEXT NOT NULL DEFAULT 'pendiente',
                expira_en DATETIME NOT NULL,
                creado_en DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cliente_id) REFERENCES clients(id) ON DELETE SET NULL
            )
        """)
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_invitations_token ON invitations(token)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_invitations_email ON invitations(email)")

        conn.commit()

    # Si la base de datos está vacía, generar datos de muestra para pruebas y demostración ejecutiva
    seed_sample_data_if_empty()

def record_event(
    camera_id: str,
    track_id: int,
    vehicle_type: str,
    direction: str,
    confidence: float = 0.0,
    dwell_time: float = 0.0,
    event_time: Optional[datetime.datetime] = None,
    entity_type: Optional[str] = None,
    line_id: str = "line_1"
) -> int:
    """Registra un evento de cruce de línea en la base de datos (vehículo o peatón y línea de conteo)."""
    if event_time is None:
        event_time = datetime.datetime.now()
    
    if entity_type is None:
        entity_type = "PEDESTRIAN" if vehicle_type in ("Peaton", "Persona") else "VEHICLE"

    time_str = event_time.strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO traffic_events (timestamp, camera_id, track_id, vehicle_type, direction, confidence, dwell_time, entity_type, line_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (time_str, camera_id, track_id, vehicle_type, direction, round(confidence, 2), round(dwell_time, 1), entity_type, line_id))
        conn.commit()
        return cursor.lastrowid or 0

def record_sighting(
    camera_id: str,
    track_id: int,
    vehicle_type: str,
    entity_type: str = "VEHICLE",
    confidence: float = 0.0,
    timestamp: Optional[datetime.datetime] = None
) -> None:
    """Registra la detección de un vehículo o persona en el campo visual de la cámara (Métrica 1)."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO camera_sightings (timestamp, camera_id, track_id, vehicle_type, entity_type, confidence, crossed)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (time_str, camera_id, track_id, vehicle_type, entity_type, round(confidence, 2)))
        conn.commit()

def mark_sighting_crossed(
    camera_id: str,
    track_id: int,
    line_id: str = "line_1",
    timestamp: Optional[datetime.datetime] = None
) -> None:
    """Marca que un vehículo o persona visto en la cámara cruzó por una línea delimitadora (Métrica 2)."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE camera_sightings
            SET crossed = 1, line_id = ?
            WHERE camera_id = ? AND track_id = ? AND date(timestamp) = ?
        """, (line_id, camera_id, track_id, date_str))
        conn.commit()

def get_today_counts(camera_id: str) -> dict[str, Any]:
    """Obtiene los conteos acumulados de hoy para inicializar TrafficAnalyzer tras un reinicio."""
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity_type, COUNT(DISTINCT track_id) as count
            FROM camera_sightings
            WHERE camera_id = ? AND date(timestamp) = ?
            GROUP BY entity_type
        """, (camera_id, today_str))
        seen_map = {r["entity_type"]: r["count"] for r in cursor.fetchall()}
        
        cursor.execute("""
            SELECT entity_type, direction, COUNT(*) as count
            FROM traffic_events
            WHERE camera_id = ? AND date(timestamp) = ?
            GROUP BY entity_type, direction
        """, (camera_id, today_str))
        events_rows = cursor.fetchall()
        
        veh_in = sum(r["count"] for r in events_rows if r["entity_type"] == "VEHICLE" and r["direction"] == "IN")
        veh_out = sum(r["count"] for r in events_rows if r["entity_type"] == "VEHICLE" and r["direction"] == "OUT")
        ped_in = sum(r["count"] for r in events_rows if r["entity_type"] == "PEDESTRIAN" and r["direction"] == "IN")
        ped_out = sum(r["count"] for r in events_rows if r["entity_type"] == "PEDESTRIAN" and r["direction"] == "OUT")

        cursor.execute("""
            SELECT entity_type, COUNT(DISTINCT track_id) as count
            FROM traffic_events
            WHERE camera_id = ? AND date(timestamp) = ?
            GROUP BY entity_type
        """, (camera_id, today_str))
        crossed_map = {r["entity_type"]: r["count"] for r in cursor.fetchall()}

        return {
            "vehicles_seen": max(seen_map.get("VEHICLE", 0), crossed_map.get("VEHICLE", 0)),
            "pedestrians_seen": max(seen_map.get("PEDESTRIAN", 0), crossed_map.get("PEDESTRIAN", 0)),
            "vehicles_crossed": crossed_map.get("VEHICLE", 0),
            "pedestrians_crossed": crossed_map.get("PEDESTRIAN", 0),
            "vehicles_in": veh_in,
            "vehicles_out": veh_out,
            "pedestrians_in": ped_in,
            "pedestrians_out": ped_out,
        }

def get_today_line_counts(camera_id: str) -> dict[str, dict[str, int]]:
    """Devuelve los conteos acumulados de hoy desglosados por línea para una cámara."""
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT line_id, entity_type, direction, COUNT(*) as count
            FROM traffic_events
            WHERE camera_id = ? AND date(timestamp) = ?
            GROUP BY line_id, entity_type, direction
        """, (camera_id, today_str))
        res = {}
        for r in cursor.fetchall():
            lid = r["line_id"]
            if not lid:
                continue
            if lid not in res:
                res[lid] = {"vehicles_in": 0, "vehicles_out": 0, "pedestrians_in": 0, "pedestrians_out": 0}
            ent = r["entity_type"]
            dirn = r["direction"]
            cnt = r["count"]
            if ent == "PEDESTRIAN":
                if dirn == "IN":
                    res[lid]["pedestrians_in"] += cnt
                else:
                    res[lid]["pedestrians_out"] += cnt
            else:
                if dirn == "IN":
                    res[lid]["vehicles_in"] += cnt
                else:
                    res[lid]["vehicles_out"] += cnt
        return res

def get_kpis(
    camera_id: Optional[str] = None,
    date_str: Optional[str] = None,
    entity_type: Optional[str] = None,
    line_id: Optional[str] = None,
    period: str = "day",
    month_str: Optional[str] = None,
    year_str: Optional[str] = None
) -> dict[str, Any]:
    """
    Obtiene indicadores clave de rendimiento (KPIs) globales y desglosados con Doble Métrica:
    Vistos en Cámara vs Cruces por Línea, con soporte para granularidad de tiempo:
    - period="day": por fecha (date_str, por defecto hoy)
    - period="month": por mes (month_str ej. '2026-09')
    - period="year": por año (year_str ej. '2026')
    """
    base_where_clauses = []
    base_params: list[Any] = []

    if period == "month":
        m_val = month_str or (date_str[:7] if date_str else datetime.date.today().strftime("%Y-%m"))
        base_where_clauses.append("strftime('%Y-%m', timestamp) = ?")
        base_params.append(m_val)
        display_date = m_val
    elif period == "year":
        y_val = year_str or (date_str[:4] if date_str else datetime.date.today().strftime("%Y"))
        base_where_clauses.append("strftime('%Y', timestamp) = ?")
        base_params.append(y_val)
        display_date = y_val
    else:
        d_val = date_str or datetime.date.today().strftime("%Y-%m-%d")
        base_where_clauses.append("date(timestamp) = ?")
        base_params.append(d_val)
        display_date = d_val
        
    if camera_id and camera_id != "all":
        base_where_clauses.append("camera_id = ?")
        base_params.append(camera_id)

    if line_id and line_id != "all":
        base_where_clauses.append("line_id = ?")
        base_params.append(line_id)
        
    base_where_sql = " AND ".join(base_where_clauses)
    
    # Cláusula para filtro específico si se solicita un panel concreto
    active_where_clauses = list(base_where_clauses)
    active_params = list(base_params)
    if entity_type and entity_type.upper() != "ALL":
        active_where_clauses.append("UPPER(entity_type) = ?")
        active_params.append(entity_type.upper())
    active_where_sql = " AND ".join(active_where_clauses)

    # Cláusula para tabla de avistamientos (Métrica 1: Vistos en campo de visión)
    s_clauses = list(base_where_clauses)
    s_params = list(base_params)
    s_sql = " AND ".join(s_clauses)

    with get_connection() as conn:
        cursor = conn.cursor()

        # Avistamientos registrados en cámara (Métrica 1)
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_sightings,
                SUM(CASE WHEN UPPER(entity_type) = 'VEHICLE' THEN 1 ELSE 0 END) as veh_sightings,
                SUM(CASE WHEN UPPER(entity_type) = 'PEDESTRIAN' THEN 1 ELSE 0 END) as ped_sightings
            FROM camera_sightings
            WHERE {s_sql}
        """, s_params)
        s_row = cursor.fetchone()
        db_veh_seen = (s_row["veh_sightings"] if s_row else 0) or 0
        db_ped_seen = (s_row["ped_sightings"] if s_row else 0) or 0
        db_total_seen = (s_row["total_sightings"] if s_row else 0) or 0
        
        # Totales In / Out de cruces por líneas según filtro activo (Métrica 2)
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_flow,
                COUNT(DISTINCT track_id) as total_crossed_tracks,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                AVG(dwell_time) as avg_dwell
            FROM traffic_events
            WHERE {active_where_sql}
        """, active_params)
        row = cursor.fetchone()
        
        total_flow = row["total_flow"] or 0
        total_crossed_tracks = row["total_crossed_tracks"] or 0
        total_in = row["total_in"] or 0
        total_out = row["total_out"] or 0
        avg_dwell = round(row["avg_dwell"] or 0.0, 1)
        
        # Hora pico según filtro activo
        cursor.execute(f"""
            SELECT strftime('%H:00', timestamp) as hour_slot, COUNT(*) as count
            FROM traffic_events
            WHERE {active_where_sql}
            GROUP BY hour_slot
            ORDER BY count DESC
            LIMIT 1
        """, active_params)
        peak_row = cursor.fetchone()
        peak_label = "peatones" if entity_type and entity_type.upper() == "PEDESTRIAN" else "veh."
        peak_hour = f"{peak_row['hour_slot']} ({peak_row['count']} {peak_label})" if peak_row else "N/A"
        
        # Desglose por tipo de objeto
        cursor.execute(f"""
            SELECT vehicle_type, COUNT(*) as count
            FROM traffic_events
            WHERE {active_where_sql}
            GROUP BY vehicle_type
            ORDER BY count DESC
        """, active_params)
        classes = {r["vehicle_type"]: r["count"] for r in cursor.fetchall()}

        # ── SUBPANEL ESPECÍFICO: VEHÍCULOS ──────────────────────────────────
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_flow,
                COUNT(DISTINCT track_id) as total_crossed_tracks,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                AVG(dwell_time) as avg_dwell
            FROM traffic_events
            WHERE {base_where_sql} AND UPPER(entity_type) = 'VEHICLE'
        """, base_params)
        v_row = cursor.fetchone()
        v_total_flow = v_row["total_flow"] or 0
        v_total_crossed = v_row["total_crossed_tracks"] or 0
        v_total_in = v_row["total_in"] or 0
        v_total_out = v_row["total_out"] or 0
        v_avg_dwell = round(v_row["avg_dwell"] or 0.0, 1)

        v_seen = max(db_veh_seen, v_total_crossed)
        v_crossing_rate = round((v_total_crossed / max(v_seen, 1)) * 100.0, 1) if v_seen > 0 else 0.0

        cursor.execute(f"""
            SELECT strftime('%H:00', timestamp) as hour_slot, COUNT(*) as count
            FROM traffic_events
            WHERE {base_where_sql} AND UPPER(entity_type) = 'VEHICLE'
            GROUP BY hour_slot
            ORDER BY count DESC
            LIMIT 1
        """, base_params)
        v_peak_row = cursor.fetchone()
        v_peak_hour = f"{v_peak_row['hour_slot']} ({v_peak_row['count']} veh.)" if v_peak_row else "N/A"

        cursor.execute(f"""
            SELECT vehicle_type, COUNT(*) as count
            FROM traffic_events
            WHERE {base_where_sql} AND UPPER(entity_type) = 'VEHICLE'
            GROUP BY vehicle_type
            ORDER BY count DESC
        """, base_params)
        v_classes = {r["vehicle_type"]: r["count"] for r in cursor.fetchall()}

        # ── SUBPANEL ESPECÍFICO: PEATONES ───────────────────────────────────
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_flow,
                COUNT(DISTINCT track_id) as total_crossed_tracks,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                AVG(dwell_time) as avg_dwell
            FROM traffic_events
            WHERE {base_where_sql} AND UPPER(entity_type) = 'PEDESTRIAN'
        """, base_params)
        p_row = cursor.fetchone()
        p_total_flow = p_row["total_flow"] or 0
        p_total_crossed = p_row["total_crossed_tracks"] or 0
        p_total_in = p_row["total_in"] or 0
        p_total_out = p_row["total_out"] or 0
        p_avg_dwell = round(p_row["avg_dwell"] or 0.0, 1)

        p_seen = max(db_ped_seen, p_total_crossed)
        p_crossing_rate = round((p_total_crossed / max(p_seen, 1)) * 100.0, 1) if p_seen > 0 else 0.0

        cursor.execute(f"""
            SELECT strftime('%H:00', timestamp) as hour_slot, COUNT(*) as count
            FROM traffic_events
            WHERE {base_where_sql} AND UPPER(entity_type) = 'PEDESTRIAN'
            GROUP BY hour_slot
            ORDER BY count DESC
            LIMIT 1
        """, base_params)
        p_peak_row = cursor.fetchone()
        p_peak_hour = f"{p_peak_row['hour_slot']} ({p_peak_row['count']} peatones)" if p_peak_row else "N/A"

        final_total_seen = max(db_total_seen, total_crossed_tracks, v_seen + p_seen)
        tot_crossing_rate = round((total_crossed_tracks / max(final_total_seen, 1)) * 100.0, 1) if final_total_seen > 0 else 0.0
        
        return {
            "date": display_date,
            "period": period,
            "total_seen": final_total_seen,
            "total_crossed": total_crossed_tracks,
            "total_flow": total_flow,
            "total_in": total_in,
            "total_out": total_out,
            "crossing_rate": tot_crossing_rate,
            "avg_dwell_seconds": avg_dwell,
            "peak_hour": peak_hour,
            "vehicle_classes": classes,
            "vehicles": {
                "seen": v_seen,
                "crossed": v_total_crossed,
                "total_flow": v_total_flow,
                "total_in": v_total_in,
                "total_out": v_total_out,
                "crossing_rate": v_crossing_rate,
                "avg_dwell_seconds": v_avg_dwell,
                "peak_hour": v_peak_hour,
                "classes": v_classes
            },
            "pedestrians": {
                "seen": p_seen,
                "crossed": p_total_crossed,
                "total_flow": p_total_flow,
                "total_in": p_total_in,
                "total_out": p_total_out,
                "crossing_rate": p_crossing_rate,
                "avg_dwell_seconds": p_avg_dwell,
                "peak_hour": p_peak_hour
            }
        }

def get_hourly_metrics(camera_id: Optional[str] = None, date_str: Optional[str] = None, entity_type: Optional[str] = None) -> list[dict[str, Any]]:
    """Obtiene el aforo desglosado hora por hora (00:00 a 23:00) para un día, con separación vehicular y peatonal."""
    if date_str is None:
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        
    where_clauses = ["date(timestamp) = ?"]
    params: list[Any] = [date_str]
    
    if camera_id and camera_id != "all":
        where_clauses.append("camera_id = ?")
        params.append(camera_id)

    if entity_type and entity_type.upper() != "ALL":
        where_clauses.append("UPPER(entity_type) = ?")
        params.append(entity_type.upper())
        
    where_sql = " AND ".join(where_clauses)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT 
                cast(strftime('%H', timestamp) as integer) as hour,
                COUNT(*) as total,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                SUM(CASE WHEN UPPER(entity_type) = 'VEHICLE' THEN 1 ELSE 0 END) as veh_total,
                SUM(CASE WHEN UPPER(entity_type) = 'VEHICLE' AND direction = 'IN' THEN 1 ELSE 0 END) as veh_in,
                SUM(CASE WHEN UPPER(entity_type) = 'VEHICLE' AND direction = 'OUT' THEN 1 ELSE 0 END) as veh_out,
                SUM(CASE WHEN UPPER(entity_type) = 'PEDESTRIAN' THEN 1 ELSE 0 END) as ped_total,
                SUM(CASE WHEN UPPER(entity_type) = 'PEDESTRIAN' AND direction = 'IN' THEN 1 ELSE 0 END) as ped_in,
                SUM(CASE WHEN UPPER(entity_type) = 'PEDESTRIAN' AND direction = 'OUT' THEN 1 ELSE 0 END) as ped_out
            FROM traffic_events
            WHERE {where_sql}
            GROUP BY hour
            ORDER BY hour ASC
        """, params)
        
        rows = {r["hour"]: r for r in cursor.fetchall()}
        
        # Completar las 24 horas del día
        hourly_data = []
        for h in range(24):
            hour_label = f"{h:02d}:00"
            if h in rows:
                r = rows[h]
                hourly_data.append({
                    "hour": hour_label,
                    "total": r["total"],
                    "in": r["total_in"],
                    "out": r["total_out"],
                    "vehicles_total": r["veh_total"] or 0,
                    "vehicles_in": r["veh_in"] or 0,
                    "vehicles_out": r["veh_out"] or 0,
                    "pedestrians_total": r["ped_total"] or 0,
                    "pedestrians_in": r["ped_in"] or 0,
                    "pedestrians_out": r["ped_out"] or 0,
                })
            else:
                hourly_data.append({
                    "hour": hour_label,
                    "total": 0,
                    "in": 0,
                    "out": 0,
                    "vehicles_total": 0,
                    "vehicles_in": 0,
                    "vehicles_out": 0,
                    "pedestrians_total": 0,
                    "pedestrians_in": 0,
                    "pedestrians_out": 0,
                })
        return hourly_data

def get_daily_metrics(camera_id: Optional[str] = None, days: int = 7, entity_type: Optional[str] = None) -> list[dict[str, Any]]:
    """Obtiene el volumen total por día para los últimos N días con desglose vehicular y peatonal."""
    where_clauses = ["timestamp >= date('now', ?)"]
    params: list[Any] = [f"-{days} days"]
    
    if camera_id and camera_id != "all":
        where_clauses.append("camera_id = ?")
        params.append(camera_id)

    if entity_type and entity_type.upper() != "ALL":
        where_clauses.append("UPPER(entity_type) = ?")
        params.append(entity_type.upper())
        
    where_sql = " AND ".join(where_clauses)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT 
                date(timestamp) as day_date,
                COUNT(*) as total,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                SUM(CASE WHEN UPPER(entity_type) = 'VEHICLE' THEN 1 ELSE 0 END) as veh_total,
                SUM(CASE WHEN UPPER(entity_type) = 'PEDESTRIAN' THEN 1 ELSE 0 END) as ped_total
            FROM traffic_events
            WHERE {where_sql}
            GROUP BY day_date
            ORDER BY day_date ASC
        """, params)
        
        return [
            {
                "date": r["day_date"],
                "total": r["total"],
                "in": r["total_in"],
                "out": r["total_out"],
                "vehicles_total": r["veh_total"] or 0,
                "pedestrians_total": r["ped_total"] or 0,
            }
            for r in cursor.fetchall()
        ]

def get_period_chart_data(
    period: str = "day",
    date_str: Optional[str] = None,
    month_str: Optional[str] = None,
    year_str: Optional[str] = None,
    camera_id: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Genera los puntos de datos para la gráfica del reporte según el período:
    - period="day": 24 horas (00:00 a 23:00)
    - period="month": Días del mes (01 a 31)
    - period="year": Meses del año (Ene a Dic)
    """
    if period == "day":
        return get_hourly_metrics(camera_id=camera_id, date_str=date_str)

    with get_connection() as conn:
        cursor = conn.cursor()

        if period == "month":
            m_val = month_str or (date_str[:7] if date_str else datetime.date.today().strftime("%Y-%m"))
            where_clauses = ["strftime('%Y-%m', timestamp) = ?"]
            params: list[Any] = [m_val]
            if camera_id and camera_id != "all":
                where_clauses.append("camera_id = ?")
                params.append(camera_id)

            where_sql = " AND ".join(where_clauses)
            cursor.execute(f"""
                SELECT 
                    cast(strftime('%d', timestamp) as integer) as day_num,
                    COUNT(*) as total,
                    SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                    SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                    SUM(CASE WHEN UPPER(entity_type) = 'VEHICLE' THEN 1 ELSE 0 END) as veh_total,
                    SUM(CASE WHEN UPPER(entity_type) = 'PEDESTRIAN' THEN 1 ELSE 0 END) as ped_total
                FROM traffic_events
                WHERE {where_sql}
                GROUP BY day_num
                ORDER BY day_num ASC
            """, params)
            day_rows = {r["day_num"]: r for r in cursor.fetchall()}

            # Hasta 31 días
            result = []
            for d in range(1, 32):
                r = day_rows.get(d)
                result.append({
                    "hour": f"Día {d:02d}",
                    "label": f"{d:02d}",
                    "slot_index": d,
                    "total": r["total"] if r else 0,
                    "in": r["total_in"] if r else 0,
                    "out": r["total_out"] if r else 0,
                    "vehicles_total": (r["veh_total"] or 0) if r else 0,
                    "pedestrians_total": (r["ped_total"] or 0) if r else 0,
                })
            return result

        elif period == "year":
            y_val = year_str or (date_str[:4] if date_str else datetime.date.today().strftime("%Y"))
            where_clauses = ["strftime('%Y', timestamp) = ?"]
            params = [y_val]
            if camera_id and camera_id != "all":
                where_clauses.append("camera_id = ?")
                params.append(camera_id)

            where_sql = " AND ".join(where_clauses)
            cursor.execute(f"""
                SELECT 
                    cast(strftime('%m', timestamp) as integer) as month_num,
                    COUNT(*) as total,
                    SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                    SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                    SUM(CASE WHEN UPPER(entity_type) = 'VEHICLE' THEN 1 ELSE 0 END) as veh_total,
                    SUM(CASE WHEN UPPER(entity_type) = 'PEDESTRIAN' THEN 1 ELSE 0 END) as ped_total
                FROM traffic_events
                WHERE {where_sql}
                GROUP BY month_num
                ORDER BY month_num ASC
            """, params)
            month_rows = {r["month_num"]: r for r in cursor.fetchall()}

            month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
            result = []
            for m in range(1, 13):
                r = month_rows.get(m)
                result.append({
                    "hour": month_names[m - 1],
                    "label": month_names[m - 1],
                    "slot_index": m,
                    "total": r["total"] if r else 0,
                    "in": r["total_in"] if r else 0,
                    "out": r["total_out"] if r else 0,
                    "vehicles_total": (r["veh_total"] or 0) if r else 0,
                    "pedestrians_total": (r["ped_total"] or 0) if r else 0,
                })
            return result

    return get_hourly_metrics(camera_id=camera_id, date_str=date_str)

def get_events(
    camera_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    vehicle_type: Optional[str] = None,
    direction: Optional[str] = None,
    entity_type: Optional[str] = None,
    line_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> dict[str, Any]:
    """Consulta la lista de eventos filtrada y paginada, con soporte para entity_type y line_id."""
    where_clauses = []
    params: list[Any] = []
    
    if camera_id and camera_id != "all":
        where_clauses.append("camera_id = ?")
        params.append(camera_id)

    if line_id and line_id != "all":
        where_clauses.append("line_id = ?")
        params.append(line_id)
        
    if start_date:
        where_clauses.append("timestamp >= ?")
        params.append(f"{start_date} 00:00:00")
        
    if end_date:
        where_clauses.append("timestamp <= ?")
        params.append(f"{end_date} 23:59:59")
        
    if vehicle_type and vehicle_type != "all":
        where_clauses.append("vehicle_type = ?")
        params.append(vehicle_type)
        
    if direction and direction != "all":
        where_clauses.append("direction = ?")
        params.append(direction)

    if entity_type and entity_type.upper() != "ALL":
        where_clauses.append("UPPER(entity_type) = ?")
        params.append(entity_type.upper())
        
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Conteo total para paginación
        cursor.execute(f"SELECT COUNT(*) as total FROM traffic_events {where_sql}", params)
        total_count = cursor.fetchone()["total"]
        
        # Registros
        cursor.execute(f"""
            SELECT id, timestamp, camera_id, track_id, vehicle_type, direction, confidence, dwell_time, entity_type, line_id
            FROM traffic_events
            {where_sql}
            ORDER BY timestamp DESC, id DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset])
        
        events = [dict(r) for r in cursor.fetchall()]
        
        return {
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "events": events
        }

def seed_sample_data_if_empty() -> None:
    """Inserta datos históricos realistas de los últimos 7 días si la base de datos está vacía."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM traffic_events")
        if cursor.fetchone()["count"] == 0:
            logger_events = []
            now = datetime.datetime.now()
            cameras = ["cam1", "cam2"]
            vehicle_types = [
                ("Automovil", 0.45),
                ("Peaton", 0.28),
                ("Camion", 0.10),
                ("Autobus", 0.05),
                ("Motocicleta", 0.08),
                ("Bicicleta", 0.04)
            ]
            
            # Generar datos para los últimos 7 días
            for day_offset in range(6, -1, -1):
                target_day = now.date() - datetime.timedelta(days=day_offset)
                
                # Generar tráfico con curva de horas pico (8-10h y 18-20h)
                for hour in range(24):
                    if 7 <= hour <= 9:
                        hourly_volume = random.randint(35, 60) # Pico matutino
                    elif 17 <= hour <= 20:
                        hourly_volume = random.randint(40, 70) # Pico vespertino
                    elif 11 <= hour <= 16:
                        hourly_volume = random.randint(20, 35) # Flujo moderado
                    elif 0 <= hour <= 5:
                        hourly_volume = random.randint(1, 6)   # Flujo nocturno bajo
                    else:
                        hourly_volume = random.randint(10, 20)

                    # Si es hoy, no generar horas futuras
                    if day_offset == 0 and hour > now.hour:
                        continue
                    if day_offset == 0 and hour == now.hour:
                        hourly_volume = max(2, int(hourly_volume * (now.minute / 60.0)))

                    for _ in range(hourly_volume):
                        minute = random.randint(0, 59)
                        second = random.randint(0, 59)
                        if day_offset == 0 and hour == now.hour and minute > now.minute:
                            continue
                            
                        event_dt = datetime.datetime(target_day.year, target_day.month, target_day.day, hour, minute, second)
                        cam = random.choice(cameras)
                        track_id = random.randint(100, 9999)
                        
                        # Selección ponderada de tipo de vehículo o peatón
                        r = random.random()
                        cumulative = 0.0
                        v_type = "Automovil"
                        for v_name, prob in vehicle_types:
                            cumulative += prob
                            if r <= cumulative:
                                v_type = v_name
                                break
                                
                        direction = "IN" if random.random() > 0.48 else "OUT"
                        conf = round(random.uniform(0.70, 0.96), 2)
                        dwell = round(random.uniform(3.5, 30.0) if v_type == "Peaton" else random.uniform(2.5, 18.0), 1)
                        ent_type = "PEDESTRIAN" if v_type == "Peaton" else "VEHICLE"
                        
                        logger_events.append((
                            event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            cam,
                            track_id,
                            v_type,
                            direction,
                            conf,
                            dwell,
                            ent_type
                        ))
                        
            cursor.executemany("""
                INSERT INTO traffic_events (timestamp, camera_id, track_id, vehicle_type, direction, confidence, dwell_time, entity_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, logger_events)
            conn.commit()
        else:
            # Si ya existen eventos pero no hay peatones registrados, insertar una muestra para visualización del panel
            cursor.execute("SELECT COUNT(*) as count FROM traffic_events WHERE UPPER(entity_type) = 'PEDESTRIAN'")
            if cursor.fetchone()["count"] == 0:
                now = datetime.datetime.now()
                cameras = ["cam1", "cam2"]
                ped_events = []
                for day_offset in range(6, -1, -1):
                    target_day = now.date() - datetime.timedelta(days=day_offset)
                    for hour in range(8, 22):
                        if day_offset == 0 and hour > now.hour:
                            continue
                        vol = random.randint(10, 28)
                        for _ in range(vol):
                            minute = random.randint(0, 59)
                            second = random.randint(0, 59)
                            event_dt = datetime.datetime(target_day.year, target_day.month, target_day.day, hour, minute, second)
                            cam = random.choice(cameras)
                            track_id = random.randint(5000, 9999)
                            direction = "IN" if random.random() > 0.45 else "OUT"
                            conf = round(random.uniform(0.75, 0.95), 2)
                            dwell = round(random.uniform(5.0, 35.0), 1)
                            ped_events.append((
                                event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                cam,
                                track_id,
                                "Peaton",
                                direction,
                                conf,
                                dwell,
                                "PEDESTRIAN"
                            ))
                if ped_events:
                    cursor.executemany("""
                        INSERT INTO traffic_events (timestamp, camera_id, track_id, vehicle_type, direction, confidence, dwell_time, entity_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, ped_events)
                    conn.commit()

        # Sembrar o sincronizar avistamientos en camera_sightings (Métrica 1)
        cursor.execute("SELECT COUNT(*) as count FROM camera_sightings")
        if cursor.fetchone()["count"] == 0:
            # Todos los eventos de cruce corresponden a objetos vistos que cruzaron
            cursor.execute("""
                INSERT OR IGNORE INTO camera_sightings (timestamp, camera_id, track_id, vehicle_type, entity_type, confidence, crossed, line_id)
                SELECT timestamp, camera_id, track_id, vehicle_type, entity_type, confidence, 1, line_id
                FROM traffic_events
                GROUP BY camera_id, track_id, date(timestamp)
            """)
            conn.commit()

            # Añadir avistamientos realistas de vehículos y personas que estuvieron en encuadre pero NO cruzaron la línea
            cursor.execute("""
                SELECT DISTINCT date(timestamp) as event_date, camera_id, entity_type
                FROM traffic_events
            """)
            date_combos = cursor.fetchall()
            synthetic_sightings = []
            for row in date_combos:
                edate = row["event_date"]
                cam = row["camera_id"]
                ent = row["entity_type"]
                extra_count = random.randint(18, 45)
                for _ in range(extra_count):
                    hr = random.randint(7, 21)
                    mn = random.randint(0, 59)
                    sc = random.randint(0, 59)
                    ts_str = f"{edate} {hr:02d}:{mn:02d}:{sc:02d}"
                    tid = random.randint(20000, 99999)
                    v_type = "Peaton" if ent == "PEDESTRIAN" else random.choice(["Automovil", "Automovil", "Camion", "Motocicleta"])
                    conf = round(random.uniform(0.65, 0.94), 2)
                    synthetic_sightings.append((ts_str, cam, tid, v_type, ent, conf, 0))

            if synthetic_sightings:
                cursor.executemany("""
                    INSERT OR IGNORE INTO camera_sightings (timestamp, camera_id, track_id, vehicle_type, entity_type, confidence, crossed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, synthetic_sightings)
                conn.commit()

    # Sembrar ubicaciones y campañas si no existen
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM locations")
        if cursor.fetchone()["count"] == 0:
            cursor.execute("""
                INSERT INTO locations (id, name, address, novastar_ip, novastar_port, shelly_ip, shelly_channel, screen_area_m2, cost_per_kwh)
                VALUES 
                ('loc_minerva', 'Ubicación 01 — Glorieta Minerva', 'Av. Vallarta y López Mateos, Guadalajara', '192.168.1.140', 8001, '192.168.1.150', 0, 32.0, 3.85),
                ('loc_periferico', 'Ubicación 02 — Periférico Sur', 'Anillo Periférico Sur #4500, Tlaquepaque', '192.168.1.141', 8001, '192.168.1.151', 0, 50.0, 4.10)
            """)
            conn.commit()

        cursor.execute("SELECT COUNT(*) as count FROM campaigns")
        if cursor.fetchone()["count"] == 0:
            today = datetime.date.today()
            start_d = (today - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
            end_d = (today + datetime.timedelta(days=15)).strftime("%Y-%m-%d")
            cursor.execute("""
                INSERT INTO campaigns (name, client, location_id, start_date, end_date, start_hour, end_hour, spot_seconds, spots_per_hour, status)
                VALUES 
                ('Campaña Telcel 5G — Cobertura Total', 'Telcel / América Móvil', 'loc_minerva', ?, ?, 7, 22, 15, 12, 'Activa'),
                ('Campaña Coca-Cola Sin Azúcar', 'Coca-Cola FEMSA', 'loc_minerva', ?, ?, 6, 23, 20, 10, 'Activa'),
                ('Campaña Seguros de Auto Digital', 'BBVA México', 'loc_periferico', ?, ?, 8, 20, 15, 8, 'Activa')
            """, (start_d, end_d, start_d, end_d, start_d, end_d))
            conn.commit()

        # Sembrar clientes de muestra si no existen
        cursor.execute("SELECT COUNT(*) as count FROM clients")
        if cursor.fetchone()["count"] == 0:
            cursor.execute("""
                INSERT INTO clients (id, name, contact_email, logo_url)
                VALUES 
                ('cli_cerveceria', 'Cervecería Cuauhtémoc Moctezuma', 'marketing@heineken.com.mx', '/static/uploads/logos/default_cerveceria.svg'),
                ('cli_banco', 'Banco Azteca', 'publicidad@bancoazteca.com.mx', '/static/uploads/logos/default_banco.svg'),
                ('cli_retail', 'Liverpool México', 'dooh@liverpool.com.mx', '/static/uploads/logos/default_retail.svg'),
                ('cli_nissan', 'Nissan México', 'marcas@nissan.com.mx', '/static/uploads/logos/default_automotriz.svg')
            """)
            cursor.execute("""
                INSERT OR IGNORE INTO client_locations (client_id, location_id)
                VALUES 
                ('cli_cerveceria', 'loc_minerva'),
                ('cli_cerveceria', 'loc_periferico'),
                ('cli_banco', 'loc_periferico'),
                ('cli_retail', 'loc_minerva'),
                ('cli_nissan', 'loc_minerva'),
                ('cli_nissan', 'loc_periferico')
            """)
            conn.commit()

        # Sembrar controladores y procesadores de hardware si no existen
        cursor.execute("SELECT COUNT(*) as count FROM hardware_devices")
        if cursor.fetchone()["count"] == 0:
            cursor.execute("""
                INSERT INTO hardware_devices (id, location_id, name, device_type, ip_address, port, dual_screen_enabled, active_preset, input_source_1, input_source_2, notes)
                VALUES 
                ('dev_tb40_minerva', 'loc_minerva', 'NovaStar TB40 Maestro — Minerva', 'TB40', '192.168.1.140', 8001, 0, 'preset_single_tb40', 'TB40_MASTER', '', 'Reproductor multimedia primario para pantalla exterior'),
                ('dev_vx600_periferico', 'loc_periferico', 'NovaStar VX600 Pro — Periférico Sur (2 Caras)', 'VX600_PRO', '192.168.1.160', 6000, 1, 'preset_mirror_1tb40', 'HDMI-1 (TB40-A)', 'HDMI-2 (TB40-B)', 'Controlador y escalador para espectacular bipolar Cara A y Cara B'),
                ('dev_shelly_minerva', 'loc_minerva', 'Shelly Pro 4PM — Minerva', 'SHELLY_PRO', '192.168.1.150', 80, 0, '', '', '', 'Monitoreo de energía eléctrica, potencia kW y encendido remoto'),
                ('dev_shelly_periferico', 'loc_periferico', 'Shelly Pro 4PM — Periférico', 'SHELLY_PRO', '192.168.1.151', 80, 0, '', '', '', 'Medición de consumo kWh y encendido de pantalla')
            """)
            conn.commit()

        # Sembrar campañas asociadas a clientes y spots de video si la tabla está vacía
        cursor.execute("SELECT COUNT(*) as count FROM campaign_videos")
        if cursor.fetchone()["count"] == 0:
            today = datetime.date.today()
            start_d = (today - datetime.timedelta(days=20)).strftime("%Y-%m-%d")
            end_d = (today + datetime.timedelta(days=20)).strftime("%Y-%m-%d")

            cursor.execute("""
                INSERT OR REPLACE INTO campaigns (id, name, client, client_id, location_id, start_date, end_date, start_hour, end_hour, spot_seconds, spots_per_hour, status)
                VALUES 
                (10, 'Campaña Cerveza Minerva — Edición Artesanal Verano', 'Cervecería Cuauhtémoc Moctezuma', 'cli_cerveceria', 'loc_minerva', ?, ?, 6, 23, 15, 12, 'Activa'),
                (11, 'Campaña Bohemia Cristal — Maridaje DOOH', 'Cervecería Cuauhtémoc Moctezuma', 'cli_cerveceria', 'loc_periferico', ?, ?, 7, 22, 15, 10, 'Activa'),
                (12, 'Lanzamiento Nacional Nuevo Nissan Kicks e-POWER', 'Nissan México', 'cli_nissan', 'loc_minerva', ?, ?, 6, 23, 20, 10, 'Activa'),
                (13, 'Nissan Intelligent Mobility — Seguridad 360', 'Nissan México', 'cli_nissan', 'loc_periferico', ?, ?, 7, 22, 15, 8, 'Activa'),
                (14, 'Crédito Nómina y Cuenta Digital Banco Azteca', 'Banco Azteca', 'cli_banco', 'loc_periferico', ?, ?, 8, 21, 15, 12, 'Activa'),
                (15, 'Gran Venta Nocturna Aniversario Liverpool', 'Liverpool México', 'cli_retail', 'loc_minerva', ?, ?, 6, 23, 15, 14, 'Activa')
            """, (start_d, end_d, start_d, end_d, start_d, end_d, start_d, end_d, start_d, end_d, start_d, end_d))

            cursor.execute("""
                INSERT INTO campaign_videos (campaign_id, client_id, video_name, title, duration_seconds, resolution, fps, plays_today, total_plays, total_impressions, kwh_consumed, status)
                VALUES
                (10, 'cli_cerveceria', 'minerva_ipa_artesanal_15s.mp4', 'Minerva IPA — Especial Verano', 15, '1920x1080', 60, 194, 3880, 18450, 4.8, 'En rotación'),
                (10, 'cli_cerveceria', 'minerva_stout_imperial_20s.mp4', 'Minerva Stout Imperial Gourmet', 20, '1920x1080', 60, 142, 2840, 13520, 3.5, 'En rotación'),
                (11, 'cli_cerveceria', 'bohemia_cristal_refrescante_15s.mp4', 'Bohemia Cristal — Refrescante por Naturaleza', 15, '1920x1080', 60, 168, 3360, 16100, 4.1, 'En rotación'),
                (12, 'cli_nissan', 'nissan_kicks_epower_aceleracion_20s.mp4', 'Nissan Kicks e-POWER — Aceleración 100% Eléctrica', 20, '1920x1080', 60, 180, 3600, 21500, 5.4, 'En rotación'),
                (13, 'cli_nissan', 'nissan_intelligent_mobility_15s.mp4', 'Nissan Intelligent Mobility — Seguridad 360', 15, '1920x1080', 60, 150, 3000, 16200, 3.9, 'En rotación'),
                (14, 'cli_banco', 'banco_azteca_credito_nomina_15s.mp4', 'Crédito Nómina Inmediato en tu App', 15, '1920x1080', 60, 175, 3500, 17800, 4.3, 'En rotación'),
                (15, 'cli_retail', 'liverpool_venta_nocturna_ofertas_15s.mp4', 'Gran Venta Nocturna — Exclusivo Tarjetas', 15, '1920x1080', 60, 210, 4200, 24300, 5.9, 'En rotación')
            """)
            conn.commit()

        # Sembrar usuarios por defecto (Administrador y Cliente de demostración) si no existen
        cursor.execute("SELECT COUNT(*) as count FROM users")
        if cursor.fetchone()["count"] == 0:
            admin_pwd = auth_service.hash_password("admin123")
            client_pwd = auth_service.hash_password("cliente123")
            cursor.execute("""
                INSERT INTO users (id, email, nombre, password_hash, rol, cliente_id, activo)
                VALUES 
                ('usr_admin_master', 'admin@apexcompany.com.mx', 'Administrador Apex', ?, 'admin', NULL, 1),
                ('usr_cliente_cerveceria', 'marketing@heineken.com.mx', 'Gerente DOOH Heineken/Minerva', ?, 'cliente', 'cli_cerveceria', 1)
            """, (admin_pwd, client_pwd))
            conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE GESTIÓN MULTI-UBICACIÓN Y CAMPAÑAS
# ─────────────────────────────────────────────────────────────────────────────

def get_locations() -> list[dict[str, Any]]:
    """Obtiene todas las ubicaciones/sitios de pantallas registradas."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM locations ORDER BY name ASC")
        return [dict(r) for r in cursor.fetchall()]

def get_location(loc_id: str) -> Optional[dict[str, Any]]:
    """Obtiene el detalle de una ubicación por su ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM locations WHERE id = ?", (loc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def add_location(
    loc_id: str,
    name: str,
    address: str = "",
    novastar_ip: str = "192.168.1.140",
    novastar_port: int = 8001,
    shelly_ip: str = "192.168.1.150",
    shelly_channel: int = 0,
    screen_area_m2: float = 32.0,
    cost_per_kwh: float = 3.85
) -> None:
    """Registra una nueva ubicación / nodo de pantalla."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO locations 
            (id, name, address, novastar_ip, novastar_port, shelly_ip, shelly_channel, screen_area_m2, cost_per_kwh)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (loc_id, name, address, novastar_ip, novastar_port, shelly_ip, shelly_channel, screen_area_m2, cost_per_kwh))
        conn.commit()

def update_location_solar_config(
    loc_id: str,
    screen_orientation_deg: float,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    min_night_brightness: Optional[int] = None,
    max_day_brightness: Optional[int] = None,
    auto_brightness_enabled: Optional[int] = None
) -> bool:
    """Actualiza los parámetros solares, orientación y límites de brillo de una ubicación."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM locations WHERE id = ?", (loc_id,))
        current = cursor.fetchone()
        if not current:
            return False

        lat = latitude if latitude is not None else (current["latitude"] if "latitude" in current.keys() else 20.6736)
        lon = longitude if longitude is not None else (current["longitude"] if "longitude" in current.keys() else -103.3855)
        min_b = min_night_brightness if min_night_brightness is not None else (current["min_night_brightness"] if "min_night_brightness" in current.keys() else 25)
        max_b = max_day_brightness if max_day_brightness is not None else (current["max_day_brightness"] if "max_day_brightness" in current.keys() else 95)
        auto_b = auto_brightness_enabled if auto_brightness_enabled is not None else (current["auto_brightness_enabled"] if "auto_brightness_enabled" in current.keys() else 1)

        cursor.execute("""
            UPDATE locations 
            SET screen_orientation_deg = ?, latitude = ?, longitude = ?, min_night_brightness = ?, max_day_brightness = ?, auto_brightness_enabled = ?
            WHERE id = ?
        """, (screen_orientation_deg, lat, lon, min_b, max_b, auto_b, loc_id))
        conn.commit()
        return True

def get_campaigns(location_id: Optional[str] = None, client_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Obtiene las campañas publicitarias filtradas opcionalmente por ubicación o cliente."""
    with get_connection() as conn:
        cursor = conn.cursor()
        query = """
            SELECT c.*, l.name as location_name, l.screen_area_m2, l.cost_per_kwh,
                   cl.name as client_name, cl.logo_url as client_logo
            FROM campaigns c
            JOIN locations l ON c.location_id = l.id
            LEFT JOIN clients cl ON c.client_id = cl.id
            WHERE 1=1
        """
        params: list[Any] = []
        if location_id and location_id != "all":
            query += " AND c.location_id = ?"
            params.append(location_id)
        if client_id and client_id != "all":
            query += " AND c.client_id = ?"
            params.append(client_id)
        query += " ORDER BY c.id DESC"
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]

def get_campaign(campaign_id: int) -> Optional[dict[str, Any]]:
    """Obtiene el detalle de una campaña por ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.*, l.name as location_name, l.screen_area_m2, l.cost_per_kwh,
                   cl.name as client_name, cl.logo_url as client_logo
            FROM campaigns c
            JOIN locations l ON c.location_id = l.id
            LEFT JOIN clients cl ON c.client_id = cl.id
            WHERE c.id = ?
        """, (campaign_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_campaign_videos(
    campaign_id: Optional[int] = None,
    client_id: Optional[str] = None,
    location_id: Optional[str] = None
) -> list[dict[str, Any]]:
    """Obtiene los spots y videos publicitarios desglosados por campaña, cliente o pantalla."""
    with get_connection() as conn:
        cursor = conn.cursor()
        query = """
            SELECT v.*, c.name as campaign_name, c.location_id,
                   l.name as location_name, l.screen_area_m2, l.cost_per_kwh,
                   cl.name as client_name, cl.logo_url as client_logo
            FROM campaign_videos v
            JOIN campaigns c ON v.campaign_id = c.id
            JOIN locations l ON c.location_id = l.id
            JOIN clients cl ON v.client_id = cl.id
            WHERE 1=1
        """
        params: list[Any] = []
        if campaign_id:
            query += " AND v.campaign_id = ?"
            params.append(campaign_id)
        if client_id and client_id != "all":
            query += " AND v.client_id = ?"
            params.append(client_id)
        if location_id and location_id != "all":
            query += " AND c.location_id = ?"
            params.append(location_id)
        query += " ORDER BY v.id ASC"
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]

def add_campaign_video(
    campaign_id: int,
    client_id: str,
    video_name: str,
    title: str,
    duration_seconds: int = 15,
    resolution: str = "1920x1080",
    fps: int = 60,
    status: str = "En rotación"
) -> int:
    """Registra un nuevo video/spot dentro de una campaña."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO campaign_videos (campaign_id, client_id, video_name, title, duration_seconds, resolution, fps, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (campaign_id, client_id, video_name, title, duration_seconds, resolution, fps, status))
        conn.commit()
        return cursor.lastrowid or 0

def get_client_campaign_report(
    client_id: str,
    campaign_id: Optional[int] = None,
    location_id: Optional[str] = None,
    video_id: Optional[int] = None
) -> dict[str, Any]:
    """
    Genera el reporte ejecutivo completo para el cliente organizado por:
    - Campaña
    - Pantalla (ubicación)
    - Desglose por Videos / Spots (o un video específico)
    Y presenta los datos limpios de consumo eléctrico de la pantalla (kW, kWh, $ MXN)
    sin exponer nombres técnicos de hardware interno (Shelly, puertos, relés).
    """
    client = get_client(client_id)
    if not client:
        clients_list = get_clients()
        client = clients_list[0] if clients_list else {"id": client_id, "name": "Cliente Corporativo", "logo_url": ""}

    c_id = client["id"]
    screens = get_client_screens(c_id)
    if location_id and location_id != "all":
        screens = [s for s in screens if s["id"] == location_id]

    campaigns = get_campaigns(location_id=location_id, client_id=c_id)
    if campaign_id:
        campaigns = [c for c in campaigns if c["id"] == campaign_id]
        
    videos = get_campaign_videos(campaign_id=campaign_id, client_id=c_id, location_id=location_id)
    if video_id:
        videos = [v for v in videos if v["id"] == video_id]
    
    # Calcular métricas consolidadas
    total_spots_today = sum(v.get("plays_today", 0) for v in videos)
    total_spots_all = sum(v.get("total_plays", 0) for v in videos)
    total_impressions = sum(v.get("total_impressions", 0) for v in videos)
    total_duration_sec = sum(v.get("total_plays", 0) * v.get("duration_seconds", 15) for v in videos)
    exposure_hours = round(total_duration_sec / 3600.0, 1)

    # Consumo eléctrico de la pantalla
    total_kwh = round(sum(v.get("kwh_consumed", 0.0) for v in videos), 2)
    client_cfe_rate = float(client.get("kwh_rate_cfe") or 3.85)
    avg_kwh_cost = client_cfe_rate
    if screens:
        avg_kwh_cost = sum(float(s.get("cost_per_kwh", client_cfe_rate)) for s in screens) / len(screens)
    energy_cost_mxn = round(total_kwh * avg_kwh_cost, 2)
    
    # Calcular costo para cada video individual usando tarifa CFE real
    for v in videos:
        v_rate = float(v.get("cost_per_kwh") or avg_kwh_cost)
        v["cost_mxn"] = round(float(v.get("kwh_consumed", 0.0)) * v_rate, 2)
        v["effective_kwh_rate"] = v_rate

    # Potencia de pantalla estimada
    screen_area_total = sum(s.get("screen_area_m2", 32.0) for s in screens) or 32.0
    estimated_power_kw = round((screen_area_total / 32.0) * 4.2, 2)

    return {
        "client": client,
        "client_name": client.get("name", ""),
        "filters": {
            "campaign_id": campaign_id,
            "location_id": location_id
        },
        "screens": screens,
        "campaigns": campaigns,
        "videos": videos,
        "summary": {
            "total_campaigns": len(campaigns),
            "total_videos": len(videos),
            "total_spots_today": total_spots_today,
            "today_plays": total_spots_today,
            "total_spots_campaign": total_spots_all,
            "total_plays": total_spots_all,
            "total_impressions": total_impressions,
            "estimated_reach_vehicles": int(total_impressions / 1.6),
            "total_exposure_hours": exposure_hours,
            "screen_time_hours": exposure_hours,
            "screen_power_kw": estimated_power_kw,
            "screen_kw": estimated_power_kw,
            "campaign_energy_kwh": total_kwh,
            "energy_cost_mxn": energy_cost_mxn,
            "campaign_cost_mxn": energy_cost_mxn,
            "cost_per_kwh": round(avg_kwh_cost, 2),
            "screen_area_m2": round(screen_area_total, 1)
        }
    }

def add_campaign(
    name: str,
    client: str,
    location_id: str,
    start_date: str,
    end_date: str,
    start_hour: int = 6,
    end_hour: int = 23,
    spot_seconds: int = 15,
    spots_per_hour: int = 12,
    status: str = "Activa",
    client_id: Optional[str] = None
) -> int:
    """Crea una nueva campaña publicitaria."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO campaigns (name, client, client_id, location_id, start_date, end_date, start_hour, end_hour, spot_seconds, spots_per_hour, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, client, client_id, location_id, start_date, end_date, start_hour, end_hour, spot_seconds, spots_per_hour, status))
        conn.commit()
        return cursor.lastrowid or 0

def record_edge_sync(
    node_id: str,
    location_id: str,
    date_reported: str,
    total_vehicles: int,
    total_in: int,
    total_out: int,
    peak_hour: str,
    raw_summary: str = ""
) -> int:
    """Registra la recepción de un reporte ligero sincronizado desde un nodo local."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO edge_sync_reports (node_id, location_id, date_reported, total_vehicles, total_in, total_out, peak_hour, raw_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (node_id, location_id, date_reported, total_vehicles, total_in, total_out, peak_hour, raw_summary))
        conn.commit()
        return cursor.lastrowid or 0


# ─────────────────────────────────────────────────────────────────────────────
# GESTIÓN MULTI-CLIENTE Y LOGOTIPOS
# ─────────────────────────────────────────────────────────────────────────────

def get_clients() -> list[dict[str, Any]]:
    """Obtiene todos los clientes registrados junto con sus ubicaciones asignadas."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM clients ORDER BY name ASC")
        clients = [dict(r) for r in cursor.fetchall()]
        for c in clients:
            cursor.execute("""
                SELECT l.* FROM locations l
                JOIN client_locations cl ON l.id = cl.location_id
                WHERE cl.client_id = ?
            """, (c["id"],))
            c["locations"] = [dict(loc) for loc in cursor.fetchall()]
        return clients

def get_client(client_id: str) -> Optional[dict[str, Any]]:
    """Obtiene los datos de un cliente por su ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM clients WHERE id = ?", (client_id,))
        row = cursor.fetchone()
        if not row:
            return None
        c = dict(row)
        cursor.execute("""
            SELECT l.* FROM locations l
            JOIN client_locations cl ON l.id = cl.location_id
            WHERE cl.client_id = ?
        """, (client_id,))
        c["locations"] = [dict(loc) for loc in cursor.fetchall()]
        return c

def add_client(client_id: str, name: str, contact_email: str = "", logo_url: str = "", kwh_rate_cfe: float = 3.85) -> str:
    """Registra o actualiza un cliente comercial junto con su tarifa CFE por kWh."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO clients (id, name, contact_email, logo_url, kwh_rate_cfe)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                contact_email = excluded.contact_email,
                logo_url = CASE WHEN excluded.logo_url != '' THEN excluded.logo_url ELSE clients.logo_url END,
                kwh_rate_cfe = CASE WHEN excluded.kwh_rate_cfe > 0 THEN excluded.kwh_rate_cfe ELSE clients.kwh_rate_cfe END
        """, (client_id, name, contact_email, logo_url, kwh_rate_cfe))
        conn.commit()
        return client_id

def update_client_kwh_rate(client_id: str, kwh_rate_cfe: float, location_id: Optional[str] = None) -> bool:
    """
    Actualiza la tarifa de energía que CFE le cobra al cliente.
    Si se especifica location_id, actualiza la tarifa específica para esa pantalla en client_locations.
    En caso contrario o adicionalmente, actualiza la tarifa base global del cliente.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        if location_id and location_id != "all":
            cursor.execute("""
                UPDATE client_locations SET custom_kwh_rate = ?
                WHERE client_id = ? AND location_id = ?
            """, (kwh_rate_cfe, client_id, location_id))
            # Si no existe la asociación, se crea
            if cursor.rowcount == 0:
                cursor.execute("""
                    INSERT INTO client_locations (client_id, location_id, custom_kwh_rate)
                    VALUES (?, ?, ?)
                """, (client_id, location_id, kwh_rate_cfe))

        # Actualizar también tarifa base en la ficha del cliente
        cursor.execute("UPDATE clients SET kwh_rate_cfe = ? WHERE id = ?", (kwh_rate_cfe, client_id))
        conn.commit()
        return True

def update_client_logo(client_id: str, logo_url: str) -> bool:
    """Actualiza la URL o ruta del logotipo del cliente."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE clients SET logo_url = ? WHERE id = ?", (logo_url, client_id))
        conn.commit()
        return cursor.rowcount > 0

def assign_client_location(client_id: str, location_id: str, custom_kwh_rate: Optional[float] = None) -> bool:
    """Asocia una pantalla/ubicación a un cliente con tarifa opcional personalizada."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO client_locations (client_id, location_id, custom_kwh_rate)
            VALUES (?, ?, ?)
        """, (client_id, location_id, custom_kwh_rate))
        conn.commit()
        return True

def get_client_screens(client_id: str) -> list[dict[str, Any]]:
    """Devuelve las ubicaciones y pantallas autorizadas para un cliente con la tarifa real CFE aplicada."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT l.*, 
                   COALESCE(cl.custom_kwh_rate, c.kwh_rate_cfe, l.cost_per_kwh, 3.85) as cost_per_kwh,
                   COALESCE(cl.custom_kwh_rate, c.kwh_rate_cfe, l.cost_per_kwh, 3.85) as effective_kwh_rate,
                   c.kwh_rate_cfe as client_cfe_rate
            FROM locations l
            JOIN client_locations cl ON l.id = cl.location_id
            JOIN clients c ON cl.client_id = c.id
            WHERE cl.client_id = ? AND l.is_active = 1
            ORDER BY l.name ASC
        """, (client_id,))
        return [dict(r) for r in cursor.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# GESTIÓN DE DISPOSITIVOS DE HARDWARE (TB40 / VX600 PRO / SHELLY)
# ─────────────────────────────────────────────────────────────────────────────

def get_hardware_devices(location_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Lista los dispositivos de hardware registrados, opcionalmente filtrados por ubicación."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if location_id and location_id != "all":
            cursor.execute("""
                SELECT d.*, l.name as location_name 
                FROM hardware_devices d
                JOIN locations l ON d.location_id = l.id
                WHERE d.location_id = ?
                ORDER BY d.created_at DESC
            """, (location_id,))
        else:
            cursor.execute("""
                SELECT d.*, l.name as location_name 
                FROM hardware_devices d
                JOIN locations l ON d.location_id = l.id
                ORDER BY d.created_at DESC
            """)
        return [dict(r) for r in cursor.fetchall()]

def get_hardware_device(device_id: str) -> Optional[dict[str, Any]]:
    """Obtiene el detalle de un dispositivo por su ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.*, l.name as location_name 
            FROM hardware_devices d
            JOIN locations l ON d.location_id = l.id
            WHERE d.id = ?
        """, (device_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def add_hardware_device(
    device_id: str,
    location_id: str,
    name: str,
    device_type: str,
    ip_address: str,
    port: int = 8001,
    dual_screen_enabled: int = 0,
    active_preset: str = "preset_mirror_1tb40",
    input_source_1: str = "TB40_MASTER",
    input_source_2: str = "",
    notes: str = ""
) -> str:
    """Registra o actualiza un dispositivo de hardware en la base de datos."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO hardware_devices 
            (id, location_id, name, device_type, ip_address, port, dual_screen_enabled, active_preset, input_source_1, input_source_2, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (device_id, location_id, name, device_type, ip_address, port, dual_screen_enabled, active_preset, input_source_1, input_source_2, notes))
        conn.commit()
        return device_id

def update_hardware_device_preset(device_id: str, preset_id: str) -> bool:
    """Actualiza el preset activo de un procesador de video (VX600 Pro)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE hardware_devices SET active_preset = ? WHERE id = ?", (preset_id, device_id))
        conn.commit()
        return cursor.rowcount > 0

def delete_hardware_device(device_id: str) -> bool:
    """Elimina un dispositivo de hardware del registro."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM hardware_devices WHERE id = ?", (device_id,))
        conn.commit()
        return cursor.rowcount > 0


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO DE MONITOREO DE HARDWARE: CONEXIONES Y DESCONEXIONES DE CÁMARAS
# ─────────────────────────────────────────────────────────────────────────────

def record_camera_connection_event(
    camera_id: str,
    event_type: str,
    status: str,
    source: str = "",
    details: str = "",
    duration_offline_sec: float = 0.0,
    duration_online_sec: float = 0.0,
    event_time: Optional[datetime.datetime] = None
) -> int:
    """
    Registra un evento de conectividad de cámara (CONNECTED, DISCONNECTED, RECONNECTING, ERROR, ENABLED, DISABLED).
    """
    if event_time is None:
        event_time = datetime.datetime.now()
    time_str = event_time.strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO camera_connection_logs 
            (timestamp, camera_id, event_type, status, source, details, duration_offline_sec, duration_online_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (time_str, camera_id, event_type.upper(), status.upper(), str(source), str(details), round(float(duration_offline_sec), 1), round(float(duration_online_sec), 1)))
        conn.commit()
        return cursor.lastrowid or 0


def get_camera_connection_logs(
    camera_id: str = "all",
    limit: int = 100,
    offset: int = 0,
    date_str: Optional[str] = None,
    event_type: str = "all"
) -> dict:
    """
    Obtiene la bitácora de eventos de conectividad de cámaras con filtros y duración formateada.
    """
    query = "SELECT * FROM camera_connection_logs WHERE 1=1"
    params: list[Any] = []
    
    if camera_id != "all":
        query += " AND camera_id = ?"
        params.append(camera_id)
        
    if date_str:
        query += " AND date(timestamp) = ?"
        params.append(date_str)
        
    if event_type != "all":
        query += " AND event_type = ?"
        params.append(event_type.upper())
        
    count_query = query.replace("SELECT *", "SELECT COUNT(*)")
    
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params_with_paging = list(params) + [limit, offset]
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]
        
        cursor.execute(query, params_with_paging)
        rows = cursor.fetchall()
        
        def format_sec(s: float) -> str:
            if s <= 0:
                return "--"
            if s < 60:
                return f"{int(s)}s"
            m = int(s // 60)
            sec = int(s % 60)
            if m < 60:
                return f"{m}m {sec}s"
            h = int(m // 60)
            m = int(m % 60)
            return f"{h}h {m}m"

        records = []
        for r in rows:
            dur_off = r["duration_offline_sec"] or 0.0
            dur_on = r["duration_online_sec"] or 0.0
            records.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "camera_id": r["camera_id"],
                "event_type": r["event_type"],
                "status": r["status"],
                "source": r["source"] or "",
                "details": r["details"] or "",
                "duration_offline_sec": dur_off,
                "duration_online_sec": dur_on,
                "duration_offline_formatted": format_sec(dur_off),
                "duration_online_formatted": format_sec(dur_on),
            })
            
        return {
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "records": records
        }


def get_camera_connection_stats(camera_id: str = "all", date_str: Optional[str] = None) -> dict:
    """
    Retorna métricas de disponibilidad (uptime, caídas, downtime) para hoy o una fecha dada.
    """
    if not date_str:
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        
    with get_connection() as conn:
        cursor = conn.cursor()
        
        where = "WHERE date(timestamp) = ?"
        params: list[Any] = [date_str]
        if camera_id != "all":
            where += " AND camera_id = ?"
            params.append(camera_id)
            
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_events,
                SUM(CASE WHEN event_type = 'DISCONNECTED' THEN 1 ELSE 0 END) as disconnect_count,
                SUM(CASE WHEN event_type = 'CONNECTED' THEN 1 ELSE 0 END) as connect_count,
                SUM(duration_offline_sec) as total_offline_sec,
                SUM(duration_online_sec) as total_online_sec
            FROM camera_connection_logs
            {where}
        """, params)
        row = cursor.fetchone()
        
        total_events = row["total_events"] or 0
        disconnect_count = row["disconnect_count"] or 0
        connect_count = row["connect_count"] or 0
        total_offline_sec = row["total_offline_sec"] or 0.0
        total_online_sec = row["total_online_sec"] or 0.0
        
        # Último evento registrado
        last_event_query = f"SELECT * FROM camera_connection_logs {where} ORDER BY id DESC LIMIT 1"
        cursor.execute(last_event_query, params)
        last_event = cursor.fetchone()
        last_event_dict = dict(last_event) if last_event else None
        
        # Desglose por cámara
        cam_where = "WHERE date(timestamp) = ?"
        cursor.execute(f"""
            SELECT 
                camera_id,
                COUNT(*) as total_events,
                SUM(CASE WHEN event_type = 'DISCONNECTED' THEN 1 ELSE 0 END) as disconnects,
                SUM(duration_offline_sec) as downtime_sec
            FROM camera_connection_logs
            {cam_where}
            GROUP BY camera_id
        """, [date_str])
        by_camera = {}
        for cr in cursor.fetchall():
            by_camera[cr["camera_id"]] = {
                "total_events": cr["total_events"],
                "disconnects": cr["disconnects"] or 0,
                "downtime_sec": round(cr["downtime_sec"] or 0.0, 1)
            }
            
        now = datetime.datetime.now()
        seconds_so_far_today = (now.hour * 3600) + (now.minute * 60) + now.second
        if seconds_so_far_today <= 0:
            seconds_so_far_today = 86400
            
        uptime_pct = 100.0
        if total_offline_sec > 0:
            uptime_pct = max(0.0, min(100.0, 100.0 * (1.0 - (total_offline_sec / max(1.0, seconds_so_far_today)))))
            
        return {
            "date": date_str,
            "camera_id": camera_id,
            "total_events": total_events,
            "disconnect_count": disconnect_count,
            "connect_count": connect_count,
            "total_offline_sec": round(total_offline_sec, 1),
            "total_online_sec": round(total_online_sec, 1),
            "uptime_percentage": round(uptime_pct, 2),
            "last_event": last_event_dict,
            "by_camera": by_camera
        }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE GESTIÓN DE USUARIOS, AUTENTICACIÓN E INVITACIONES
# ─────────────────────────────────────────────────────────────────────────────

def create_user(
    email: str,
    nombre: str,
    password_hash: str,
    rol: str = "cliente",
    cliente_id: Optional[str] = None,
    activo: int = 1
) -> dict[str, Any]:
    """Crea un nuevo usuario en la plataforma."""
    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (id, email, nombre, password_hash, rol, cliente_id, activo)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, email.strip().lower(), nombre.strip(), password_hash, rol, cliente_id, activo))
        conn.commit()
    return get_user_by_id(user_id)  # type: ignore


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    """Busca un usuario por su correo electrónico."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.*, c.name as cliente_nombre
            FROM users u
            LEFT JOIN clients c ON u.cliente_id = c.id
            WHERE LOWER(u.email) = LOWER(?)
        """, (email.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    """Busca un usuario por su ID primario."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.*, c.name as cliente_nombre
            FROM users u
            LEFT JOIN clients c ON u.cliente_id = c.id
            WHERE u.id = ?
        """, (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_all_users() -> list[dict[str, Any]]:
    """Retorna la lista de todos los usuarios registrados (sin password_hash)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.id, u.email, u.nombre, u.rol, u.cliente_id, u.activo, u.creado_en, c.name as cliente_nombre
            FROM users u
            LEFT JOIN clients c ON u.cliente_id = c.id
            ORDER BY u.creado_en DESC
        """)
        return [dict(r) for r in cursor.fetchall()]


def toggle_user_status(user_id: str) -> bool:
    """Activa o desactiva un usuario existente."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET activo = CASE WHEN activo = 1 THEN 0 ELSE 1 END WHERE id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount > 0


def change_user_role(user_id: str, new_role: str) -> bool:
    """Modifica el rol de un usuario ('admin' o 'cliente')."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET rol = ? WHERE id = ?", (new_role, user_id))
        conn.commit()
        return cursor.rowcount > 0


def delete_user(user_id: str) -> bool:
    """Elimina permanentemente un usuario."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount > 0


def create_invitation(
    email: str,
    nombre: str,
    rol: str = "cliente",
    cliente_id: Optional[str] = None,
    token: Optional[str] = None,
    hours_valid: int = 168
) -> dict[str, Any]:
    """
    Crea un registro de invitación con token criptográfico válido por X horas (default 7 días).
    """
    inv_id = f"inv_{uuid.uuid4().hex[:12]}"
    if not token:
        token = auth_service.generate_invitation_token()
        
    expira_dt = datetime.datetime.now() + datetime.timedelta(hours=hours_valid)
    expira_str = expira_dt.strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO invitations (id, email, nombre, rol, cliente_id, token, estado, expira_en)
            VALUES (?, ?, ?, ?, ?, ?, 'pendiente', ?)
        """, (inv_id, email.strip().lower(), nombre.strip(), rol, cliente_id, token, expira_str))
        conn.commit()
        
    return get_invitation_by_token(token)  # type: ignore


def get_invitation_by_token(token: str) -> Optional[dict[str, Any]]:
    """Obtiene el detalle de una invitación a través de su token."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT i.*, c.name as cliente_nombre
            FROM invitations i
            LEFT JOIN clients c ON i.cliente_id = c.id
            WHERE i.token = ?
        """, (token,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            expira_dt = datetime.datetime.strptime(res["expira_en"], "%Y-%m-%d %H:%M:%S")
            if datetime.datetime.now() > expira_dt and res["estado"] == "pendiente":
                res["estado"] = "expirada"
        except Exception:
            pass
        return res


def get_all_invitations() -> list[dict[str, Any]]:
    """Lista todas las invitaciones generadas por el administrador."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT i.*, c.name as cliente_nombre
            FROM invitations i
            LEFT JOIN clients c ON i.cliente_id = c.id
            ORDER BY i.creado_en DESC
        """)
        rows = cursor.fetchall()
        inv_list = []
        now = datetime.datetime.now()
        for r in rows:
            d = dict(r)
            try:
                expira_dt = datetime.datetime.strptime(d["expira_en"], "%Y-%m-%d %H:%M:%S")
                if now > expira_dt and d["estado"] == "pendiente":
                    d["estado"] = "expirada"
            except Exception:
                pass
            inv_list.append(d)
        return inv_list


def accept_invitation(token: str, password_hash: str) -> Optional[dict[str, Any]]:
    """
    Acepta una invitación válida, crea o actualiza la cuenta de usuario con la contraseña
    proporcionada y marca la invitación como 'aceptada'.
    """
    inv = get_invitation_by_token(token)
    if not inv:
        return None
    if inv["estado"] != "pendiente":
        return None

    existing_user = get_user_by_email(inv["email"])
    with get_connection() as conn:
        cursor = conn.cursor()
        if existing_user:
            cursor.execute("""
                UPDATE users 
                SET password_hash = ?, nombre = ?, rol = ?, cliente_id = ?, activo = 1
                WHERE email = ?
            """, (password_hash, inv["nombre"], inv["rol"], inv["cliente_id"], inv["email"]))
            user_id = existing_user["id"]
        else:
            user_id = f"usr_{uuid.uuid4().hex[:12]}"
            cursor.execute("""
                INSERT INTO users (id, email, nombre, password_hash, rol, cliente_id, activo)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (user_id, inv["email"], inv["nombre"], password_hash, inv["rol"], inv["cliente_id"]))
            
        cursor.execute("UPDATE invitations SET estado = 'aceptada' WHERE token = ?", (token,))
        conn.commit()

    return get_user_by_id(user_id)




