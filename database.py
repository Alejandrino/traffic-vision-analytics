"""
database.py — Motor de persistencia SQLite para analítica y aforo vehicular.

Almacena y consulta eventos históricos de cruce de vehículos para auditoría,
métricas horarias, detección de horas pico y reporteo ejecutivo para toma de decisiones.
"""

import sqlite3
import datetime
import random
import os
from typing import Optional, Any
from pathlib import Path

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
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── TABLA DE ASOCIACIÓN CLIENTE ↔ PANTALLAS / UBICACIONES ────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_locations (
                client_id TEXT NOT NULL,
                location_id TEXT NOT NULL,
                PRIMARY KEY (client_id, location_id),
                FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE,
                FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE CASCADE
            )
        """)

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
    event_time: Optional[datetime.datetime] = None
) -> int:
    """Registra un evento de cruce de línea en la base de datos."""
    if event_time is None:
        event_time = datetime.datetime.now()
    
    time_str = event_time.strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO traffic_events (timestamp, camera_id, track_id, vehicle_type, direction, confidence, dwell_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (time_str, camera_id, track_id, vehicle_type, round(confidence, 2), round(dwell_time, 1), round(dwell_time, 1)))
        conn.commit()
        return cursor.lastrowid or 0

def get_kpis(camera_id: Optional[str] = None, date_str: Optional[str] = None) -> dict[str, Any]:
    """Obtiene indicadores clave de rendimiento (KPIs) para una fecha o globales."""
    if date_str is None:
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        
    where_clauses = ["date(timestamp) = ?"]
    params: list[Any] = [date_str]
    
    if camera_id and camera_id != "all":
        where_clauses.append("camera_id = ?")
        params.append(camera_id)
        
    where_sql = " AND ".join(where_clauses)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Totales In / Out
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_flow,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                AVG(dwell_time) as avg_dwell
            FROM traffic_events
            WHERE {where_sql}
        """, params)
        row = cursor.fetchone()
        
        total_flow = row["total_flow"] or 0
        total_in = row["total_in"] or 0
        total_out = row["total_out"] or 0
        avg_dwell = round(row["avg_dwell"] or 0.0, 1)
        
        # Hora pico (hora con mayor aforo)
        cursor.execute(f"""
            SELECT strftime('%H:00', timestamp) as hour_slot, COUNT(*) as count
            FROM traffic_events
            WHERE {where_sql}
            GROUP BY hour_slot
            ORDER BY count DESC
            LIMIT 1
        """, params)
        peak_row = cursor.fetchone()
        peak_hour = f"{peak_row['hour_slot']} ({peak_row['count']} veh.)" if peak_row else "N/A"
        
        # Desglose por vehículo
        cursor.execute(f"""
            SELECT vehicle_type, COUNT(*) as count
            FROM traffic_events
            WHERE {where_sql}
            GROUP BY vehicle_type
            ORDER BY count DESC
        """, params)
        classes = {r["vehicle_type"]: r["count"] for r in cursor.fetchall()}
        
        return {
            "date": date_str,
            "total_flow": total_flow,
            "total_in": total_in,
            "total_out": total_out,
            "avg_dwell_seconds": avg_dwell,
            "peak_hour": peak_hour,
            "vehicle_classes": classes,
        }

def get_hourly_metrics(camera_id: Optional[str] = None, date_str: Optional[str] = None) -> list[dict[str, Any]]:
    """Obtiene el aforo desglosado hora por hora (00:00 a 23:00) para un día."""
    if date_str is None:
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        
    where_clauses = ["date(timestamp) = ?"]
    params: list[Any] = [date_str]
    
    if camera_id and camera_id != "all":
        where_clauses.append("camera_id = ?")
        params.append(camera_id)
        
    where_sql = " AND ".join(where_clauses)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT 
                cast(strftime('%H', timestamp) as integer) as hour,
                COUNT(*) as total,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out
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
                hourly_data.append({
                    "hour": hour_label,
                    "total": rows[h]["total"],
                    "in": rows[h]["total_in"],
                    "out": rows[h]["total_out"]
                })
            else:
                hourly_data.append({
                    "hour": hour_label,
                    "total": 0,
                    "in": 0,
                    "out": 0
                })
        return hourly_data

def get_daily_metrics(camera_id: Optional[str] = None, days: int = 7) -> list[dict[str, Any]]:
    """Obtiene el volumen total por día para los últimos N días."""
    where_clauses = ["timestamp >= date('now', ?)"]
    params: list[Any] = [f"-{days} days"]
    
    if camera_id and camera_id != "all":
        where_clauses.append("camera_id = ?")
        params.append(camera_id)
        
    where_sql = " AND ".join(where_clauses)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT 
                date(timestamp) as day_date,
                COUNT(*) as total,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out
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
                "out": r["total_out"]
            }
            for r in cursor.fetchall()
        ]

def get_events(
    camera_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    vehicle_type: Optional[str] = None,
    direction: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> dict[str, Any]:
    """Consulta la lista de eventos filtrada y paginada."""
    where_clauses = []
    params: list[Any] = []
    
    if camera_id and camera_id != "all":
        where_clauses.append("camera_id = ?")
        params.append(camera_id)
        
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
        
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Conteo total para paginación
        cursor.execute(f"SELECT COUNT(*) as total FROM traffic_events {where_sql}", params)
        total_count = cursor.fetchone()["total"]
        
        # Registros
        cursor.execute(f"""
            SELECT id, timestamp, camera_id, track_id, vehicle_type, direction, confidence, dwell_time
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
                ("Automovil", 0.65),
                ("Camion", 0.15),
                ("Autobus", 0.08),
                ("Motocicleta", 0.10),
                ("Bicicleta", 0.02)
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
                        
                        # Selección ponderada de tipo de vehículo
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
                        dwell = round(random.uniform(2.5, 24.0), 1)
                        
                        logger_events.append((
                            event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            cam,
                            track_id,
                            v_type,
                            direction,
                            conf,
                            dwell
                        ))
                        
            cursor.executemany("""
                INSERT INTO traffic_events (timestamp, camera_id, track_id, vehicle_type, direction, confidence, dwell_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, logger_events)
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
    location_id: Optional[str] = None
) -> dict[str, Any]:
    """
    Genera el reporte ejecutivo completo para el cliente organizado por:
    - Campaña
    - Pantalla (ubicación)
    - Desglose por Videos / Spots
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
    
    # Calcular métricas consolidadas
    total_spots_today = sum(v.get("plays_today", 0) for v in videos)
    total_spots_all = sum(v.get("total_plays", 0) for v in videos)
    total_impressions = sum(v.get("total_impressions", 0) for v in videos)
    total_duration_sec = sum(v.get("total_plays", 0) * v.get("duration_seconds", 15) for v in videos)
    exposure_hours = round(total_duration_sec / 3600.0, 1)

    # Consumo eléctrico de la pantalla
    total_kwh = round(sum(v.get("kwh_consumed", 0.0) for v in videos), 2)
    avg_kwh_cost = 3.85
    if screens:
        avg_kwh_cost = sum(s.get("cost_per_kwh", 3.85) for s in screens) / len(screens)
    energy_cost_mxn = round(total_kwh * avg_kwh_cost, 2)
    
    # Calcular costo para cada video individual
    for v in videos:
        v["cost_mxn"] = round(float(v.get("kwh_consumed", 0.0)) * avg_kwh_cost, 2)

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

def add_client(client_id: str, name: str, contact_email: str = "", logo_url: str = "") -> str:
    """Registra o actualiza un cliente comercial."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO clients (id, name, contact_email, logo_url)
            VALUES (?, ?, ?, ?)
        """, (client_id, name, contact_email, logo_url))
        conn.commit()
        return client_id

def update_client_logo(client_id: str, logo_url: str) -> bool:
    """Actualiza la URL o ruta del logotipo del cliente."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE clients SET logo_url = ? WHERE id = ?", (logo_url, client_id))
        conn.commit()
        return cursor.rowcount > 0

def assign_client_location(client_id: str, location_id: str) -> bool:
    """Asocia una pantalla/ubicación a un cliente."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO client_locations (client_id, location_id)
            VALUES (?, ?)
        """, (client_id, location_id))
        conn.commit()
        return True

def get_client_screens(client_id: str) -> list[dict[str, Any]]:
    """Devuelve las ubicaciones y pantallas autorizadas para un cliente."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT l.* FROM locations l
            JOIN client_locations cl ON l.id = cl.location_id
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


