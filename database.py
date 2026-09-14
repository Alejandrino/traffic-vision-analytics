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

def get_campaigns(location_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Obtiene las campañas publicitarias activas o por ubicación."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if location_id and location_id != "all":
            cursor.execute("""
                SELECT c.*, l.name as location_name, l.cost_per_kwh
                FROM campaigns c
                JOIN locations l ON c.location_id = l.id
                WHERE c.location_id = ?
                ORDER BY c.id DESC
            """, (location_id,))
        else:
            cursor.execute("""
                SELECT c.*, l.name as location_name, l.cost_per_kwh
                FROM campaigns c
                JOIN locations l ON c.location_id = l.id
                ORDER BY c.id DESC
            """)
        return [dict(r) for r in cursor.fetchall()]

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
    status: str = "Activa"
) -> int:
    """Crea una nueva campaña publicitaria."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO campaigns (name, client, location_id, start_date, end_date, start_hour, end_hour, spot_seconds, spots_per_hour, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, client, location_id, start_date, end_date, start_hour, end_hour, spot_seconds, spots_per_hour, status))
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

