"""
solar_brightness_engine.py — Motor de cálculo astronómico-solar y regulación dinámica de brillo.

Calcula la posición del sol (elevación, azimut, horas de amanecer/cenit/atardecer)
conforme a las coordenadas geográficas (latitud, longitud) y el ángulo de orientación
física de la pantalla LED (azimut en grados, ej. 90° Este, 270° Oeste).

Determina el ángulo de incidencia solar frontal sobre la cara de los módulos LED y
produce la curva horaria de brillo recomendada para maximizar la legibilidad con sol directo
y minimizar el consumo eléctrico y la contaminación lumínica nocturna.
"""

import math
import datetime
from typing import Any, Optional

def _to_rad(deg: float) -> float:
    return deg * math.pi / 180.0

def _to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def calculate_solar_position(
    lat: float,
    lon: float,
    dt: Optional[datetime.datetime] = None,
    utc_offset_hours: float = -6.0
) -> dict[str, Any]:
    """
    Calcula la posición solar astronómica (altura angular y azimut) para una fecha/hora y ubicación.
    lat: Latitud en grados (-90 a 90, ej. 20.6736 para Guadalajara, México)
    lon: Longitud en grados (-180 a 180, ej. -103.3855)
    dt: Fecha y hora local (si es None, toma datetime.now())
    utc_offset_hours: Desfase horario respecto a UTC en horas (ej. -6 para México Central)
    """
    if dt is None:
        dt = datetime.datetime.now()

    # Día del año (1 a 365/366)
    day_of_year = dt.timetuple().tm_yday
    time_hours = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    # Ángulo fraccional del año en radianes (Spencer, 1971)
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (time_hours - 12.0) / 24.0)

    # Ecuación del Tiempo (EoT) en minutos
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )

    # Declinación solar (delta) en radianes
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.001480 * math.sin(3.0 * gamma)
    )

    # Corrección por longitud respecto al meridiano estándar
    time_offset = eqtime + 4.0 * lon - 60.0 * utc_offset_hours
    true_solar_time = (time_hours * 60.0 + time_offset) % 1440.0

    # Ángulo horario solar (H) en grados y radianes (0° al mediodía solar)
    hour_angle_deg = (true_solar_time / 4.0) - 180.0
    hour_angle = _to_rad(hour_angle_deg)

    lat_rad = _to_rad(lat)

    # Coseno del ángulo cenital solar
    cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith_rad = math.acos(cos_zenith)
    elevation_rad = (math.pi / 2.0) - zenith_rad
    elevation_deg = _to_deg(elevation_rad)

    # NOAA Azimut solar en grados desde el Norte en sentido horario (0° Norte, 90° Este, 180° Sur, 270° Oeste)
    y_az = -math.cos(decl) * math.sin(hour_angle)
    x_az = math.sin(decl) * math.cos(lat_rad) - math.cos(decl) * math.sin(lat_rad) * math.cos(hour_angle)
    solar_azimuth_deg = (math.degrees(math.atan2(y_az, x_az)) + 360.0) % 360.0

    return {
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "elevation_deg": round(elevation_deg, 2),
        "azimuth_deg": round(solar_azimuth_deg, 1),
        "is_daylight": elevation_deg > 0.0,
        "is_civil_twilight": -6.0 <= elevation_deg <= 0.0,
        "is_night": elevation_deg < -6.0,
        "true_solar_time_str": f"{int(true_solar_time // 60):02d}:{int(true_solar_time % 60):02d}"
    }


def calculate_sun_ephemeris(
    lat: float,
    lon: float,
    date_obj: Optional[datetime.date] = None,
    utc_offset_hours: float = -6.0
) -> dict[str, str]:
    """
    Calcula horas clave del día: Amanecer (sunrise), Mediodía solar (cenit) y Atardecer (sunset).
    """
    if date_obj is None:
        date_obj = datetime.date.today()

    day_of_year = date_obj.timetuple().tm_yday
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1)
    
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma) - 0.040849 * math.sin(2.0 * gamma)
    )
    
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma) + 0.000907 * math.sin(2.0 * gamma)
    )

    lat_rad = _to_rad(lat)
    zenith_sunrise = _to_rad(90.833) # Corrección por refracción atmosférica y semidiámetro solar

    # Ángulo horario al amanecer/atardecer
    cos_ha = (math.cos(zenith_sunrise) - math.sin(lat_rad) * math.sin(decl)) / (math.cos(lat_rad) * math.cos(decl))
    
    if cos_ha > 1.0:
        return {"sunrise": "Noche Polar", "solar_noon": "12:00", "sunset": "Noche Polar"}
    elif cos_ha < -1.0:
        return {"sunrise": "Sol de Medianoche", "solar_noon": "12:00", "sunset": "Sol de Medianoche"}

    ha_deg = _to_deg(math.acos(cos_ha))
    
    solar_noon_min = (720.0 - 4.0 * lon - eqtime + 60.0 * utc_offset_hours) % 1440.0
    sunrise_min = (solar_noon_min - ha_deg * 4.0) % 1440.0
    sunset_min = (solar_noon_min + ha_deg * 4.0) % 1440.0

    def _fmt(m: float) -> str:
        h = int(m // 60)
        mi = int(m % 60)
        return f"{h:02d}:{mi:02d}"

    return {
        "sunrise": _fmt(sunrise_min),
        "solar_noon": _fmt(solar_noon_min),
        "sunset": _fmt(sunset_min)
    }


def compute_target_brightness(
    solar_elevation_deg: float,
    solar_azimuth_deg: float,
    screen_orientation_deg: float,
    min_night_brightness: int = 25,
    max_day_brightness: int = 95
) -> dict[str, Any]:
    """
    Determina el nivel óptimo de brillo (10% - 100%) para la pantalla LED en función de:
    - Altura angular del sol (solar_elevation_deg)
    - Azimut del sol (solar_azimuth_deg)
    - Orientación angular de la cara de la pantalla (screen_orientation_deg: 0° N, 90° E, 180° S, 270° O)
    - Límites configurados de brillo nocturno y diurno
    """
    min_night = max(10, min(50, min_night_brightness))
    max_day = max(min_night + 10, min(100, max_day_brightness))

    # 1. NOCHE ASTRONÓMICA (Elevación < -6°)
    if solar_elevation_deg < -6.0:
        return {
            "recommended_brightness": min_night,
            "condition": "Noche (Mínimo consumo & seguridad vial)",
            "incidence_factor": 0.0,
            "is_direct_sun": False
        }

    # 2. CREPÚSCULO CIVIL / AMANECER / ATARDECER (-6° <= Elevación <= 0°)
    if -6.0 <= solar_elevation_deg <= 0.0:
        progress = (solar_elevation_deg - (-6.0)) / 6.0
        # Transición suave
        smooth_ramp = min_night + progress * (45.0 - min_night)
        return {
            "recommended_brightness": int(round(smooth_ramp)),
            "condition": "Crepúsculo (Transición día/noche)",
            "incidence_factor": 0.0,
            "is_direct_sun": False
        }

    # 3. DÍA (Elevación > 0°)
    # Cálculo de incidencia sobre superficie vertical orientada a screen_orientation_deg
    # Ángulo horizontal relativo entre el sol y el frente de la pantalla
    diff_azimuth = abs(solar_azimuth_deg - screen_orientation_deg) % 360.0
    if diff_azimuth > 180.0:
        diff_azimuth = 360.0 - diff_azimuth

    elev_rad = _to_rad(solar_elevation_deg)
    diff_az_rad = _to_rad(diff_azimuth)

    # Coseno del ángulo de incidencia frontal:
    # Si diff_azimuth < 90°, el sol está delante de la pantalla.
    # Si diff_azimuth >= 90°, el sol está detrás (pantalla en sombra propia).
    cos_theta = math.cos(elev_rad) * math.cos(diff_az_rad)

    # Brillo diurno ambiental base (depende de qué tan alto está el sol, cielo iluminado)
    # Rango base: 50% hasta 70%
    base_ambient = 48.0 + 22.0 * math.pow(math.sin(elev_rad), 0.6)

    if cos_theta > 0.05 and solar_elevation_deg > 2.0:
        # El sol incide directamente en la cara de los módulos LED.
        # Requiere sobre-brillo para contrarrestar el reflejo y saturación del sol directo.
        incidence_factor = round(cos_theta, 3)
        boost = (max_day - base_ambient) * math.pow(cos_theta, 0.8) * math.sin(elev_rad)
        recommended = min(max_day, base_ambient + boost)
        condition = "Sol Frontal Directo (Máximo contraste publicitario)"
        is_direct = True
    else:
        # La pantalla está en sombra propia o sol lateral rasante; iluminación difusa.
        # Se ahorra energía sin perder visibilidad.
        incidence_factor = 0.0
        recommended = base_ambient
        condition = "Luz Diurna Indirecta / Sombra (Ahorro energético)"
        is_direct = False

    return {
        "recommended_brightness": int(round(recommended)),
        "condition": condition,
        "incidence_factor": incidence_factor,
        "is_direct_sun": is_direct
    }


def generate_24h_solar_schedule(
    lat: float,
    lon: float,
    screen_orientation_deg: float,
    min_night_brightness: int = 25,
    max_day_brightness: int = 95,
    target_date: Optional[datetime.date] = None,
    utc_offset_hours: float = -6.0
) -> dict[str, Any]:
    """
    Genera el calendario completo de 24 horas (intervalos de 30 minutos)
    con las proyecciones de posición solar y nivel de brillo recomendado.
    """
    if target_date is None:
        target_date = datetime.date.today()

    ephemeris = calculate_sun_ephemeris(lat, lon, target_date, utc_offset_hours)
    
    schedule_points = []
    # Generar muestras cada 30 minutos (48 puntos)
    for half_hour in range(48):
        h = half_hour // 2
        m = (half_hour % 2) * 30
        point_dt = datetime.datetime(target_date.year, target_date.month, target_date.day, h, m, 0)
        
        pos = calculate_solar_position(lat, lon, point_dt, utc_offset_hours)
        bright_calc = compute_target_brightness(
            pos["elevation_deg"],
            pos["azimuth_deg"],
            screen_orientation_deg,
            min_night_brightness,
            max_day_brightness
        )
        
        time_label = f"{h:02d}:{m:02d}"
        schedule_points.append({
            "time": time_label,
            "elevation_deg": pos["elevation_deg"],
            "azimuth_deg": pos["azimuth_deg"],
            "brightness": bright_calc["recommended_brightness"],
            "condition": bright_calc["condition"],
            "is_direct_sun": bright_calc["is_direct_sun"]
        })

    # Resumen de horas de pico de brillo
    peak_points = sorted(schedule_points, key=lambda x: x["brightness"], reverse=True)
    max_b = peak_points[0]["brightness"] if peak_points else max_day_brightness
    peak_times = [p["time"] for p in peak_points if p["brightness"] >= max_b - 3]

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "coordinates": {"lat": lat, "lon": lon},
        "screen_orientation_deg": screen_orientation_deg,
        "screen_orientation_cardinal": _get_cardinal_label(screen_orientation_deg),
        "ephemeris": ephemeris,
        "peak_brightness": max_b,
        "peak_window": f"{peak_times[0]} - {peak_times[-1]}" if len(peak_times) > 1 else (peak_times[0] if peak_times else "N/A"),
        "points": schedule_points
    }


def _get_cardinal_label(degrees: float) -> str:
    """Convierte grados de azimut a etiqueta cardinal amigable."""
    deg = (degrees % 360.0)
    cardinals = [
        ("Norte (0°)", 0.0),
        ("Noreste (45°)", 45.0),
        ("Este / Oriente (90°)", 90.0),
        ("Sureste (135°)", 135.0),
        ("Sur (180°)", 180.0),
        ("Suroeste (225°)", 225.0),
        ("Oeste / Poniente (270°)", 270.0),
        ("Noroeste (315°)", 315.0),
        ("Norte (360°)", 360.0),
    ]
    closest = min(cardinals, key=lambda c: abs(c[1] - deg))
    return closest[0]
