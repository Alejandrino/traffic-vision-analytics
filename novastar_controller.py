"""
novastar_controller.py — Controlador para reproductores multimedia Novastar Taurus TB40.

Gestiona el encendido/standby de salida de video, ajuste de brillo y estado
de conexión con procesadores de pantallas LED Novastar Taurus TB40 vía API HTTP / JSON-RPC.
Incluye modo de simulación cuando el hardware físico no está disponible en la red local.
"""

import logging
import json
import time
from typing import Any, Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Estado simulado en memoria cuando el equipo físico no responde
_simulated_state: dict[str, dict[str, Any]] = {}

def _get_or_create_sim(ip: str) -> dict[str, Any]:
    if ip not in _simulated_state:
        _simulated_state[ip] = {
            "connected": True,
            "screen_power": True,
            "brightness": 75,
            "model": "NovaStar Taurus TB40",
            "firmware": "V3.8.4",
            "current_play": "Playlist_Publicidad_DOOH_01",
            "temperature_c": 38.5,
            "is_simulated": True
        }
    return _simulated_state[ip]


def get_status(ip: str, port: int = 8001, timeout: float = 2.0) -> dict[str, Any]:
    """
    Obtiene el estado de conexión, pantalla y brillo del reproductor Novastar TB40.
    """
    url = f"http://{ip}:{port}/api/v1.0/screen/status"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TrafficVision/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return {
                    "connected": True,
                    "screen_power": data.get("power", True),
                    "brightness": data.get("brightness", 75),
                    "model": data.get("model", "NovaStar TB40"),
                    "firmware": data.get("firmware", "V3.8.4"),
                    "is_simulated": False
                }
    except Exception as e:
        logger.debug(f"Novastar TB40 en {ip}:{port} no respondió vía HTTP ({e}). Usando modo emulado.")

    # Retorno en modo emulado / resiliente
    sim = _get_or_create_sim(ip)
    return {
        "connected": True,
        "screen_power": sim["screen_power"],
        "brightness": sim["brightness"],
        "model": sim["model"],
        "firmware": sim["firmware"],
        "current_play": sim["current_play"],
        "temperature_c": sim["temperature_c"],
        "is_simulated": True
    }


def set_screen_power(ip: str, power_on: bool, port: int = 8001, timeout: float = 2.0) -> dict[str, Any]:
    """
    Activa o desactiva la salida de video del Novastar TB40 (Pantalla Negra / Standby).
    """
    url = f"http://{ip}:{port}/api/v1.0/screen/power"
    payload = json.dumps({"power": power_on}).encode("utf-8")
    
    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status in (200, 204):
                return {"status": "ok", "screen_power": power_on, "is_simulated": False}
    except Exception as e:
        logger.debug(f"Novastar TB40 {ip} error set_power ({e}). Aplicando a estado emulado.")

    sim = _get_or_create_sim(ip)
    sim["screen_power"] = power_on
    return {"status": "ok", "screen_power": power_on, "is_simulated": True}


def set_brightness(ip: str, brightness: int, port: int = 8001, timeout: float = 2.0) -> dict[str, Any]:
    """
    Ajusta el brillo de la pantalla LED entre 0% y 100%.
    """
    brightness = max(0, min(100, brightness))
    url = f"http://{ip}:{port}/api/v1.0/screen/brightness"
    payload = json.dumps({"brightness": brightness}).encode("utf-8")
    
    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status in (200, 204):
                return {"status": "ok", "brightness": brightness, "is_simulated": False}
    except Exception as e:
        logger.debug(f"Novastar TB40 {ip} error set_brightness ({e}). Aplicando a estado emulado.")

    sim = _get_or_create_sim(ip)
    sim["brightness"] = brightness
    return {"status": "ok", "brightness": brightness, "is_simulated": True}


# Catálogo de spots de video en rotación en los reproductores NovaStar TB40
_ROTATION_PLAYLIST = [
    {
        "video_name": "minerva_ipa_artesanal_15s.mp4",
        "title": "Minerva IPA — Especial Verano",
        "client_id": "cli_cerveceria",
        "client_name": "Cervecería Cuauhtémoc Moctezuma",
        "campaign_name": "Campaña Cerveza Minerva — Edición Artesanal Verano",
        "duration_seconds": 15,
        "resolution": "1920x1080",
        "fps": 60,
        "bitrate": "14.2 Mbps",
        "aspect_ratio": "16:9",
        "location_id": "loc_minerva",
        "plays_today": 194
    },
    {
        "video_name": "nissan_kicks_epower_aceleracion_20s.mp4",
        "title": "Nissan Kicks e-POWER — Aceleración 100% Eléctrica",
        "client_id": "cli_nissan",
        "client_name": "Nissan México",
        "campaign_name": "Lanzamiento Nacional Nuevo Nissan Kicks e-POWER",
        "duration_seconds": 20,
        "resolution": "1920x1080",
        "fps": 60,
        "bitrate": "15.0 Mbps",
        "aspect_ratio": "16:9",
        "location_id": "loc_minerva",
        "plays_today": 180
    },
    {
        "video_name": "liverpool_venta_nocturna_ofertas_15s.mp4",
        "title": "Gran Venta Nocturna — Exclusivo Tarjetas",
        "client_id": "cli_retail",
        "client_name": "Liverpool México",
        "campaign_name": "Gran Venta Nocturna Aniversario Liverpool",
        "duration_seconds": 15,
        "resolution": "1920x1080",
        "fps": 60,
        "bitrate": "12.8 Mbps",
        "aspect_ratio": "16:9",
        "location_id": "loc_minerva",
        "plays_today": 210
    },
    {
        "video_name": "bohemia_cristal_refrescante_15s.mp4",
        "title": "Bohemia Cristal — Refrescante por Naturaleza",
        "client_id": "cli_cerveceria",
        "client_name": "Cervecería Cuauhtémoc Moctezuma",
        "campaign_name": "Campaña Bohemia Cristal — Maridaje DOOH",
        "duration_seconds": 15,
        "resolution": "1920x1080",
        "fps": 60,
        "bitrate": "13.5 Mbps",
        "aspect_ratio": "16:9",
        "location_id": "loc_periferico",
        "plays_today": 168
    },
    {
        "video_name": "banco_azteca_credito_nomina_15s.mp4",
        "title": "Crédito Nómina Inmediato en tu App",
        "client_id": "cli_banco",
        "client_name": "Banco Azteca",
        "campaign_name": "Crédito Nómina y Cuenta Digital Banco Azteca",
        "duration_seconds": 15,
        "resolution": "1920x1080",
        "fps": 60,
        "bitrate": "11.9 Mbps",
        "aspect_ratio": "16:9",
        "location_id": "loc_periferico",
        "plays_today": 175
    }
]


def get_current_playback(
    ip: str = "192.168.1.140",
    port: int = 8001,
    location_id: Optional[str] = None,
    timeout: float = 2.0
) -> dict[str, Any]:
    """
    Obtiene los detalles del video o spot publicitario que se está reproduciendo
    actualmente en el reproductor NovaStar TB40 en tiempo real.
    """
    url = f"http://{ip}:{port}/api/v1.0/player/current"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TrafficVision/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                if "video_name" in data:
                    return {**data, "is_simulated": False}
    except Exception as e:
        logger.debug(f"Novastar TB40 {ip} no respondió /player/current ({e}). Usando rotación activa sincronizada.")

    # Filtrar catálogo según la ubicación si se especifica
    pool = [v for v in _ROTATION_PLAYLIST if (location_id is None or location_id == "all" or v.get("location_id") == location_id)]
    if not pool:
        pool = _ROTATION_PLAYLIST

    total_cycle = sum(v["duration_seconds"] for v in pool)
    now_sec = int(time.time()) % max(1, total_cycle)

    accum = 0
    current_idx = 0
    elapsed_in_spot = 0
    for idx, spot in enumerate(pool):
        dur = spot["duration_seconds"]
        if accum <= now_sec < accum + dur:
            current_idx = idx
            elapsed_in_spot = now_sec - accum
            break
        accum += dur

    current_spot = pool[current_idx]
    next_spot = pool[(current_idx + 1) % len(pool)]
    duration = current_spot["duration_seconds"]
    remaining = max(0, duration - elapsed_in_spot)
    progress = round((elapsed_in_spot / max(1, duration)) * 100, 1)

    return {
        "status": "playing",
        "video_name": current_spot["video_name"],
        "title": current_spot["title"],
        "client_id": current_spot["client_id"],
        "client_name": current_spot["client_name"],
        "campaign_name": current_spot["campaign_name"],
        "duration_seconds": duration,
        "elapsed_seconds": elapsed_in_spot,
        "remaining_seconds": remaining,
        "progress_percent": progress,
        "resolution": current_spot["resolution"],
        "fps": current_spot["fps"],
        "bitrate": current_spot["bitrate"],
        "aspect_ratio": current_spot["aspect_ratio"],
        "playlist_name": f"DOOH_Rotacion_{location_id or 'Minerva'}",
        "location_id": current_spot.get("location_id", location_id or "loc_minerva"),
        "plays_today": current_spot.get("plays_today", 180),
        "next_video": {
            "title": next_spot["title"],
            "video_name": next_spot["video_name"],
            "duration_seconds": next_spot["duration_seconds"],
            "client_name": next_spot["client_name"]
        },
        "is_simulated": True
    }

