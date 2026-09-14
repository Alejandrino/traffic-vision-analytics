"""
novastar_controller.py — Controlador para reproductores multimedia Novastar Taurus TB40.

Gestiona el encendido/standby de salida de video, ajuste de brillo y estado
de conexión con procesadores de pantallas LED Novastar Taurus TB40 vía API HTTP / JSON-RPC.
Incluye modo de simulación cuando el hardware físico no está disponible en la red local.
"""

import logging
import json
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
