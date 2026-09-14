"""
novastar_vx600.py — Controlador para Procesador de Video Todo-en-Uno NovaStar VX600 Pro.

Gestiona controladores de pantalla NovaStar VX600 Pro (6 puertos Ethernet, 3 capas independientes,
entradas HDMI/DVI/SDI). Soporta cambio de presets de hardware para pantallas de doble cara
(Cara A / Cara B) conmutando entre:
1. Preset 1 (Espejo Unificado): 1x TB40 alimenta ambas pantallas simultáneamente.
2. Preset 2 (Contenido Dual Independiente): 2x TB40 independientes (uno por cada cara).
3. Preset 3 (PIP / Multiventana): Composición multicapa para pautas simultáneas.
"""

import logging
import json
from typing import Any, Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Catálogo oficial de Presets para el Procesador NovaStar VX600 Pro
PRESETS_CATALOG = {
    "preset_mirror_1tb40": {
        "id": "preset_mirror_1tb40",
        "name": "Preset 1 — 1x TB40 Espejo / Unificado (Ambas Caras)",
        "description": "1 reproductor TB40 en HDMI-1. El procesador clona la señal hacia Cara A (Puertos 1-3) y Cara B (Puertos 4-6).",
        "input_primary": "HDMI-1 (TB40 Maestro)",
        "input_secondary": "Ninguna (Clonado)",
        "mode": "Mirror / Split Unificado",
        "active_layers": 1,
        "badge_color": "emerald"
    },
    "preset_dual_2tb40": {
        "id": "preset_dual_2tb40",
        "name": "Preset 2 — 2x TB40 Contenido Independiente (Cara A + Cara B)",
        "description": "2 reproductores TB40 independientes. HDMI-1 asignado a Cara A y HDMI-2/DVI asignado a Cara B para pautas distintas.",
        "input_primary": "HDMI-1 (TB40 Cara A)",
        "input_secondary": "HDMI-2 (TB40 Cara B)",
        "mode": "Dual Independent Split",
        "active_layers": 2,
        "badge_color": "blue"
    },
    "preset_pip_promo": {
        "id": "preset_pip_promo",
        "name": "Preset 3 — PIP / Multiventana Dinámica",
        "description": "Composición Picture-in-Picture con TB40 principal en fondo y contenido promocional/cámara en recuadro escalado.",
        "input_primary": "HDMI-1 (Capa Fondo)",
        "input_secondary": "HDMI-2 (Capa Flotante)",
        "mode": "Picture-in-Picture (PIP)",
        "active_layers": 2,
        "badge_color": "purple"
    }
}

# Estado simulado en memoria para resiliencia local o hardware desconectado
_simulated_vx600: dict[str, dict[str, Any]] = {}

def _get_or_create_sim(ip: str) -> dict[str, Any]:
    if ip not in _simulated_vx600:
        _simulated_vx600[ip] = {
            "connected": True,
            "power": True,
            "brightness": 85,
            "model": "NovaStar VX600 Pro",
            "firmware": "V1.4.2",
            "active_preset": "preset_mirror_1tb40",
            "temperature_c": 41.2,
            "total_output_ports": 6,
            "ports_in_use": 4,
            "genlock_locked": True,
            "input_signals": {
                "HDMI-1": "1920x1080@60Hz (Activo)",
                "HDMI-2": "1920x1080@60Hz (Activo)",
                "3G-SDI": "Sin señal",
                "DVI": "1920x1080@60Hz (En espera)"
            },
            "is_simulated": True
        }
    return _simulated_vx600[ip]


def get_presets() -> list[dict[str, Any]]:
    """Devuelve la lista de presets disponibles para el VX600 Pro."""
    return list(PRESETS_CATALOG.values())


def get_status(ip: str, port: int = 6000, timeout: float = 2.0) -> dict[str, Any]:
    """
    Obtiene el estado completo del procesador NovaStar VX600 Pro:
    conexión, preset activo, entradas de video, temperatura y brillo.
    """
    url = f"http://{ip}:{port}/api/v1/device/status"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ApexCompany-DOOH/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return {
                    "connected": True,
                    "power": data.get("power", True),
                    "brightness": data.get("brightness", 85),
                    "model": data.get("model", "NovaStar VX600 Pro"),
                    "firmware": data.get("firmware", "V1.4.2"),
                    "active_preset": data.get("active_preset", "preset_mirror_1tb40"),
                    "preset_info": PRESETS_CATALOG.get(data.get("active_preset", "preset_mirror_1tb40")),
                    "temperature_c": data.get("temperature_c", 40.0),
                    "is_simulated": False
                }
    except Exception as e:
        logger.debug(f"NovaStar VX600 Pro en {ip}:{port} no respondió vía HTTP ({e}). Usando modo emulado.")

    sim = _get_or_create_sim(ip)
    active_pid = sim["active_preset"]
    return {
        "connected": True,
        "power": sim["power"],
        "brightness": sim["brightness"],
        "model": sim["model"],
        "firmware": sim["firmware"],
        "active_preset": active_pid,
        "preset_info": PRESETS_CATALOG.get(active_pid, PRESETS_CATALOG["preset_mirror_1tb40"]),
        "temperature_c": sim["temperature_c"],
        "total_output_ports": sim["total_output_ports"],
        "ports_in_use": sim["ports_in_use"],
        "genlock_locked": sim["genlock_locked"],
        "input_signals": sim["input_signals"],
        "is_simulated": True
    }


def apply_preset(ip: str, preset_id: str, port: int = 6000, timeout: float = 2.0) -> dict[str, Any]:
    """
    Aplica un preset de conmutación de entradas y capas en el NovaStar VX600 Pro.
    Permite alternar entre contenido unificado (1x TB40) o independiente (2x TB40).
    """
    if preset_id not in PRESETS_CATALOG:
        preset_id = "preset_mirror_1tb40"

    preset_data = PRESETS_CATALOG[preset_id]
    url = f"http://{ip}:{port}/api/v1/preset/apply"
    payload = json.dumps({"preset_id": preset_id}).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status in (200, 204):
                logger.info(f"[VX600 Pro] Preset '{preset_id}' aplicado con éxito en {ip}:{port}")
                return {"status": "ok", "active_preset": preset_id, "preset_info": preset_data, "is_simulated": False}
    except Exception as e:
        logger.debug(f"VX600 Pro {ip} error aplicando preset ({e}). Aplicando en simulación.")

    sim = _get_or_create_sim(ip)
    sim["active_preset"] = preset_id
    logger.info(f"[VX600 Pro SIM] Preset cambiado a '{preset_id}' ({preset_data['name']}) para {ip}")
    return {"status": "ok", "active_preset": preset_id, "preset_info": preset_data, "is_simulated": True}


def set_brightness(ip: str, brightness: int, port: int = 6000, timeout: float = 2.0) -> dict[str, Any]:
    """Ajusta el brillo global en el procesador NovaStar VX600 Pro (0 - 100%)."""
    brightness = max(0, min(100, brightness))
    url = f"http://{ip}:{port}/api/v1/screen/brightness"
    payload = json.dumps({"brightness": brightness}).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status in (200, 204):
                return {"status": "ok", "brightness": brightness, "is_simulated": False}
    except Exception as e:
        pass

    sim = _get_or_create_sim(ip)
    sim["brightness"] = brightness
    return {"status": "ok", "brightness": brightness, "is_simulated": True}


def set_power(ip: str, power_on: bool, port: int = 6000, timeout: float = 2.0) -> dict[str, Any]:
    """Enciende o apaga la salida de video en el VX600 Pro (Blackout / Freeze)."""
    url = f"http://{ip}:{port}/api/v1/screen/power"
    payload = json.dumps({"power": power_on}).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status in (200, 204):
                return {"status": "ok", "power": power_on, "is_simulated": False}
    except Exception as e:
        pass

    sim = _get_or_create_sim(ip)
    sim["power"] = power_on
    return {"status": "ok", "power": power_on, "is_simulated": True}
