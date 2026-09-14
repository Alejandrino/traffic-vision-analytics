"""
shelly_controller.py — Controlador para relés y medidores de potencia Shelly Pro.

Gestiona el encendido/apagado físico del contactor de las pantallas LED y la telemetría
de consumo eléctrico (Potencia en Watts, Voltaje, Corriente y Energía acumulada en kWh)
utilizando el protocolo RPC local de Shelly (Gen2 / Gen3).
Incluye modo de simulación cuando el dispositivo físico no se encuentra en la red local.
"""

import logging
import json
import random
import time
from typing import Any, Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Estado simulado en memoria
_shelly_sim: dict[str, dict[str, Any]] = {}

def _get_or_create_sim(ip: str, channel: int = 0) -> dict[str, Any]:
    key = f"{ip}:{channel}"
    if key not in _shelly_sim:
        _shelly_sim[key] = {
            "is_on": True,
            "voltage": 220.5,
            "current": 20.4,
            "power_w": 4500.0,
            "total_kwh": 342.8,
            "last_tick": time.time(),
            "device_name": "Shelly Pro 1PM (Pantalla LED)",
            "mac": "C8:2E:18:4A:BC:90",
            "is_simulated": True
        }
    
    # Simular acumulación de energía con el paso del tiempo
    sim = _shelly_sim[key]
    now = time.time()
    dt_hours = (now - sim["last_tick"]) / 3600.0
    sim["last_tick"] = now

    if sim["is_on"]:
        # Fluctuación de consumo de pantalla LED con brillo activo (3500W a 6500W)
        sim["power_w"] = round(random.uniform(4200.0, 4900.0), 1)
        sim["voltage"] = round(random.uniform(218.0, 224.0), 1)
        sim["current"] = round(sim["power_w"] / max(sim["voltage"], 1.0), 2)
        sim["total_kwh"] = round(sim["total_kwh"] + ((sim["power_w"] / 1000.0) * dt_hours), 4)
    else:
        # En standby / apagada consume solo la electrónica mínima
        sim["power_w"] = round(random.uniform(15.0, 25.0), 1)
        sim["current"] = 0.12
        sim["total_kwh"] = round(sim["total_kwh"] + ((sim["power_w"] / 1000.0) * dt_hours), 4)

    return sim


def get_status(ip: str, channel: int = 0, timeout: float = 2.0) -> dict[str, Any]:
    """
    Lee el estado del interruptor y las métricas eléctricas del Shelly Pro.
    """
    url = f"http://{ip}/rpc/Switch.GetStatus?id={channel}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TrafficVision/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                aenergy = data.get("aenergy", {})
                return {
                    "is_on": data.get("output", False),
                    "power_w": round(data.get("apower", 0.0), 1),
                    "voltage": round(data.get("voltage", 220.0), 1),
                    "current": round(data.get("current", 0.0), 2),
                    "total_kwh": round(aenergy.get("total", 0.0) / 1000.0, 3),
                    "temperature_c": data.get("temperature", {}).get("tC", 36.0),
                    "is_simulated": False
                }
    except Exception as e:
        logger.debug(f"Shelly Pro en {ip} no respondió vía RPC ({e}). Usando modo emulado.")

    sim = _get_or_create_sim(ip, channel)
    return {
        "is_on": sim["is_on"],
        "power_w": sim["power_w"],
        "voltage": sim["voltage"],
        "current": sim["current"],
        "total_kwh": round(sim["total_kwh"], 3),
        "temperature_c": 38.0,
        "is_simulated": True
    }


def set_power(ip: str, turn_on: bool, channel: int = 0, timeout: float = 2.5) -> dict[str, Any]:
    """
    Enciende o apaga el relé del Shelly Pro (contactor principal de la pantalla).
    """
    on_str = "true" if turn_on else "false"
    url = f"http://{ip}/rpc/Switch.Set?id={channel}&on={on_str}"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TrafficVision/2.0"}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return {"status": "ok", "is_on": turn_on, "is_simulated": False}
    except Exception as e:
        logger.debug(f"Shelly Pro {ip} error set_power ({e}). Aplicando a estado emulado.")

    sim = _get_or_create_sim(ip, channel)
    sim["is_on"] = turn_on
    if not turn_on:
        sim["power_w"] = 18.0
        sim["current"] = 0.1
    else:
        sim["power_w"] = 4500.0
    return {"status": "ok", "is_on": turn_on, "is_simulated": True}
