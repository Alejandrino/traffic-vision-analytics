"""
config.py - Configuracion central del sistema de analitica de trafico vehicular.
"""

from typing import Union

# SECCION 1: FUENTES DE VIDEO (CAMARAS)
CAMERA_SOURCES: dict[str, Union[str, int]] = {
    "cam1": 0,   # Camara USB 0 (1280x720)
    "cam2": 1,   # Camara USB 1 (640x480) - nueva camara conectada
}

# SECCION 2: LINEAS DE CONTEO (Soporte multi-línea vehicular y peatonal)
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

def normalize_counting_lines(cam_id: str, raw_data: Union[dict, list, tuple, None]) -> list[dict]:
    """Convierte cualquier formato de líneas (legado de 1 par o multi-línea) a formato estandarizado."""
    if not raw_data:
        return [{
            "id": f"{cam_id}_line_1",
            "name": "Línea Principal",
            "target_entity": "ALL",
            "coords": ((50, FRAME_HEIGHT // 2), (FRAME_WIDTH - 50, FRAME_HEIGHT // 2))
        }]
    
    # Caso 1: Formato legado tuple/list de 2 puntos: [[x1, y1], [x2, y2]] o ((x1, y1), (x2, y2))
    if isinstance(raw_data, (list, tuple)) and len(raw_data) == 2 and isinstance(raw_data[0], (list, tuple)) and len(raw_data[0]) == 2 and isinstance(raw_data[0][0], (int, float)):
        return [{
            "id": f"{cam_id}_line_1",
            "name": "Línea Principal",
            "target_entity": "ALL",
            "coords": ((int(raw_data[0][0]), int(raw_data[0][1])), (int(raw_data[1][0]), int(raw_data[1][1])))
        }]

    # Caso 2: Lista de diccionarios de líneas
    if isinstance(raw_data, list):
        normalized = []
        for idx, item in enumerate(raw_data):
            if isinstance(item, dict):
                lid = item.get("id") or f"{cam_id}_line_{idx+1}"
                lname = item.get("name") or f"Línea {idx+1}"
                target_ent = str(item.get("target_entity", "ALL")).upper()
                if target_ent not in ("ALL", "VEHICLE", "PEDESTRIAN"):
                    target_ent = "ALL"
                coords_raw = item.get("coords") or item.get("line") or [[50, FRAME_HEIGHT // 2], [FRAME_WIDTH - 50, FRAME_HEIGHT // 2]]
                c1 = (int(coords_raw[0][0]), int(coords_raw[0][1]))
                c2 = (int(coords_raw[1][0]), int(coords_raw[1][1]))
                normalized.append({
                    "id": lid,
                    "name": lname,
                    "target_entity": target_ent,
                    "coords": (c1, c2)
                })
        if normalized:
            return normalized

    return [{
        "id": f"{cam_id}_line_1",
        "name": "Línea Principal",
        "target_entity": "ALL",
        "coords": ((50, FRAME_HEIGHT // 2), (FRAME_WIDTH - 50, FRAME_HEIGHT // 2))
    }]

# Configuración por defecto de líneas múltiples
DEFAULT_MULTI_LINES: dict[str, list[dict]] = {
    "cam1": [
        {
            "id": "cam1_line_1",
            "name": "Carril Vehicular Principal",
            "target_entity": "VEHICLE",
            "coords": ((50, 360), (1230, 360)),
        },
        {
            "id": "cam1_line_2",
            "name": "Paso Peatonal / Banqueta",
            "target_entity": "PEDESTRIAN",
            "coords": ((50, 520), (600, 520)),
        }
    ],
    "cam2": [
        {
            "id": "cam2_line_1",
            "name": "Línea Principal",
            "target_entity": "ALL",
            "coords": ((20, 240), (620, 240)),
        }
    ],
}

COUNTING_LINES: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "cam1": ((50, 360), (1230, 360)),
    "cam2": ((20, 240), (620, 240)),
}
MULTI_COUNTING_LINES: dict[str, list[dict]] = dict(DEFAULT_MULTI_LINES)

# SECCION 3: MODELO DE INFERENCIA
YOLO_MODEL_PATH: str = "yolov8n_openvino_model"
INFERENCE_DEVICE: str = "intel:NPU"
CONFIDENCE_THRESHOLD: float = 0.25
IOU_THRESHOLD: float = 0.45

PEDESTRIAN_CLASS_IDS: list[int] = [0]
VEHICLE_CLASS_IDS: list[int] = [1, 2, 3, 5, 7]
TARGET_CLASS_IDS: list[int] = [0, 1, 2, 3, 5, 7]

CLASS_LABELS: dict[int, str] = {
    0: "Peaton",
    1: "Bicicleta",
    2: "Automovil",
    3: "Motocicleta",
    5: "Autobus",
    7: "Camion",
}

# Mapeo de clase a categoría de entidad ("PEDESTRIAN" vs "VEHICLE")
ENTITY_TYPE_MAP: dict[str, str] = {
    "Peaton": "PEDESTRIAN",
    "Bicicleta": "VEHICLE",
    "Automovil": "VEHICLE",
    "Motocicleta": "VEHICLE",
    "Autobus": "VEHICLE",
    "Camion": "VEHICLE",
}

# SECCION 4: TRACKING
TRACK_LOST_THRESHOLD: int = 30
TRACK_MINIMUM_CONSECUTIVE_FRAMES: int = 1

# SECCION 5: DWELL TIME
DWELL_LEAD_SECONDS: float = 10.0
ASSUMED_FPS: float = 25.0

# SECCION 6: SERVIDOR
SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 8000
WEBSOCKET_BROADCAST_INTERVAL: float = 1.0

# SECCION 7: RTSP
RTSP_RECONNECT_DELAY: float = 5.0
RTSP_MAX_RETRIES: int = -1
RTSP_FRAME_TIMEOUT: float = 10.0
RTSP_TRANSPORT: str = "tcp"