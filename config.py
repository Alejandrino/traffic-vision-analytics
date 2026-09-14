"""
config.py - Configuracion central del sistema de analitica de trafico vehicular.
"""

from typing import Union

# SECCION 1: FUENTES DE VIDEO (CAMARAS)
CAMERA_SOURCES: dict[str, Union[str, int]] = {
    "cam1": 0,   # Camara USB 0 (1280x720)
    "cam2": 1,   # Camara USB 1 (640x480) - nueva camara conectada
}

# SECCION 2: LINEAS DE CONTEO
COUNTING_LINES: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "cam1": ((50, 360), (1230, 360)),
    "cam2": ((20, 240), (620, 240)),
}

FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

# SECCION 3: MODELO DE INFERENCIA
YOLO_MODEL_PATH: str = "yolov8n_openvino_model"
INFERENCE_DEVICE: str = "intel:NPU"
CONFIDENCE_THRESHOLD: float = 0.40
IOU_THRESHOLD: float = 0.45

VEHICLE_CLASS_IDS: list[int] = [1, 2, 3, 5, 7]

CLASS_LABELS: dict[int, str] = {
    1: "Bicicleta",
    2: "Automovil",
    3: "Motocicleta",
    5: "Autobus",
    7: "Camion",
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