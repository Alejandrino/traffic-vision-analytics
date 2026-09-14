import cv2
import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def scan_usb_cameras(max_cameras: int = 5) -> List[Dict[str, Any]]:
    """
    Escanea puertos locales buscando cámaras USB o integradas disponibles.
    Devuelve una lista de diccionarios con info de cada cámara encontrada.
    """
    available_cameras = []
    
    for i in range(max_cameras):
        # En Windows a veces DSHOW es más rápido, pero para escanear probamos normal
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            continue
            
        ret, frame = cap.read()
        if ret and frame is not None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            available_cameras.append({
                "id": i,
                "name": f"USB Camera {i}",
                "resolution": f"{width}x{height}",
                "fps": fps,
                "source": i
            })
            logger.info(f"[Scanner] Encontrada cámara {i} ({width}x{height} @ {fps}fps)")
            
        cap.release()
        
    return available_cameras

def validate_rtsp_stream(url: str) -> bool:
    """Verifica rápidamente si una URL RTSP o de video es válida y accesible."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret
