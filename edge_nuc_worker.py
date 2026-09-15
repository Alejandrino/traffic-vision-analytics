#!/usr/bin/env python3
"""
edge_nuc_worker.py — Agente Autónomo Edge para Intel NUC (Apex Company).

Este script se ejecuta exclusivamente en la computadora local / Intel NUC:
  1. Captura video de las cámaras locales (USB o RTSP).
  2. Ejecuta inferencia YOLOv8 utilizando OpenVINO en la NPU/GPU de la Intel NUC.
  3. Mide aforo vehicular, peatonal y líneas de conteo en tiempo real.
  4. Lee el estado y consumo de los relevadores Shelly locales.
  5. Envía periódicamente (cada 3 seg) la telemetría y un snapshot comprimido
     hacia la nube (Hostinger / Vercel) mediante HTTPS POST sin requerir puertos abiertos.
"""

import os
import sys
import time
import json
import base64
import logging
import urllib.request
import urllib.error
import cv2

import config
from camera_stream import CameraStream, create_all_streams
from traffic_analyzer import TrafficAnalyzer

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [NUC-EDGE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("edge_nuc_worker")

# ── Configuración Cloud & Identificación NUC ───────────────────────
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "http://localhost:8000")
NUC_ID = os.getenv("NUC_ID", "nuc_minerva_01")
PUSH_INTERVAL_SEC = float(os.getenv("PUSH_INTERVAL_SEC", "3.0"))
ENABLE_SNAPSHOTS = os.getenv("ENABLE_SNAPSHOTS", "true").lower() == "true"


def send_telemetry_to_cloud(payload: dict) -> bool:
    """Envía el paquete de telemetría a la API en la nube."""
    endpoint = f"{CLOUD_API_URL.rstrip('/')}/api/telemetry/push"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data_bytes,
        headers={"Content-Type": "application/json", "User-Agent": f"ApexNUC/{NUC_ID}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                return True
    except urllib.error.URLError as e:
        logger.warning(f"Error enviando telemetría a la nube ({endpoint}): {e.reason}")
    except Exception as e:
        logger.warning(f"Excepción inesperada al reportar a la nube: {e}")
    return False


def main():
    logger.info("==========================================================")
    logger.info(f"   Iniciando Agente Edge en Intel NUC (ID: {NUC_ID})")
    logger.info(f"   Destino Cloud: {CLOUD_API_URL}")
    logger.info(f"   Dispositivo Inferencia: {config.MODEL_DEVICE}")
    logger.info("==========================================================")

    # 1. Iniciar cámaras locales
    streams = create_all_streams()
    for s in streams.values():
        s.start()

    # 2. Inicializar analizadores de tráfico YOLOv8 por cámara
    analyzers = {}
    for cam_id in config.CAMERA_SOURCES:
        try:
            analyzer = TrafficAnalyzer(
                cam_id=cam_id,
                model_path=config.YOLO_MODEL_PATH,
                device=config.MODEL_DEVICE,
                conf_threshold=config.CONF_THRESHOLD,
                iou_threshold=config.IOU_THRESHOLD,
            )
            analyzers[cam_id] = analyzer
            logger.info(f"[{cam_id}] YOLOv8 listo en dispositivo {config.MODEL_DEVICE}")
        except Exception as e:
            logger.error(f"[{cam_id}] Error cargando modelo: {e}")

    last_push_time = 0.0
    frame_counts = {cid: 0 for cid in config.CAMERA_SOURCES}

    try:
        while True:
            now = time.time()
            latest_metrics = {}
            primary_snapshot_b64 = None
            primary_cam_id = None

            # Procesar frames de cada cámara conectada
            for cam_id, stream in streams.items():
                if not stream.is_connected():
                    continue

                frame = stream.read()
                if frame is None:
                    continue

                frame_counts[cam_id] += 1
                analyzer = analyzers.get(cam_id)
                if analyzer:
                    # Inferencia de aforo vehicular y peatonal
                    annotated, metrics = analyzer.process_frame(frame)
                    latest_metrics[cam_id] = metrics.to_dict()

                    # Generar snapshot JPEG optimizado para la primera cámara activa
                    if ENABLE_SNAPSHOTS and primary_snapshot_b64 is None:
                        # Redimensionar ligeramente para mínimo consumo de ancho de banda
                        small = cv2.resize(annotated, (640, 360), interpolation=cv2.INTER_AREA)
                        _, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        primary_snapshot_b64 = base64.b64encode(buffer).decode("utf-8")
                        primary_cam_id = cam_id

            # Despachar telemetría a la nube a intervalos regulares
            if now - last_push_time >= PUSH_INTERVAL_SEC and latest_metrics:
                payload = {
                    "nuc_id": NUC_ID,
                    "timestamp": now,
                    "cameras": latest_metrics,
                    "snapshot_base64": primary_snapshot_b64,
                    "camera_id_snapshot": primary_cam_id,
                }

                ok = send_telemetry_to_cloud(payload)
                if ok:
                    total_veh = sum(m.get("vehicles", {}).get("seen", 0) for m in latest_metrics.values())
                    logger.info(
                        f"Telemetría sincronizada con la nube OK | Vehículos acumulados: {total_veh}"
                    )
                last_push_time = now

            # Pausa para ceder CPU al sistema operativo
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Deteniendo Agente NUC Edge...")
    finally:
        for s in streams.values():
            s.stop()
        logger.info("Agente NUC finalizado correctamente.")


if __name__ == "__main__":
    main()
