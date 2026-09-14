"""
main.py — Servidor FastAPI que une todos los módulos del sistema de tráfico.

Responsabilidades:
  - Iniciar todos los CameraStream (hilos de captura).
  - Ejecutar el loop de inferencia en segundo plano (asyncio + ThreadPoolExecutor).
  - Servir el dashboard HTML en la raíz ("/").
  - Exponer:
      GET  /video_feed/{cam_id}   → MJPEG streaming con bounding boxes.
      WS   /ws/metrics            → WebSocket con métricas en tiempo real (JSON).
      GET  /api/stats             → REST snapshot de métricas (para n8n, Novastar, etc.).
      POST /api/reset/{cam_id}    → Reiniciar contadores de una cámara.
      GET  /api/cameras           → Lista de cámaras y su estado de conexión.

Dependencias: fastapi, uvicorn, opencv-python, camera_stream, traffic_analyzer, config
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import cv2
import numpy as np
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pathlib import Path

import config
from camera_stream import CameraStream, create_all_streams
from traffic_analyzer import TrafficAnalyzer, VehicleMetrics
import camera_manager

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO GLOBAL DE LA APLICACIÓN
# (En producción se reemplazaría por un objeto de estado gestionado por inyección)
# ─────────────────────────────────────────────────────────────────────────────

# Streams de captura: cam_id → CameraStream
streams: dict[str, CameraStream] = {}

# Analizadores de tráfico: cam_id → TrafficAnalyzer
analyzers: dict[str, TrafficAnalyzer] = {}

# Último frame anotado (JPEG bytes): cam_id → bytes
latest_frames: dict[str, bytes] = {}

# Últimas métricas calculadas: cam_id → dict
latest_metrics: dict[str, dict] = {}

# Cámaras actualmente habilitadas para inferencia
active_processing: set[str] = set(config.CAMERA_SOURCES)

# Pool de hilos para ejecutar inferencia (bloqueante) sin bloquear el event loop
thread_pool = ThreadPoolExecutor(max_workers=len(config.CAMERA_SOURCES) + 2)

# Clientes WebSocket conectados
ws_clients: list[WebSocket] = []


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN: INICIAR Y DETENER RECURSOS
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Gestiona el ciclo de vida de la aplicación FastAPI:
      - Al arrancar: inicializa streams, analizadores y lanza el loop de inferencia.
      - Al cerrar: detiene hilos y libera recursos limpiamente.
    """
    global streams, analyzers

    logger.info("=== Iniciando sistema de analítica de tráfico ===")

    # ── Cargar líneas persistentes (si existen) ──────────────────────────────
    lines_file = Path("lines.json")
    if lines_file.exists():
        try:
            saved_lines = json.loads(lines_file.read_text(encoding="utf-8"))
            for cam, coords in saved_lines.items():
                if len(coords) == 2 and len(coords[0]) == 2 and len(coords[1]) == 2:
                    config.COUNTING_LINES[cam] = ((coords[0][0], coords[0][1]), (coords[1][0], coords[1][1]))
            logger.info("Líneas persistentes cargadas desde lines.json")
        except Exception as e:
            logger.error(f"Error al cargar lines.json: {e}")

    # ── Inicializar streams de captura ───────────────────────────────────────
    streams = create_all_streams()
    logger.info(f"Streams iniciados: {list(streams.keys())}")

    # ── Inicializar analizadores (carga el modelo YOLO una vez por cámara) ──
    for cam_id in config.CAMERA_SOURCES:
        try:
            analyzers[cam_id] = TrafficAnalyzer(cam_id=cam_id)
            latest_metrics[cam_id] = VehicleMetrics(cam_id=cam_id).to_dict()
        except Exception as exc:
            logger.error(f"No se pudo inicializar el analizador para '{cam_id}': {exc}")

    # ── Lanzar loop de inferencia en background ──────────────────────────────
    inference_task = asyncio.create_task(inference_loop())
    broadcast_task = asyncio.create_task(broadcast_metrics_loop())

    logger.info("=== Sistema listo. Servidor escuchando... ===")

    yield  # Aquí corre el servidor

    # ── Limpieza al cerrar ───────────────────────────────────────────────────
    logger.info("=== Deteniendo sistema... ===")
    inference_task.cancel()
    broadcast_task.cancel()

    for cam_id, stream in streams.items():
        stream.stop()
        logger.info(f"Stream '{cam_id}' detenido.")

    thread_pool.shutdown(wait=False)
    logger.info("=== Sistema detenido limpiamente. ===")


# ─────────────────────────────────────────────────────────────────────────────
# APLICACIÓN FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Traffic Vision Analytics",
    description="Sistema de analítica de tráfico vehicular multicámara con YOLOv8.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOOPS DE BACKGROUND
# ─────────────────────────────────────────────────────────────────────────────

async def inference_loop() -> None:
    """
    Loop asíncrono que ejecuta inferencia YOLO en cada cámara continuamente.

    Usa run_in_executor para que la inferencia (bloqueante) no congele el event loop.
    Actualiza latest_frames y latest_metrics globales.
    """
    loop = asyncio.get_event_loop()

    while True:
        tasks = []
        for cam_id, stream in streams.items():
            if not stream.connected or cam_id not in active_processing:
                continue
            
            frame = stream.get_frame()
            if frame is None:
                continue

            # Ejecutar inferencia en hilo del pool para no bloquear asyncio
            tasks.append(
                loop.run_in_executor(
                    thread_pool,
                    _run_inference,
                    cam_id,
                    frame,
                    stream.fps,
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Pequeño yield para permitir que otros coroutines corran
        await asyncio.sleep(0.001)


def _run_inference(cam_id: str, frame: np.ndarray, fps: float) -> None:
    """
    Ejecuta el ciclo completo de inferencia para una cámara (función síncrona).
    Actualiza latest_frames y latest_metrics.

    Args:
        cam_id : Identificador de la cámara.
        frame  : Frame BGR actual.
        fps    : FPS reportado por el stream (para dwell time).
    """
    try:
        analyzer = analyzers[cam_id]
        annotated_frame, metrics = analyzer.process_frame(frame, stream_fps=fps)

        # Codificar frame anotado como JPEG para streaming MJPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]
        _, buffer = cv2.imencode(".jpg", annotated_frame, encode_params)
        latest_frames[cam_id] = buffer.tobytes()

        # Actualizar métricas globales
        latest_metrics[cam_id] = metrics.to_dict()

    except Exception as exc:
        logger.exception(f"[{cam_id}] Error en inferencia: {exc}")


async def broadcast_metrics_loop() -> None:
    """
    Loop que envía métricas actualizadas a todos los clientes WebSocket conectados.
    Se ejecuta cada WEBSOCKET_BROADCAST_INTERVAL segundos.
    """
    while True:
        await asyncio.sleep(config.WEBSOCKET_BROADCAST_INTERVAL)

        if not ws_clients or not latest_metrics:
            continue

        # Construir payload con métricas de todas las cámaras
        payload = json.dumps({
            "type": "metrics_update",
            "cameras": latest_metrics,
            "server_time": time.time(),
        })

        # Enviar a todos los clientes; eliminar los desconectados
        disconnected = []
        for ws in ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            ws_clients.remove(ws)
            logger.info("Cliente WebSocket desconectado y removido de la lista.")


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, summary="Dashboard principal")
async def serve_dashboard() -> HTMLResponse:
    """
    Sirve el dashboard HTML del sistema.
    Lee el archivo templates/index.html desde disco.
    """
    template_path = Path(__file__).parent / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template index.html no encontrado.")
    
    response = HTMLResponse(content=template_path.read_text(encoding="utf-8"))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — VIDEO FEED (MJPEG Streaming)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/video_feed/{cam_id}", summary="Stream de video con bounding boxes")
async def video_feed(cam_id: str) -> StreamingResponse:
    """
    Devuelve un stream MJPEG con el video anotado de la cámara indicada.
    Compatible con la etiqueta HTML <img src="/video_feed/cam1">.

    Args:
        cam_id: Identificador de la cámara (ej. "cam1").
    """
    if cam_id not in streams:
        raise HTTPException(status_code=404, detail=f"Cámara '{cam_id}' no encontrada.")

    return StreamingResponse(
        _mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _mjpeg_generator(cam_id: str):
    """
    Generador asíncrono que emite frames JPEG en formato MJPEG.
    Soporta desconexión limpia cuando el cliente cierra la conexión.
    """
    # Frame de "sin señal" para cuando no hay frame disponible
    no_signal_frame = _create_no_signal_frame(cam_id)

    while True:
        frame_bytes = latest_frames.get(cam_id)

        if frame_bytes is None:
            # Todavía no hay frames del analizador; enviar frame de espera
            frame_bytes = no_signal_frame

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

        # Limitar a ~30 fps máximo para no saturar el navegador
        await asyncio.sleep(1 / 30)


def _create_no_signal_frame(cam_id: str) -> bytes:
    """
    Crea un frame JPEG de "Sin señal" para mostrar mientras la cámara conecta.

    Returns:
        Bytes JPEG del frame de placeholder.
    """
    frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)   # Fondo gris muy oscuro

    cv2.putText(
        frame, f"Sin senal - {cam_id}",
        (config.FRAME_WIDTH // 2 - 200, config.FRAME_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "Esperando conexion...",
        (config.FRAME_WIDTH // 2 - 160, config.FRAME_HEIGHT // 2 + 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2, cv2.LINE_AA,
    )

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return buffer.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — WEBSOCKET
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket) -> None:
    """
    Endpoint WebSocket para métricas en tiempo real.

    El servidor envía un JSON cada WEBSOCKET_BROADCAST_INTERVAL seg con la estructura:
    {
      "type": "metrics_update",
      "cameras": { "cam1": { ...VehicleMetrics... }, "cam2": { ... } },
      "server_time": 1234567890.123
    }
    """
    await websocket.accept()
    ws_clients.append(websocket)
    client_host = websocket.client.host if websocket.client else "desconocido"
    logger.info(f"Nuevo cliente WebSocket conectado: {client_host}")

    try:
        # Enviar estado inicial inmediatamente al conectar
        initial_payload = json.dumps({
            "type": "initial_state",
            "cameras": latest_metrics,
            "server_time": time.time(),
        })
        await websocket.send_text(initial_payload)

        # Mantener la conexión abierta; el broadcast_metrics_loop envía updates
        while True:
            # Escuchar mensajes del cliente (comandos futuros: ej. reset)
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                await _handle_ws_message(msg, websocket)
            except json.JSONDecodeError:
                logger.warning(f"Mensaje WebSocket no válido de {client_host}: {data[:100]}")

    except WebSocketDisconnect:
        logger.info(f"Cliente WebSocket desconectado: {client_host}")
    except Exception as exc:
        logger.exception(f"Error en WebSocket ({client_host}): {exc}")
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)


async def _handle_ws_message(msg: dict, websocket: WebSocket) -> None:
    """
    Procesa comandos recibidos por WebSocket desde el frontend.
    Comandos soportados:
      { "action": "reset", "cam_id": "cam1" }
      { "action": "ping" }
    """
    action = msg.get("action")

    if action == "reset":
        cam_id = msg.get("cam_id")
        if cam_id and cam_id in analyzers:
            analyzers[cam_id].reset_counts()
            await websocket.send_text(json.dumps({"type": "reset_ok", "cam_id": cam_id}))
        else:
            await websocket.send_text(json.dumps({"type": "error", "detail": f"cam_id '{cam_id}' no encontrado"}))

    elif action == "ping":
        await websocket.send_text(json.dumps({"type": "pong", "server_time": time.time()}))


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — API REST (para n8n, Novastar, integraciones externas)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/stats", summary="Snapshot de métricas de todas las cámaras")
async def api_stats() -> JSONResponse:
    """
    Devuelve un snapshot JSON con las métricas actuales de todas las cámaras.
    Ideal para consumo externo desde n8n, dashboards BI, o procesadores Novastar.

    Respuesta:
    {
      "status": "ok",
      "server_time": 1234567890.123,
      "cameras": { "cam1": { ...métricas... } }
    }
    """
    return JSONResponse({
        "status": "ok",
        "server_time": time.time(),
        "cameras": latest_metrics,
    })


@app.get("/api/stats/{cam_id}", summary="Métricas de una cámara específica")
async def api_stats_cam(cam_id: str) -> JSONResponse:
    """
    Devuelve las métricas de una sola cámara.

    Args:
        cam_id: Identificador de la cámara (ej. "cam1").
    """
    if cam_id not in latest_metrics:
        raise HTTPException(status_code=404, detail=f"Cámara '{cam_id}' no encontrada.")
    return JSONResponse({
        "status": "ok",
        "server_time": time.time(),
        "camera": latest_metrics[cam_id],
    })


@app.post("/api/reset/{cam_id}", summary="Reiniciar contadores de una cámara")
async def api_reset(cam_id: str) -> JSONResponse:
    """
    Reinicia todos los contadores de la cámara indicada.
    Útil al inicio de un nuevo turno o jornada de medición.
    """
    if cam_id not in analyzers:
        raise HTTPException(status_code=404, detail=f"Cámara '{cam_id}' no encontrada.")
    analyzers[cam_id].reset_counts()
    latest_metrics[cam_id] = VehicleMetrics(cam_id=cam_id).to_dict()
    logger.info(f"[API] Contadores reiniciados para '{cam_id}'.")
    return JSONResponse({"status": "ok", "message": f"Contadores de '{cam_id}' reiniciados."})


class LineUpdate(BaseModel):
    start: tuple[int, int]
    end: tuple[int, int]

@app.get("/api/line/{cam_id}", summary="Obtener línea actual de la cámara")
async def get_line(cam_id: str) -> JSONResponse:
    if cam_id not in config.COUNTING_LINES:
        raise HTTPException(status_code=404, detail=f"Cámara '{cam_id}' no configurada.")
    coords = config.COUNTING_LINES[cam_id]
    return JSONResponse({
        "status": "ok",
        "cam_id": cam_id,
        "line": {"start": coords[0], "end": coords[1]}
    })

@app.post("/api/line/{cam_id}", summary="Actualizar línea de conteo")
async def update_line(cam_id: str, line_data: LineUpdate) -> JSONResponse:
    if cam_id not in analyzers:
        raise HTTPException(status_code=404, detail=f"Cámara '{cam_id}' no encontrada.")
    
    new_coords = (line_data.start, line_data.end)
    
    # 1. Actualizar configuración en memoria
    config.COUNTING_LINES[cam_id] = new_coords
    
    # 2. Persistir en disco
    lines_file = Path("lines.json")
    saved_lines = {}
    if lines_file.exists():
        try:
            saved_lines = json.loads(lines_file.read_text(encoding="utf-8"))
        except:
            pass
    saved_lines[cam_id] = new_coords
    lines_file.write_text(json.dumps(saved_lines, indent=2), encoding="utf-8")
    
    # 3. Actualizar el analizador
    analyzers[cam_id].update_line(new_coords)
    
    return JSONResponse({
        "status": "ok",
        "message": f"Línea de '{cam_id}' actualizada y guardada."
    })


class CamToggle(BaseModel):
    enabled: bool

@app.post("/api/camera/{cam_id}/toggle", summary="Habilitar o deshabilitar inferencia de la cámara")
async def toggle_cam(cam_id: str, data: CamToggle) -> JSONResponse:
    if cam_id not in analyzers:
        raise HTTPException(status_code=404, detail=f"Cámara '{cam_id}' no encontrada.")
    if data.enabled:
        active_processing.add(cam_id)
    else:
        active_processing.discard(cam_id)
    logger.info(f"[API] Cámara '{cam_id}' inferencia activa: {data.enabled}")
    return JSONResponse({"status": "ok", "cam_id": cam_id, "enabled": data.enabled})


@app.get("/api/devices", summary="Listar cámaras USB y streams activos")
async def api_devices() -> JSONResponse:
    """
    Escanea puertos USB y devuelve las cámaras disponibles,
    junto con los streams actualmente configurados y activos.
    """
    usb_cameras = camera_manager.scan_usb_cameras()
    active = list(streams.keys())
    return JSONResponse({
        "usb_cameras": usb_cameras,
        "active_streams": active,
        "sources": config.CAMERA_SOURCES
    })


class CameraAddRequest(BaseModel):
    source: str | int
    cam_id: str

@app.post("/api/camera/add", summary="Agregar nueva cámara al dashboard")
async def api_camera_add(req: CameraAddRequest) -> JSONResponse:
    """Añade una cámara nueva (USB o RTSP) en tiempo real."""
    if req.cam_id in streams:
        raise HTTPException(status_code=400, detail=f"La cámara {req.cam_id} ya existe.")
    
    if isinstance(req.source, str) and str(req.source).startswith("rtsp"):
        if not camera_manager.validate_rtsp_stream(req.source):
            raise HTTPException(status_code=400, detail="El stream RTSP no es válido o está inactivo.")
            
    # Agregar a globales
    config.CAMERA_SOURCES[req.cam_id] = req.source
    
    stream = CameraStream(req.cam_id, req.source)
    streams[req.cam_id] = stream
    stream.start()
    
    analyzers[req.cam_id] = TrafficAnalyzer(req.cam_id)
    active_processing.add(req.cam_id)
    
    # Línea por defecto si no existe
    if req.cam_id not in config.COUNTING_LINES:
        config.COUNTING_LINES[req.cam_id] = ((100, 360), (1180, 360))
        # Actualizar config de líneas para persistir? 
        # (Aquí podríamos llamar a la función que guarda en lines.json si existiera)
    
    return JSONResponse({"status": "ok", "cam_id": req.cam_id, "message": "Cámara agregada exitosamente."})



@app.get("/api/cameras", summary="Lista de cámaras y estado de conexión")
async def api_cameras() -> JSONResponse:
    """
    Devuelve el estado de conexión de todas las cámaras configuradas.
    """
    cam_status = {}
    for cam_id, stream in streams.items():
        cam_status[cam_id] = {
            "source": str(stream.source),
            "connected": stream.connected,
            "fps": round(stream.fps, 1),
            "reconnect_count": stream.reconnect_count,
        }
    return JSONResponse({"status": "ok", "cameras": cam_status})


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,         # Desactivar reload en producción (incompatible con threads)
        log_level="info",
    )
