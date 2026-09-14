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
import datetime
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import shutil

import cv2
import numpy as np
import uvicorn
import io
import csv
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Response, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import config
from camera_stream import CameraStream, create_all_streams
from traffic_analyzer import TrafficAnalyzer, VehicleMetrics
import camera_manager
import database
import novastar_controller
import novastar_vx600
import shelly_controller
import campaign_manager
import solar_brightness_engine

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

    # ── Inicializar Base de Datos SQLite ───────────────────────────────────────
    try:
        database.init_db()
        logger.info("Base de datos SQLite inicializada correctamente.")
    except Exception as e:
        logger.error(f"Error inicializando la base de datos: {e}")

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
    solar_task = asyncio.create_task(solar_auto_brightness_loop())

    logger.info("=== Sistema listo. Servidor escuchando... ===")

    yield  # Aquí corre el servidor

    # ── Limpieza al cerrar ───────────────────────────────────────────────────
    logger.info("=== Deteniendo sistema... ===")
    inference_task.cancel()
    broadcast_task.cancel()
    solar_task.cancel()

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

# ── ARCHIVOS ESTÁTICOS Y LOGOS ───────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "uploads" / "logos").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")



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


async def solar_auto_brightness_loop() -> None:
    """
    Loop en background que ajusta periódicamente el brillo del Novastar TB40
    conforme a la posición solar y orientación de la pantalla de cada ubicación activa.
    """
    logger.info("Iniciando loop de autorregulación solar de brillo...")
    last_applied_brightness: dict[str, int] = {}
    
    while True:
        try:
            locs = database.get_locations()
            for loc in locs:
                loc_id = loc["id"]
                auto_enabled = bool(loc.get("auto_brightness_enabled", 1))
                if not auto_enabled:
                    continue

                lat = float(loc.get("latitude") or 20.6736)
                lon = float(loc.get("longitude") or -103.3855)
                orientation = float(loc.get("screen_orientation_deg") or 270.0)
                min_b = int(loc.get("min_night_brightness") or 25)
                max_b = int(loc.get("max_day_brightness") or 95)
                nova_ip = loc.get("novastar_ip", "192.168.1.140")
                nova_port = int(loc.get("novastar_port", 8001))

                pos = solar_brightness_engine.calculate_solar_position(lat, lon)
                calc = solar_brightness_engine.compute_target_brightness(
                    pos["elevation_deg"], pos["azimuth_deg"], orientation, min_b, max_b
                )
                target = calc["recommended_brightness"]

                if last_applied_brightness.get(loc_id) != target:
                    logger.info(f"[SOLAR-AUTO] Ajustando pantalla '{loc_id}' ({orientation}° azimut) a {target}% ({calc['condition']})")
                    novastar_controller.set_brightness(ip=nova_ip, brightness=target, port=nova_port)
                    last_applied_brightness[loc_id] = target
        except Exception as e:
            logger.debug(f"Error en loop solar auto: {e}")
            
        await asyncio.sleep(60.0)  # Verificar cada 60 segundos


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


def _create_disabled_frame(cam_id: str) -> bytes:
    """
    Crea un frame JPEG corporativo indicando que la cámara está desactivada/pausada.
    """
    frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:] = (22, 28, 38)   # Fondo slate oscuro
    
    cv2.putText(
        frame, f"CAMARA DESACTIVADA - {cam_id.upper()}",
        (config.FRAME_WIDTH // 2 - 270, config.FRAME_HEIGHT // 2 - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220, 220, 230), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "Captura e inferencia en pausa (Ahorro de NPU/CPU)",
        (config.FRAME_WIDTH // 2 - 250, config.FRAME_HEIGHT // 2 + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (130, 150, 170), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "Activala desde el panel superior o dispositivos",
        (config.FRAME_WIDTH // 2 - 230, config.FRAME_HEIGHT // 2 + 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 160, 240), 1, cv2.LINE_AA,
    )
    
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return buffer.tobytes()


async def _mjpeg_generator(cam_id: str):
    """
    Generador asíncrono que emite frames JPEG en formato MJPEG.
    Soporta desconexión limpia cuando el cliente cierra la conexión.
    """
    no_signal_frame = _create_no_signal_frame(cam_id)
    disabled_frame = _create_disabled_frame(cam_id)

    while True:
        stream = streams.get(cam_id)
        is_active = (cam_id in active_processing) and (stream is None or getattr(stream, "enabled", True))
        
        if not is_active:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + disabled_frame
                + b"\r\n"
            )
            await asyncio.sleep(0.5)
            continue

        frame_bytes = latest_frames.get(cam_id)

        if frame_bytes is None:
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

@app.post("/api/camera/{cam_id}/toggle", summary="Habilitar o deshabilitar captura e inferencia de la cámara")
async def toggle_cam(cam_id: str, data: CamToggle) -> JSONResponse:
    if cam_id not in streams and cam_id not in analyzers:
        raise HTTPException(status_code=404, detail=f"Cámara '{cam_id}' no encontrada.")
    
    stream = streams.get(cam_id)
    if data.enabled:
        active_processing.add(cam_id)
        if stream:
            stream.set_enabled(True)
    else:
        active_processing.discard(cam_id)
        if stream:
            stream.set_enabled(False)
        if cam_id in latest_frames:
            latest_frames[cam_id] = _create_disabled_frame(cam_id)

    logger.info(f"[API] Cámara '{cam_id}' estado activada/desactivada: {data.enabled}")
    return JSONResponse({"status": "ok", "cam_id": cam_id, "enabled": data.enabled})


@app.get("/api/devices", summary="Listar cámaras USB y streams activos")
async def api_devices() -> JSONResponse:
    """
    Escanea puertos USB y devuelve las cámaras disponibles,
    junto con los streams actualmente configurados y su estado de activación.
    """
    usb_cameras = camera_manager.scan_usb_cameras()
    cam_states = {}
    for cid, st in streams.items():
        cam_states[cid] = {
            "source": str(st.source),
            "enabled": getattr(st, "enabled", True) and (cid in active_processing),
            "connected": st.connected
        }
    return JSONResponse({
        "usb_cameras": usb_cameras,
        "active_streams": list(streams.keys()),
        "configured_cameras": cam_states,
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
    
    return JSONResponse({"status": "ok", "cam_id": req.cam_id, "message": "Cámara agregada exitosamente."})



@app.get("/api/cameras", summary="Lista de cámaras y estado de conexión y activación")
async def api_cameras() -> JSONResponse:
    """
    Devuelve el estado de conexión y activación de todas las cámaras configuradas.
    """
    cam_status = {}
    for cam_id, stream in streams.items():
        is_enabled = getattr(stream, "enabled", True) and (cam_id in active_processing)
        cam_status[cam_id] = {
            "source": str(stream.source),
            "connected": stream.connected if is_enabled else False,
            "enabled": is_enabled,
            "fps": round(stream.fps, 1) if is_enabled else 0.0,
            "reconnect_count": stream.reconnect_count,
        }
    return JSONResponse({"status": "ok", "cameras": cam_status})


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS DE ANALÍTICA HISTÓRICA Y REGISTRO DE EVENTOS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/history/summary", summary="Resumen ejecutivo de KPIs históricos")
async def api_history_summary(
    camera_id: str = Query("all", description="ID de cámara o 'all'"),
    date: str | None = Query(None, description="Fecha YYYY-MM-DD")
) -> JSONResponse:
    """Retorna KPIs ejecutivos: aforo total, in, out, hora pico, tiempo de estancia y composición."""
    data = database.get_kpis(camera_id=camera_id, date_str=date)
    return JSONResponse({"status": "ok", "data": data})


@app.get("/api/history/hourly", summary="Métricas de aforo por hora (00:00 - 23:00)")
async def api_history_hourly(
    camera_id: str = Query("all", description="ID de cámara o 'all'"),
    date: str | None = Query(None, description="Fecha YYYY-MM-DD")
) -> JSONResponse:
    """Retorna la distribución horaria completa para gráficas de barras y detección de horas pico."""
    data = database.get_hourly_metrics(camera_id=camera_id, date_str=date)
    return JSONResponse({"status": "ok", "data": data})


@app.get("/api/history/daily", summary="Tendencia diaria histórica")
async def api_history_daily(
    camera_id: str = Query("all", description="ID de cámara o 'all'"),
    days: int = Query(7, description="Cantidad de días hacia atrás (7, 14, 30)")
) -> JSONResponse:
    """Retorna el volumen total de vehículos por día."""
    data = database.get_daily_metrics(camera_id=camera_id, days=days)
    return JSONResponse({"status": "ok", "data": data})


@app.get("/api/history/events", summary="Registro filtrado y paginado de eventos de cruce")
async def api_history_events(
    camera_id: str = Query("all"),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    vehicle_type: str = Query("all"),
    direction: str = Query("all"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
) -> JSONResponse:
    """Consulta la bitácora de eventos con filtros dinámicos."""
    data = database.get_events(
        camera_id=camera_id,
        start_date=start_date,
        end_date=end_date,
        vehicle_type=vehicle_type,
        direction=direction,
        limit=limit,
        offset=offset
    )
    return JSONResponse({"status": "ok", "data": data})


@app.get("/api/history/export/csv", summary="Exportar eventos a archivo CSV para Excel")
async def api_history_export_csv(
    camera_id: str = Query("all"),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    vehicle_type: str = Query("all"),
    direction: str = Query("all")
) -> Response:
    """Genera y descarga un archivo CSV con el historial de eventos para análisis y auditoría."""
    data = database.get_events(
        camera_id=camera_id,
        start_date=start_date,
        end_date=end_date,
        vehicle_type=vehicle_type,
        direction=direction,
        limit=5000,
        offset=0
    )
    
    output = io.StringIO()
    writer = csv.writer(output)
    # Encabezados
    writer.writerow(["ID", "Fecha_Hora", "Camara", "Track_ID", "Tipo_Vehiculo", "Sentido", "Confianza", "Tiempo_Permanencia_Seg"])
    
    for ev in data["events"]:
        writer.writerow([
            ev["id"],
            ev["timestamp"],
            ev["camera_id"],
            ev["track_id"],
            ev["vehicle_type"],
            ev["direction"],
            ev["confidence"],
            ev["dwell_time"]
        ])
        
    csv_content = output.getvalue()
    filename = f"reporte_aforo_vehicular_{camera_id}_{start_date or 'inicio'}.csv"
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS: GESTIÓN MULTI-UBICACIÓN, NOVASTAR TB40 Y SHELLY PRO
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/locations", summary="Listar ubicaciones y pantallas registradas")
async def api_get_locations() -> JSONResponse:
    """Devuelve todas las ubicaciones y la configuración de hardware de cada una."""
    locs = database.get_locations()
    return JSONResponse({"status": "ok", "locations": locs})


class ShellyPowerRequest(BaseModel):
    ip: str = "192.168.1.150"
    turn_on: bool
    channel: int = 0

@app.get("/api/shelly/status", summary="Telemetría de consumo eléctrico Shelly Pro")
async def api_shelly_status(ip: str = Query("192.168.1.150"), channel: int = Query(0)) -> JSONResponse:
    """Retorna potencia (Watts), voltaje (V), corriente (A) y energía acumulada (kWh)."""
    status = shelly_controller.get_status(ip=ip, channel=channel)
    return JSONResponse({"status": "ok", "telemetry": status})

@app.post("/api/shelly/power", summary="Encender o apagar relé Shelly Pro (Contactor de Pantalla)")
async def api_shelly_power(req: ShellyPowerRequest) -> JSONResponse:
    """Conmuta el suministro eléctrico principal de la pantalla LED."""
    res = shelly_controller.set_power(ip=req.ip, turn_on=req.turn_on, channel=req.channel)
    return JSONResponse({"status": "ok", "result": res})


class NovastarPowerRequest(BaseModel):
    ip: str = "192.168.1.140"
    power_on: bool
    port: int = 8001

class NovastarBrightnessRequest(BaseModel):
    ip: str = "192.168.1.140"
    brightness: int
    port: int = 8001

@app.get("/api/novastar/status", summary="Estado del reproductor Novastar TB40")
async def api_novastar_status(ip: str = Query("192.168.1.140"), port: int = Query(8001)) -> JSONResponse:
    """Obtiene el estado de conexión, pantalla y brillo del TB40."""
    status = novastar_controller.get_status(ip=ip, port=port)
    return JSONResponse({"status": "ok", "status_info": status})

@app.post("/api/novastar/power", summary="Encender o poner en Standby pantalla Novastar TB40")
async def api_novastar_power(req: NovastarPowerRequest) -> JSONResponse:
    """Activa o pone en pantalla negra la salida de video del procesador Novastar."""
    res = novastar_controller.set_screen_power(ip=req.ip, power_on=req.power_on, port=req.port)
    return JSONResponse({"status": "ok", "result": res})

@app.post("/api/novastar/brightness", summary="Ajustar brillo de pantalla Novastar TB40 (0-100%)")
async def api_novastar_brightness(req: NovastarBrightnessRequest) -> JSONResponse:
    """Regula el nivel de brillo de la pantalla LED."""
    res = novastar_controller.set_brightness(ip=req.ip, brightness=req.brightness, port=req.port)
    return JSONResponse({"status": "ok", "result": res})


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS: CALENDARIO Y REGULACIÓN AUTOMÁTICA DE BRILLO SOLAR
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/solar/status", summary="Posición solar y brillo recomendado por orientación")
async def api_solar_status(location_id: str = Query("loc_minerva")) -> JSONResponse:
    """
    Calcula la posición astronómica del sol actual y el nivel de brillo óptimo
    en función de la orientación física de la pantalla.
    """
    loc = database.get_location(location_id)
    if not loc:
        locs = database.get_locations()
        loc = locs[0] if locs else {
            "latitude": 20.6736, "longitude": -103.3855, "screen_orientation_deg": 270.0,
            "min_night_brightness": 25, "max_day_brightness": 95, "auto_brightness_enabled": 1
        }
        
    lat = float(loc.get("latitude") or 20.6736)
    lon = float(loc.get("longitude") or -103.3855)
    orientation = float(loc.get("screen_orientation_deg") or 270.0)
    min_b = int(loc.get("min_night_brightness") or 25)
    max_b = int(loc.get("max_day_brightness") or 95)
    auto_b = int(loc.get("auto_brightness_enabled") or 1)

    solar_pos = solar_brightness_engine.calculate_solar_position(lat, lon)
    ephem = solar_brightness_engine.calculate_sun_ephemeris(lat, lon)
    bright = solar_brightness_engine.compute_target_brightness(
        solar_elevation_deg=solar_pos["elevation_deg"],
        solar_azimuth_deg=solar_pos["azimuth_deg"],
        screen_orientation_deg=orientation,
        min_night_brightness=min_b,
        max_day_brightness=max_b
    )
    cardinal = solar_brightness_engine._get_cardinal_label(orientation)

    return JSONResponse({
        "status": "ok",
        "location_id": location_id,
        "location_name": loc.get("name", location_id),
        "coordinates": {"lat": lat, "lon": lon},
        "orientation": {"degrees": orientation, "cardinal": cardinal},
        "solar": solar_pos,
        "ephemeris": ephem,
        "brightness": bright,
        "auto_brightness_enabled": bool(auto_b)
    })

@app.get("/api/solar/schedule", summary="Curva horaria de 24 horas y calendario solar")
async def api_solar_schedule(location_id: str = Query("loc_minerva")) -> JSONResponse:
    """Retorna la curva de proyección solar de 24 horas desglosada cada 30 minutos."""
    loc = database.get_location(location_id)
    if not loc:
        locs = database.get_locations()
        loc = locs[0] if locs else {
            "latitude": 20.6736, "longitude": -103.3855, "screen_orientation_deg": 270.0,
            "min_night_brightness": 25, "max_day_brightness": 95
        }
        
    lat = float(loc.get("latitude") or 20.6736)
    lon = float(loc.get("longitude") or -103.3855)
    orientation = float(loc.get("screen_orientation_deg") or 270.0)
    min_b = int(loc.get("min_night_brightness") or 25)
    max_b = int(loc.get("max_day_brightness") or 95)

    sched = solar_brightness_engine.generate_24h_solar_schedule(
        lat=lat,
        lon=lon,
        screen_orientation_deg=orientation,
        min_night_brightness=min_b,
        max_day_brightness=max_b
    )
    return JSONResponse({
        "status": "ok",
        "location_id": location_id,
        "schedule": sched
    })

class SolarConfigRequest(BaseModel):
    location_id: str
    screen_orientation_deg: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    min_night_brightness: Optional[int] = None
    max_day_brightness: Optional[int] = None
    auto_brightness_enabled: Optional[bool] = None

@app.post("/api/solar/config", summary="Actualizar orientación y parámetros solares de la pantalla")
async def api_solar_config(req: SolarConfigRequest) -> JSONResponse:
    """Guarda la orientación angular (° azimut) y límites de brillo."""
    auto_int = 1 if req.auto_brightness_enabled is True else (0 if req.auto_brightness_enabled is False else None)
    success = database.update_location_solar_config(
        loc_id=req.location_id,
        screen_orientation_deg=req.screen_orientation_deg,
        latitude=req.latitude,
        longitude=req.longitude,
        min_night_brightness=req.min_night_brightness,
        max_day_brightness=req.max_day_brightness,
        auto_brightness_enabled=auto_int
    )
    if not success:
        raise HTTPException(status_code=404, detail="Ubicación no encontrada.")

    applied_br = None
    if req.auto_brightness_enabled:
        loc = database.get_location(req.location_id)
        if loc:
            pos = solar_brightness_engine.calculate_solar_position(loc["latitude"], loc["longitude"])
            calc = solar_brightness_engine.compute_target_brightness(
                pos["elevation_deg"], pos["azimuth_deg"], req.screen_orientation_deg,
                loc["min_night_brightness"], loc["max_day_brightness"]
            )
            applied_br = calc["recommended_brightness"]
            novastar_controller.set_brightness(
                ip=loc.get("novastar_ip", "192.168.1.140"),
                brightness=applied_br,
                port=loc.get("novastar_port", 8001)
            )

    return JSONResponse({
        "status": "ok",
        "message": "Configuración solar y orientación guardadas satisfactoriamente.",
        "applied_brightness": applied_br
    })

class SolarApplyRequest(BaseModel):
    location_id: str

@app.post("/api/solar/apply-now", summary="Ajustar inmediatamente pantalla según posición solar actual")
async def api_solar_apply_now(req: SolarApplyRequest) -> JSONResponse:
    """Calcula en tiempo real y empuja el brillo solar al Novastar TB40."""
    loc = database.get_location(req.location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Ubicación no encontrada.")

    lat = float(loc.get("latitude") or 20.6736)
    lon = float(loc.get("longitude") or -103.3855)
    orientation = float(loc.get("screen_orientation_deg") or 270.0)
    min_b = int(loc.get("min_night_brightness") or 25)
    max_b = int(loc.get("max_day_brightness") or 95)

    pos = solar_brightness_engine.calculate_solar_position(lat, lon)
    calc = solar_brightness_engine.compute_target_brightness(
        pos["elevation_deg"], pos["azimuth_deg"], orientation, min_b, max_b
    )
    target_br = calc["recommended_brightness"]
    
    res = novastar_controller.set_brightness(
        ip=loc.get("novastar_ip", "192.168.1.140"),
        brightness=target_br,
        port=loc.get("novastar_port", 8001)
    )
    
    return JSONResponse({
        "status": "ok",
        "location_id": req.location_id,
        "target_brightness": target_br,
        "condition": calc["condition"],
        "novastar_result": res
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS: CAMPAÑAS PUBLICITARIAS Y CONSUMO ENERGÉTICO
# ─────────────────────────────────────────────────────────────────────────────

class CampaignCreateRequest(BaseModel):
    name: str
    client: str
    location_id: str
    start_date: str
    end_date: str
    start_hour: int = 6
    end_hour: int = 23
    spot_seconds: int = 15
    spots_per_hour: int = 12

@app.get("/api/campaigns", summary="Listar campañas publicitarias")
async def api_get_campaigns(location_id: str = Query("all")) -> JSONResponse:
    """Retorna las campañas activas o filtradas por ubicación."""
    campaigns = database.get_campaigns(location_id=location_id)
    return JSONResponse({"status": "ok", "campaigns": campaigns})

@app.post("/api/campaigns", summary="Crear nueva campaña publicitaria")
async def api_create_campaign(req: CampaignCreateRequest) -> JSONResponse:
    """Registra una pauta publicitaria para seguimiento de aforo y energía."""
    cid = database.add_campaign(
        name=req.name,
        client=req.client,
        location_id=req.location_id,
        start_date=req.start_date,
        end_date=req.end_date,
        start_hour=req.start_hour,
        end_hour=req.end_hour,
        spot_seconds=req.spot_seconds,
        spots_per_hour=req.spots_per_hour
    )
    return JSONResponse({"status": "ok", "campaign_id": cid, "message": "Campaña creada exitosamente."})

@app.get("/api/campaigns/summary", summary="Resumen consolidado de aforo y energía por campaña")
async def api_campaigns_summary(location_id: str = Query("all")) -> JSONResponse:
    """Calcula aforo impactado, energía consumida (kWh) y costo monetario por campaña."""
    data = campaign_manager.get_all_campaigns_summary(location_id=location_id)
    return JSONResponse({"status": "ok", "data": data})


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS: RECEPCIÓN DE REPORTES EDGE (NODOS REMOTOS NUC)
# ─────────────────────────────────────────────────────────────────────────────

class EdgeSyncRequest(BaseModel):
    node_id: str
    location_id: str
    date_reported: str
    total_vehicles: int
    total_in: int
    total_out: int
    peak_hour: str
    raw_summary: str = ""

@app.post("/api/edge/sync-report", summary="Recepción de reporte ligero desde nodo NUC local")
async def api_edge_sync_report(req: EdgeSyncRequest) -> JSONResponse:
    """
    Permite que computadoras locales envíen únicamente su reporte estadístico consolidado
    sin transferir video pesado a la nube, optimizando el ancho de banda y los costos.
    """
    rid = database.record_edge_sync(
        node_id=req.node_id,
        location_id=req.location_id,
        date_reported=req.date_reported,
        total_vehicles=req.total_vehicles,
        total_in=req.total_in,
        total_out=req.total_out,
        peak_hour=req.peak_hour,
        raw_summary=req.raw_summary
    )
    logger.info(f"[EDGE-SYNC] Reporte recibido de nodo '{req.node_id}' ({req.location_id}): {req.total_vehicles} veh.")
    return JSONResponse({"status": "ok", "sync_id": rid, "message": "Reporte edge recibido y almacenado."})


# ─────────────────────────────────────────────────────────────────────────────
# GESTIÓN MULTI-CLIENTE Y LOGOTIPOS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/clients", summary="Listado de clientes registrados y pantallas asignadas")
async def api_get_clients() -> JSONResponse:
    clients = database.get_clients()
    return JSONResponse({"status": "ok", "clients": clients})


class ClientCreate(BaseModel):
    id: str
    name: str
    contact_email: Optional[str] = ""
    logo_url: Optional[str] = ""
    location_ids: Optional[list[str]] = []

@app.post("/api/clients", summary="Registrar o actualizar cliente")
async def api_post_client(req: ClientCreate) -> JSONResponse:
    cid = database.add_client(req.id, req.name, req.contact_email or "", req.logo_url or "")
    if req.location_ids:
        for loc in req.location_ids:
            database.assign_client_location(cid, loc)
    return JSONResponse({"status": "ok", "client_id": cid, "message": "Cliente guardado exitosamente."})


@app.post("/api/client/{client_id}/logo", summary="Subir logotipo corporativo del cliente")
async def api_upload_client_logo(client_id: str, file: UploadFile = File(...)) -> JSONResponse:
    client = database.get_client(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Cliente '{client_id}' no encontrado.")
    
    ext = Path(file.filename or "logo.png").suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".svg", ".webp"]:
        raise HTTPException(status_code=400, detail="Formato de imagen no soportado. Utilice PNG, JPG, SVG o WEBP.")
    
    filename = f"{client_id}_{int(time.time())}{ext}"
    dest_path = static_dir / "uploads" / "logos" / filename
    
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    logo_url = f"/static/uploads/logos/{filename}"
    database.update_client_logo(client_id, logo_url)
    logger.info(f"[CLIENT-LOGO] Logotipo actualizado para '{client_id}': {logo_url}")
    return JSONResponse({"status": "ok", "client_id": client_id, "logo_url": logo_url})


class LogoUrlUpdate(BaseModel):
    logo_url: str

@app.post("/api/client/{client_id}/logo-url", summary="Asignar URL de logotipo de cliente")
async def api_set_client_logo_url(client_id: str, req: LogoUrlUpdate) -> JSONResponse:
    client = database.get_client(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Cliente '{client_id}' no encontrado.")
    database.update_client_logo(client_id, req.logo_url)
    return JSONResponse({"status": "ok", "client_id": client_id, "logo_url": req.logo_url})


@app.get("/api/client/{client_id}/screens", summary="Pantallas y métricas asignadas a un cliente")
async def api_get_client_screens(client_id: str) -> JSONResponse:
    client = database.get_client(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Cliente '{client_id}' no encontrado.")
    
    screens = database.get_client_screens(client_id)
    enriched_screens = []
    for s in screens:
        loc_id = s["id"]
        devs = database.get_hardware_devices(loc_id)
        enriched_screens.append({
            **s,
            "devices": devs,
        })
    return JSONResponse({"status": "ok", "client": client, "screens": enriched_screens})


# ─────────────────────────────────────────────────────────────────────────────
# GESTIÓN DE DISPOSITIVOS DE HARDWARE (TB40 / VX600 PRO / SHELLY PRO)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/admin/devices", summary="Listado de dispositivos con telemetría en vivo")
async def api_admin_devices(location_id: Optional[str] = None) -> JSONResponse:
    devices = database.get_hardware_devices(location_id)
    enriched = []
    for d in devices:
        item = dict(d)
        dtype = item.get("device_type", "")
        ip = item.get("ip_address", "127.0.0.1")
        port = item.get("port", 8001)

        if dtype == "VX600_PRO":
            status = novastar_vx600.get_status(ip, port)
            item["telemetry"] = status
            item["active_preset"] = status.get("active_preset", item.get("active_preset"))
            item["preset_info"] = status.get("preset_info")
        elif dtype == "TB40":
            status = novastar_controller.get_status(ip, port)
            item["telemetry"] = status
        elif dtype == "SHELLY_PRO":
            status = shelly_controller.get_status(ip)
            item["telemetry"] = status
        elif dtype == "HYBRID":
            vx_status = novastar_vx600.get_status(ip, 6000)
            tb_status = novastar_controller.get_status(ip, port)
            item["telemetry"] = {"vx600": vx_status, "tb40": tb_status}
            item["active_preset"] = vx_status.get("active_preset")
            item["preset_info"] = vx_status.get("preset_info")
        else:
            item["telemetry"] = {"connected": True}

        enriched.append(item)
    return JSONResponse({
        "status": "ok",
        "devices": enriched,
        "presets_catalog": novastar_vx600.get_presets()
    })


class DeviceCreate(BaseModel):
    id: str
    location_id: str
    name: str
    device_type: str  # 'TB40', 'VX600_PRO', 'HYBRID', 'SHELLY_PRO'
    ip_address: str
    port: Optional[int] = 8001
    dual_screen_enabled: Optional[bool] = False
    active_preset: Optional[str] = "preset_mirror_1tb40"
    input_source_1: Optional[str] = "TB40_MASTER"
    input_source_2: Optional[str] = ""
    notes: Optional[str] = ""

@app.post("/api/admin/devices", summary="Alta de nuevo dispositivo o controlador")
async def api_admin_add_device(req: DeviceCreate) -> JSONResponse:
    did = database.add_hardware_device(
        device_id=req.id,
        location_id=req.location_id,
        name=req.name,
        device_type=req.device_type,
        ip_address=req.ip_address,
        port=req.port or (6000 if req.device_type == "VX600_PRO" else 8001),
        dual_screen_enabled=1 if req.dual_screen_enabled else 0,
        active_preset=req.active_preset or "preset_mirror_1tb40",
        input_source_1=req.input_source_1 or "TB40_MASTER",
        input_source_2=req.input_source_2 or "",
        notes=req.notes or ""
    )
    if req.device_type in ("VX600_PRO", "HYBRID") and req.active_preset:
        novastar_vx600.apply_preset(req.ip_address, req.active_preset, req.port or 6000)

    logger.info(f"[ADMIN-DEVICE] Dispositivo registrado: '{did}' ({req.device_type}) en {req.ip_address}")
    return JSONResponse({"status": "ok", "device_id": did, "message": "Dispositivo registrado exitosamente."})


@app.delete("/api/admin/devices/{device_id}", summary="Eliminar dispositivo de hardware")
async def api_admin_delete_device(device_id: str) -> JSONResponse:
    res = database.delete_hardware_device(device_id)
    if not res:
        raise HTTPException(status_code=404, detail="Dispositivo no encontrado.")
    return JSONResponse({"status": "ok", "message": f"Dispositivo '{device_id}' eliminado."})


class PresetChangeRequest(BaseModel):
    preset_id: str

@app.post("/api/admin/devices/{device_id}/preset", summary="Cambiar preset de hardware en VX600 Pro")
async def api_admin_set_preset(device_id: str, req: PresetChangeRequest) -> JSONResponse:
    device = database.get_hardware_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Dispositivo no encontrado.")
    
    res = novastar_vx600.apply_preset(
        ip=device["ip_address"],
        preset_id=req.preset_id,
        port=device["port"] if device["port"] != 8001 else 6000
    )
    database.update_hardware_device_preset(device_id, req.preset_id)
    logger.info(f"[VX600] Preset '{req.preset_id}' asignado al dispositivo '{device_id}'")
    return JSONResponse({"status": "ok", "device_id": device_id, "applied": res})


@app.get("/api/vx600/presets", summary="Catálogo de presets del procesador NovaStar VX600 Pro")
async def api_vx600_presets() -> JSONResponse:
    return JSONResponse({"status": "ok", "presets": novastar_vx600.get_presets()})


@app.get("/api/vx600/status", summary="Consulta de estado de procesador NovaStar VX600 Pro")
async def api_vx600_status(ip: str = "192.168.1.160", port: int = 6000) -> JSONResponse:
    status = novastar_vx600.get_status(ip, port)
    return JSONResponse(status)


# ─────────────────────────────────────────────────────────────────────────────
# REPORTE EJECUTIVO PDF / IMPRESIÓN (APEX COMPANY)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/report/print", response_class=HTMLResponse, summary="Generador de Reporte Ejecutivo PDF Imprimible")
async def report_print_view(
    client_id: Optional[str] = Query(None, description="ID del cliente"),
    campaign_id: Optional[str] = Query(None, description="ID de la campaña"),
    location_id: Optional[str] = Query("loc_minerva", description="ID de la ubicación")
) -> HTMLResponse:
    """
    Genera un informe ejecutivo imprimible en formato HTML estilizado con @media print
    para exportar directamente a PDF desde el navegador (Ctrl + P / Imprimir a PDF).
    Incluye el logotipo del cliente, sello de ingeniería de Apex Company, aforo vehicular,
    energía consumida (Shelly Pro) y el pie de página de apexcompany.com.mx.
    """
    # 1. Obtener cliente
    client = None
    if client_id and client_id != "all":
        client = database.get_client(client_id)
    if not client:
        clients = database.get_clients()
        client = clients[0] if clients else {
            "id": "cli_default",
            "name": "Cliente Comercial DOOH",
            "logo_url": "/static/uploads/logos/default_cerveceria.svg",
            "contact_email": "contacto@cliente.com"
        }

    # 2. Obtener ubicación y dispositivos
    loc = database.get_location(location_id) or {
        "id": "loc_minerva",
        "name": "Ubicación 01 — Glorieta Minerva",
        "address": "Av. Vallarta y López Mateos, Guadalajara",
        "screen_area_m2": 32.0,
        "cost_per_kwh": 3.85,
        "shelly_ip": "192.168.1.150"
    }
    devices = database.get_hardware_devices(loc["id"])

    # 3. Métricas de tráfico y KPIs
    today = datetime.date.today().strftime("%Y-%m-%d")
    kpis = database.get_kpis(date_str=today)
    hourly_data = database.get_hourly_metrics(date_str=today)
    classes_dict = kpis.get("vehicle_classes", {})
    types_breakdown = [{"type": k, "count": v} for k, v in classes_dict.items()]

    # 4. Telemetría de energía Shelly Pro
    shelly_data = shelly_controller.get_status(loc.get("shelly_ip", "192.168.1.150"))
    
    # Cálculos acumulados
    total_veh = kpis.get("total_flow", 0)
    total_in = kpis.get("total_in", 0)
    total_out = kpis.get("total_out", 0)
    impressions = int(total_veh * 1.6)   # Estimación estándar: 1.6 ocupantes promedio por vehículo
    avg_dwell = kpis.get("avg_dwell_seconds", 8.4)
    peak_hr = kpis.get("peak_hour", "18:00 - 19:00")
    
    # Energía
    power_kw = shelly_data.get("power_kw", 4.2)
    daily_kwh = round(power_kw * 16.5, 2)
    energy_cost = round(daily_kwh * loc.get("cost_per_kwh", 3.85), 2)

    # Generar barras SVG para la gráfica horaria
    chart_bars = ""
    max_count = max([h.get("count", 0) for h in hourly_data], default=100) or 1
    for h in hourly_data:
        hr_str = h.get("hour", "00:00")
        try:
            hr_num = int(hr_str.split(":")[0])
        except Exception:
            hr_num = 0
        cnt = h.get("count", 0)
        height = int((cnt / max_count) * 110)
        y = 120 - height
        x = 35 + (hr_num * 27)
        chart_bars += f"""
        <g>
          <rect x="{x}" y="{y}" width="18" height="{height}" fill="#0284c7" rx="3" opacity="0.85"/>
          <text x="{x + 9}" y="135" font-size="8" fill="#64748b" text-anchor="middle">{hr_num:02d}</text>
        </g>
        """

    # Filas de dispositivos de hardware
    device_rows = ""
    for d in devices:
        dual_text = "Sí (Cara A + Cara B)" if d.get("dual_screen_enabled") else "Pantalla Simple"
        preset_desc = d.get("active_preset", "preset_single_tb40")
        if preset_desc == "preset_mirror_1tb40":
            preset_desc = "Preset 1: 1x TB40 Espejo (Unificado en 2 Caras)"
        elif preset_desc == "preset_dual_2tb40":
            preset_desc = "Preset 2: 2x TB40 Contenido Dual Independiente"
        elif preset_desc == "preset_pip_promo":
            preset_desc = "Preset 3: PIP Multiventana Promocional"

        device_rows += f"""
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 8px 10px; font-weight: bold; color: #1e293b;">{d.get('name')}</td>
          <td style="padding: 8px 10px; color: #475569;"><span style="background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;">{d.get('device_type')}</span></td>
          <td style="padding: 8px 10px; font-family: monospace; color: #334155;">{d.get('ip_address')}:{d.get('port')}</td>
          <td style="padding: 8px 10px; color: #475569;">{dual_text}</td>
          <td style="padding: 8px 10px; font-size: 11px; color: #0284c7; font-weight: 600;">{preset_desc}</td>
        </tr>
        """
    if not device_rows:
        device_rows = """<tr><td colspan="5" style="padding: 12px; text-align: center; color: #94a3b8;">Sin controladores registrados</td></tr>"""

    # Filas de desglose vehicular
    veh_rows = ""
    for v in types_breakdown:
        pct = round((v["count"] / max(1, total_veh)) * 100, 1)
        veh_rows += f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px dashed #cbd5e1; font-size: 12px;">
          <span style="font-weight: 600; color: #1e293b;">{v['type']}</span>
          <span style="color: #475569;"><strong style="color: #0f172a;">{v['count']:,}</strong> veh ({pct}%)</span>
        </div>
        """

    client_logo = client.get("logo_url") or "/static/uploads/logos/default_cerveceria.svg"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Reporte Ejecutivo DOOH — {client.get('name', 'Cliente')} — Apex Company</title>
  <style>
    @page {{
      size: A4 portrait;
      margin: 12mm 15mm 15mm 15mm;
    }}
    * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
    body {{
      background: #ffffff;
      color: #0f172a;
      margin: 0;
      padding: 20px;
      font-size: 13px;
      line-height: 1.4;
    }}
    .no-print {{
      background: #0f172a;
      color: #ffffff;
      padding: 12px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-radius: 8px;
      margin-bottom: 24px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }}
    .btn {{
      background: #0284c7;
      color: #ffffff;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: bold;
      cursor: pointer;
      font-size: 13px;
    }}
    .btn:hover {{ background: #0369a1; }}
    @media print {{
      .no-print {{ display: none !important; }}
      body {{ padding: 0; }}
      .page-footer {{ position: fixed; bottom: 0; left: 0; right: 0; }}
    }}
    .header-table {{
      width: 100%;
      border-bottom: 2px solid #0284c7;
      padding-bottom: 14px;
      margin-bottom: 16px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 18px;
    }}
    .kpi-card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 10px 12px;
      text-align: center;
    }}
    .kpi-card .val {{
      font-size: 20px;
      font-weight: 800;
      color: #0369a1;
      margin-top: 4px;
    }}
    .kpi-card .lbl {{
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      color: #64748b;
      letter-spacing: 0.5px;
    }}
    .section-title {{
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
      color: #0f172a;
      border-left: 4px solid #0284c7;
      padding-left: 8px;
      margin: 16px 0 10px 0;
      letter-spacing: 0.5px;
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
      margin-bottom: 16px;
    }}
    table.data-table th {{
      background: #f1f5f9;
      color: #334155;
      text-align: left;
      padding: 6px 10px;
      font-weight: 700;
      border-bottom: 2px solid #cbd5e1;
    }}
    .footer-watermark {{
      margin-top: 30px;
      border-top: 1px solid #cbd5e1;
      padding-top: 12px;
      text-align: center;
      font-size: 10px;
      color: #64748b;
    }}
    .footer-watermark strong {{
      color: #0284c7;
      font-size: 11px;
    }}
  </style>
</head>
<body>

  <!-- Barra de control en pantalla (Oculta al imprimir) -->
  <div class="no-print">
    <div>
      <strong style="color: #38bdf8; font-size: 14px;">Vista Previa de Reporte Ejecutivo DOOH</strong>
      <span style="color: #94a3b8; font-size: 12px; margin-left: 8px;">Listo para impresión directa o guardar como PDF</span>
    </div>
    <div style="display: flex; gap: 8px;">
      <button onclick="window.print()" class="btn">Imprimir / Guardar PDF</button>
      <button onclick="window.close()" class="btn" style="background: #334155;">Cerrar</button>
    </div>
  </div>

  <!-- Encabezado con Logotipo del Cliente y Marca Apex Company -->
  <table class="header-table">
    <tr>
      <td style="width: 50%; vertical-align: middle;">
        <div style="display: flex; align-items: center; gap: 12px;">
          <img src="{client_logo}" alt="{client.get('name')}" style="max-height: 48px; max-width: 180px; object-fit: contain;" onerror="this.style.display='none'" />
          <div>
            <h1 style="font-size: 17px; margin: 0; color: #0f172a; font-weight: 800;">{client.get('name')}</h1>
            <p style="margin: 2px 0 0 0; font-size: 11px; color: #64748b;">Reporte de Aforo Vehicular, Audiencia e Impactos DOOH</p>
          </div>
        </div>
      </td>
      <td style="width: 50%; text-align: right; vertical-align: middle;">
        <div style="display: inline-block; text-align: right;">
          <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
            <div style="text-align: right;">
              <span style="font-size: 13px; font-weight: 900; color: #0284c7; letter-spacing: 0.5px;">APEX COMPANY</span>
              <div style="font-size: 9px; color: #64748b; font-weight: 600;">apexcompany.com.mx</div>
            </div>
            <div style="width: 28px; height: 28px; background: #0284c7; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 900; font-size: 15px;">A</div>
          </div>
          <div style="margin-top: 4px; font-size: 10px; color: #475569;">
            Fecha: <strong>{today}</strong> | Sitio: <strong>{loc.get('name')}</strong>
          </div>
        </div>
      </td>
    </tr>
  </table>

  <!-- Ficha Técnica -->
  <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 11px;">
    <div><strong>Ubicación:</strong> {loc.get('name')}</div>
    <div><strong>Área de Pantalla:</strong> {loc.get('screen_area_m2')} m² LED Exterior</div>
    <div><strong>Orientación:</strong> {loc.get('screen_orientation_deg', 270)}° Azimut</div>
    <div><strong>Contacto Cliente:</strong> {client.get('contact_email', 'N/A')}</div>
    <div><strong>ID Auditoría:</strong> DOOH-{int(time.time())}</div>
    <div><strong>Controlador:</strong> NovaStar TB40 / VX600 Pro + Shelly Pro</div>
  </div>

  <!-- Métricas Principales (KPIs) -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="lbl">Aforo Vehicular Total</div>
      <div class="val">{total_veh:,}</div>
      <div style="font-size: 10px; color: #64748b; margin-top: 2px;">Cruce validado con IA</div>
    </div>
    <div class="kpi-card">
      <div class="lbl">Impactos Estimados</div>
      <div class="val" style="color: #059669;">{impressions:,}</div>
      <div style="font-size: 10px; color: #64748b; margin-top: 2px;">1.6 ocupantes / veh.</div>
    </div>
    <div class="kpi-card">
      <div class="lbl">Tiempo en Escena</div>
      <div class="val" style="color: #d97706;">{avg_dwell} seg</div>
      <div style="font-size: 10px; color: #64748b; margin-top: 2px;">Visibilidad de pantalla</div>
    </div>
    <div class="kpi-card">
      <div class="lbl">Hora Pico Máxima</div>
      <div class="val" style="font-size: 16px; color: #7c3aed; margin-top: 6px;">{peak_hr}</div>
      <div style="font-size: 10px; color: #64748b; margin-top: 2px;">Mayor concentración</div>
    </div>
  </div>

  <!-- Tabla de Controladores NovaStar y Presets -->
  <div class="section-title">Infraestructura y Controladores de Pantalla (NovaStar & Shelly Pro)</div>
  <table class="data-table">
    <thead>
      <tr>
        <th>Dispositivo / Nombre</th>
        <th>Tipo</th>
        <th>Dirección IP / Puerto</th>
        <th>Modo Pantalla</th>
        <th>Preset Activo de Hardware</th>
      </tr>
    </thead>
    <tbody>
      {device_rows}
    </tbody>
  </table>

  <!-- Gráfica de Tráfico y Desglose por Tipo de Vehículo -->
  <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 16px;">
    <div>
      <div class="section-title">Curva de Flujo Horario de Tráfico (24 Horas)</div>
      <div style="border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; background: #fafafa;">
        <svg viewBox="0 0 700 145" style="width: 100%; height: auto;">
          <line x1="20" y1="120" x2="680" y2="120" stroke="#cbd5e1" stroke-width="1" />
          <line x1="20" y1="65" x2="680" y2="65" stroke="#f1f5f9" stroke-dasharray="4 4" stroke-width="1" />
          {chart_bars}
        </svg>
      </div>
    </div>
    <div>
      <div class="section-title">Composición del Tráfico</div>
      <div style="border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px; background: #fafafa;">
        {veh_rows}
      </div>
    </div>
  </div>

  <!-- Auditoría Energética Shelly Pro -->
  <div class="section-title">Auditoría Energética de Pantalla (Shelly Pro 4PM)</div>
  <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 12px 16px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 11px;">
    <div>
      <span style="color: #166534; font-weight: bold; display: block;">Potencia Activa Actual:</span>
      <span style="font-size: 16px; font-weight: 800; color: #15803d;">{power_kw} kW</span>
    </div>
    <div>
      <span style="color: #166534; font-weight: bold; display: block;">Energía Diaria Consumida:</span>
      <span style="font-size: 16px; font-weight: 800; color: #15803d;">{daily_kwh} kWh</span>
    </div>
    <div>
      <span style="color: #166534; font-weight: bold; display: block;">Costo Eléctrico Estimado:</span>
      <span style="font-size: 16px; font-weight: 800; color: #15803d;">${energy_cost:,.2f} MXN</span>
    </div>
    <div>
      <span style="color: #166534; font-weight: bold; display: block;">Regulación Solar de Brillo:</span>
      <span style="font-size: 13px; font-weight: 800; color: #0284c7;">Calibrado Autónomo</span>
    </div>
  </div>

  <!-- Pie de página Oficial y Obligatorio -->
  <div class="footer-watermark">
    <p style="margin: 0 0 4px 0; font-size: 11px; font-weight: bold; color: #0f172a;">
      Tecnología y Plataforma Desarrollada por <strong>apexcompany.com.mx</strong>
    </p>
    <p style="margin: 0; color: #64748b;">
      Soluciones Avanzadas de Visión Artificial, Analítica Vehicular e Infraestructura Publicitaria DOOH. Todos los derechos reservados.
    </p>
  </div>

</body>
</html>
"""
    return HTMLResponse(content=html)


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
