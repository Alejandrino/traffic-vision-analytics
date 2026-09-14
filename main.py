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
import io
import csv
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pathlib import Path

import config
from camera_stream import CameraStream, create_all_streams
from traffic_analyzer import TrafficAnalyzer, VehicleMetrics
import camera_manager
import database
import novastar_controller
import shelly_controller
import campaign_manager

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
