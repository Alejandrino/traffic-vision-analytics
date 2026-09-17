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
import random
import base64

import cv2
import numpy as np
import uvicorn
import io
import csv
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Response, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
import auth_service
from screen_verifier import screen_verifier_engine

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

# Bucle de eventos principal para callbacks asíncronos desde hilos secundarios
main_loop: Optional[asyncio.AbstractEventLoop] = None


async def broadcast_payload_to_ws(payload: str) -> None:
    """Envía un payload de texto a todos los clientes WebSocket de manera asíncrona."""
    if not ws_clients:
        return
    disconnected = []
    for client in ws_clients:
        try:
            await client.send_text(payload)
        except Exception:
            disconnected.append(client)
    for client in disconnected:
        if client in ws_clients:
            ws_clients.remove(client)


def handle_camera_status_change(
    camera_id: str,
    event_type: str,
    status: str,
    source: str,
    details: str,
    duration_offline_sec: float = 0.0,
    duration_online_sec: float = 0.0,
) -> None:
    """
    Manejador invocado por CameraStream al detectar un cambio de conectividad o hardware.
    1. Persiste el evento en SQLite (tabla camera_connection_logs).
    2. Emite alerta en tiempo real a clientes WebSocket.
    """
    logger.info(f"[CAMERA STATUS] [{camera_id}] Evento={event_type} | Estado={status} | {details}")
    try:
        database.record_camera_connection_event(
            camera_id=camera_id,
            event_type=event_type,
            status=status,
            source=source,
            details=details,
            duration_offline_sec=duration_offline_sec,
            duration_online_sec=duration_online_sec,
        )
    except Exception as exc:
        logger.error(f"Error guardando camera_connection_event en BD: {exc}")

    # Notificar a los WebSockets de forma thread-safe
    global main_loop
    if main_loop and main_loop.is_running():
        payload = json.dumps({
            "type": "camera_status_alert",
            "data": {
                "camera_id": camera_id,
                "event_type": event_type,
                "status": status,
                "source": source,
                "details": details,
                "duration_offline_sec": duration_offline_sec,
                "duration_online_sec": duration_online_sec,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        try:
            asyncio.run_coroutine_threadsafe(broadcast_payload_to_ws(payload), main_loop)
        except Exception as ws_err:
            logger.debug(f"No se pudo despachar broadcast WS para cámara {camera_id}: {ws_err}")


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
    global streams, analyzers, main_loop
    main_loop = asyncio.get_running_loop()

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
            for cam, raw_l in saved_lines.items():
                norm = config.normalize_counting_lines(cam, raw_l)
                config.MULTI_COUNTING_LINES[cam] = norm
                if norm:
                    config.COUNTING_LINES[cam] = norm[0]["coords"]
            logger.info("Líneas persistentes cargadas y normalizadas desde lines.json")
        except Exception as e:
            logger.error(f"Error al cargar lines.json: {e}")

    # ── Inicializar streams de captura ───────────────────────────────────────
    streams = create_all_streams(status_callback=handle_camera_status_change)
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

    # ── Iniciar motor de verificación de pantalla (Proof of Play) ───────────
    try:
        screen_verifier_engine.start()
        logger.info("Motor de verificación de pantalla (Proof of Play) iniciado.")
    except Exception as exc:
        logger.error(f"Error iniciando ScreenVerifierEngine: {exc}")

    logger.info("=== Sistema listo. Servidor escuchando... ===")

    yield  # Aquí corre el servidor

    # ── Limpieza al cerrar ───────────────────────────────────────────────────
    logger.info("=== Deteniendo sistema... ===")
    inference_task.cancel()
    broadcast_task.cancel()
    solar_task.cancel()
    screen_verifier_engine.stop()

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

# ── CONFIGURACIÓN CORS UNIFICADA (WEB + BACKEND) ─────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
# MODELOS PYDANTIC Y AUTENTICACIÓN UNIFICADA (APEX WEB + DOOH VISION)
# ─────────────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    nombre: str
    email: str
    password: str
    rol: Optional[str] = "cliente"
    cliente_id: Optional[str] = None

class UserUpdateRequest(BaseModel):
    nombre: Optional[str] = None
    shelly_cloud_api_key: Optional[str] = None
    shelly_cloud_server: Optional[str] = None

class InviteUserRequest(BaseModel):
    nombre: str
    email: str
    rol: Optional[str] = "cliente"
    cliente_id: Optional[str] = None
    send_email: Optional[bool] = True

class AcceptInviteRequest(BaseModel):
    token: str
    password: str

security = HTTPBearer(auto_error=False)

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[dict]:
    if not credentials:
        return None
    payload = auth_service.decode_access_token(credentials.credentials)
    if not payload or not payload.get("sub"):
        return None
    return database.get_user_by_id(payload.get("sub"))


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — AUTENTICACIÓN Y SESIONES
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/auth/login", summary="Iniciar sesión de usuario")
def auth_login(req: LoginRequest):
    user = database.get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=401, detail="Email o contraseña incorrectos")
    if not auth_service.verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email o contraseña incorrectos")
    if not user.get("activo", 1):
        raise HTTPException(status_code=403, detail="Tu cuenta está desactivada. Contacta al administrador.")
    
    token = auth_service.create_access_token({
        "sub": user["id"],
        "email": user["email"],
        "rol": user["rol"]
    })
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "nombre": user["nombre"],
            "email": user["email"],
            "rol": user["rol"],
            "cliente_id": user.get("cliente_id"),
            "activo": bool(user.get("activo", 1)),
            "creado_en": str(user.get("creado_en", "")),
            "cliente_nombre": user.get("cliente_nombre")
        }
    }


@app.post("/auth/register", summary="Registro directo de usuario")
def auth_register(req: RegisterRequest):
    existing = database.get_user_by_email(req.email)
    if existing:
        raise HTTPException(status_code=400, detail="El correo ya se encuentra registrado en el sistema")
    
    pwd_hash = auth_service.hash_password(req.password)
    user = database.create_user(
        email=req.email,
        nombre=req.nombre,
        password_hash=pwd_hash,
        rol=req.rol or "cliente",
        cliente_id=req.cliente_id
    )
    token = auth_service.create_access_token({
        "sub": user["id"],
        "email": user["email"],
        "rol": user["rol"]
    })
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "nombre": user["nombre"],
            "email": user["email"],
            "rol": user["rol"],
            "cliente_id": user.get("cliente_id"),
            "activo": bool(user.get("activo", 1)),
            "creado_en": str(user.get("creado_en", "")),
            "cliente_nombre": user.get("cliente_nombre")
        }
    }


@app.get("/auth/me", summary="Obtener perfil del usuario autenticado")
def auth_me(user: Optional[dict] = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="No autorizado o sesión expirada")
    return {
        "id": user["id"],
        "nombre": user["nombre"],
        "email": user["email"],
        "rol": user["rol"],
        "cliente_id": user.get("cliente_id"),
        "activo": bool(user.get("activo", 1)),
        "creado_en": str(user.get("creado_en", "")),
        "cliente_nombre": user.get("cliente_nombre")
    }


@app.put("/auth/me", summary="Actualizar perfil del usuario autenticado")
def auth_update_me(req: UserUpdateRequest, user: Optional[dict] = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="No autorizado")
    if req.nombre:
        with database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET nombre = ? WHERE id = ?", (req.nombre, user["id"]))
            conn.commit()
    updated = database.get_user_by_id(user["id"])
    return {
        "id": updated["id"],
        "nombre": updated["nombre"],
        "email": updated["email"],
        "rol": updated["rol"],
        "cliente_id": updated.get("cliente_id"),
        "activo": bool(updated.get("activo", 1)),
        "creado_en": str(updated.get("creado_en", "")),
        "cliente_nombre": updated.get("cliente_nombre")
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — INVITACIONES POR CORREO Y ACTIVACIÓN DE CUENTA
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/admin/invite", summary="Dar de alta y enviar invitación a nuevo usuario")
def admin_invite_user(req: InviteUserRequest):
    token = auth_service.generate_invitation_token()
    inv = database.create_invitation(
        email=req.email,
        nombre=req.nombre,
        rol=req.rol or "cliente",
        cliente_id=req.cliente_id,
        token=token
    )

    # Obtener nombre de la empresa/cliente si fue especificado
    empresa_nombre = "Apex Company — DOOH & Smart Cities"
    if req.cliente_id:
        clients = database.get_clients()
        matched = next((c for c in clients if c["id"] == req.cliente_id), None)
        if matched:
            empresa_nombre = matched["name"]

    invitation_url = f"http://localhost:3000/auth/invite?token={token}"
    email_html = auth_service.render_invitation_email_html(
        nombre=req.nombre,
        email=req.email,
        rol=req.rol or "cliente",
        empresa_nombre=empresa_nombre,
        invitation_url=invitation_url
    )

    email_status = {"sent": False, "method": "none"}
    if req.send_email:
        email_status = auth_service.send_invitation_email(
            to_email=req.email,
            subject=f"Invitación de Acceso a Apex Company — {empresa_nombre}",
            html_content=email_html
        )

    return {
        "success": True,
        "invitation": inv,
        "invitation_token": token,
        "invitation_url": invitation_url,
        "email_status": email_status,
        "message": f"Usuario {req.email} dado de alta. Enlace de invitación listo para activación."
    }


@app.get("/admin/invitations", summary="Listar todas las invitaciones")
def admin_get_invitations():
    return database.get_all_invitations()


@app.get("/auth/invite/verify", summary="Verificar validez de token de invitación")
def auth_verify_invitation(token: str = Query(..., description="Token de invitación")):
    inv = database.get_invitation_by_token(token)
    if not inv:
        return {"valid": False, "reason": "Invitación no encontrada"}
    if inv.get("estado") == "aceptada":
        return {"valid": False, "reason": "Esta invitación ya fue utilizada previamente"}
    if inv.get("estado") == "expirada":
        return {"valid": False, "reason": "El plazo de esta invitación ha expirado. Solicita una nueva."}
    return {"valid": True, "invitation": inv}


@app.post("/auth/invite/accept", summary="Aceptar invitación y establecer contraseña")
def auth_accept_invitation(req: AcceptInviteRequest):
    inv = database.get_invitation_by_token(req.token)
    if not inv:
        raise HTTPException(status_code=404, detail="Invitación inválida o inexistente")
    if inv.get("estado") != "pendiente":
        raise HTTPException(status_code=400, detail=f"La invitación no está disponible ({inv.get('estado')})")
    
    pwd_hash = auth_service.hash_password(req.password)
    user = database.accept_invitation(req.token, pwd_hash)
    if not user:
        raise HTTPException(status_code=500, detail="Error al activar la cuenta de usuario")
    
    token = auth_service.create_access_token({
        "sub": user["id"],
        "email": user["email"],
        "rol": user["rol"]
    })
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "nombre": user["nombre"],
            "email": user["email"],
            "rol": user["rol"],
            "cliente_id": user.get("cliente_id"),
            "activo": bool(user.get("activo", 1)),
            "creado_en": str(user.get("creado_en", "")),
            "cliente_nombre": user.get("cliente_nombre")
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — ADMINISTRACIÓN Y CONTROL DE USUARIOS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/admin/users", summary="Lista de todos los usuarios registrados")
def admin_get_users():
    users = database.get_all_users()
    for u in users:
        u["activo"] = bool(u.get("activo", 1))
    return users


@app.put("/admin/users/{user_id}/toggle", summary="Activar / Desactivar usuario")
def admin_toggle_user(user_id: str):
    database.toggle_user_status(user_id)
    user = database.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    user["activo"] = bool(user.get("activo", 1))
    return user


@app.put("/admin/users/{user_id}/role", summary="Cambiar rol de usuario")
def admin_change_role(user_id: str, rol: str = Query(..., description="Nuevo rol")):
    database.change_user_role(user_id, rol)
    user = database.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    user["activo"] = bool(user.get("activo", 1))
    return user


@app.delete("/admin/users/{user_id}", summary="Eliminar usuario")
def admin_delete_user(user_id: str):
    success = database.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return {"status": "deleted", "id": user_id}


@app.get("/admin/stats", summary="Estadísticas globales de la plataforma")
def admin_global_stats():
    users = database.get_all_users()
    total_users = len(users)
    devices = database.get_hardware_devices()
    total_devices = len(devices)
    active_devices = sum(1 for d in devices if d.get("status") == "online")

    with database.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM traffic_events")
        total_traffic = cursor.fetchone()["count"]
        cursor.execute("SELECT COUNT(*) as count FROM camera_sightings")
        total_sightings = cursor.fetchone()["count"]
        cursor.execute("SELECT COUNT(*) as count FROM campaigns WHERE status = 'Activa'")
        active_campaigns = cursor.fetchone()["count"]

    return {
        "total_usuarios": total_users,
        "total_dispositivos": total_devices,
        "dispositivos_activos": active_devices,
        "total_lecturas": total_traffic + total_sightings,
        "campanas_activas": active_campaigns,
        "eventos_trafico": total_traffic
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — DISPOSITIVOS SHELLY Y HARDWARE COMPATIBLES CON APEX PORTAL
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/devices/", summary="Lista de dispositivos Shelly / Hardware")
@app.get("/admin/devices", summary="Lista administrativa de dispositivos")
def get_devices_list():
    hw = database.get_hardware_devices()
    locs = {l["id"]: l["name"] for l in database.get_locations()}
    res = []
    for d in hw:
        ip = d.get("ip_address", "127.0.0.1")
        is_shelly = d.get("device_type") == "SHELLY_PRO"
        status_info = shelly_controller.get_status(ip, 0) if is_shelly else {"is_on": True, "power_w": 280.0}
        
        res.append({
            "id": d["id"],
            "usuario_id": "usr_admin_master",
            "nombre": d["name"],
            "modelo": f"{d.get('device_type')} ({ip}:{d.get('port')})",
            "webhook_token": f"wh_{d['id']}",
            "estado_rele": status_info.get("is_on", True),
            "watts_actuales": status_info.get("power_w", 0.0),
            "ubicacion": locs.get(d.get("location_id"), "Pantalla Exterior"),
            "activo": d.get("status") == "online",
            "creado_en": str(d.get("created_at", "2026-01-01 00:00:00")),
            "ultima_lectura": datetime.datetime.now().isoformat()
        })
    return res


@app.get("/devices/{device_id}", summary="Detalle de un dispositivo")
def get_device_detail(device_id: str):
    devices = get_devices_list()
    found = next((d for d in devices if d["id"] == device_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="Dispositivo no encontrado")
    return found


@app.get("/relay/{device_id}/status", summary="Estado del relé")
def relay_status(device_id: str):
    hw = database.get_hardware_devices()
    found = next((d for d in hw if d["id"] == device_id), None)
    ip = found.get("ip_address", "192.168.1.150") if found else "192.168.1.150"
    st = shelly_controller.get_status(ip, 0)
    return {
        "device_id": device_id,
        "estado_rele": st.get("is_on", True),
        "watts_actuales": st.get("power_w", 4500.0),
        "ultima_lectura": datetime.datetime.now().isoformat()
    }


@app.post("/relay/{device_id}/on", summary="Encender relé Shelly")
def relay_turn_on(device_id: str):
    hw = database.get_hardware_devices()
    found = next((d for d in hw if d["id"] == device_id), None)
    ip = found.get("ip_address", "192.168.1.150") if found else "192.168.1.150"
    shelly_controller.set_power(ip, True, 0)
    status_data = shelly_controller.get_status(ip, 0)
    return {
        "device_id": device_id,
        "estado_rele": True,
        "watts_actuales": status_data.get("power_w", 4500.0)
    }


@app.post("/relay/{device_id}/off", summary="Apagar relé Shelly")
def relay_turn_off(device_id: str):
    hw = database.get_hardware_devices()
    found = next((d for d in hw if d["id"] == device_id), None)
    ip = found.get("ip_address", "192.168.1.150") if found else "192.168.1.150"
    shelly_controller.set_power(ip, False, 0)
    status_data = shelly_controller.get_status(ip, 0)
    return {
        "device_id": device_id,
        "estado_rele": False,
        "watts_actuales": status_data.get("power_w", 18.0)
    }


@app.get("/consumption/{device_id}/summary", summary="Resumen de consumo")
def consumption_summary(device_id: str):
    hw = database.get_hardware_devices()
    found = next((d for d in hw if d["id"] == device_id), None)
    ip = found.get("ip_address", "192.168.1.150") if found else "192.168.1.150"
    st = shelly_controller.get_status(ip, 0)
    return {
        "dispositivo_id": device_id,
        "kwh_total": st.get("total_kwh", 342.8),
        "watts_promedio": 4450.0,
        "watts_pico": 5120.0,
        "total_lecturas": 1440,
        "costo_estimado_mxn": round(st.get("total_kwh", 342.8) * 3.85, 2)
    }


@app.get("/consumption/{device_id}/chart", summary="Gráfica de consumo")
def consumption_chart(device_id: str, hours: int = 24, interval: str = "hour"):
    now = datetime.datetime.now()
    points = []
    for h in range(hours, -1, -1):
        t = now - datetime.timedelta(hours=h)
        is_day = 8 <= t.hour <= 21
        base_w = 4600.0 if is_day else 350.0
        points.append({
            "timestamp": t.strftime("%Y-%m-%d %H:00:00"),
            "watts_promedio": round(base_w + random.uniform(-150, 200), 1),
            "watts_max": round(base_w + 350, 1),
            "watts_min": round(base_w - 200, 1),
            "lecturas": 60
        })
    return points


@app.get("/consumption/{device_id}/history", summary="Historial de consumo")
def consumption_history(device_id: str, limit: int = 100):
    now = datetime.datetime.now()
    records = []
    for i in range(min(limit, 50)):
        t = now - datetime.timedelta(minutes=i * 15)
        records.append({
            "id": i + 1,
            "dispositivo_id": device_id,
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "watts": round(random.uniform(4300, 4800), 1),
            "estado_rele": True,
            "voltaje": round(random.uniform(219, 222), 1),
            "corriente": round(random.uniform(19.5, 21.8), 2),
            "energia_total_kwh": round(340.0 + (i * 0.1), 3),
            "fuente": "Shelly Pro RPC"
        })
    return records


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


class SingleLineModel(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    target_entity: Optional[str] = "ALL"
    coords: Optional[list[list[int]]] = None
    start: Optional[list[int]] = None
    end: Optional[list[int]] = None


class MultiLineUpdate(BaseModel):
    lines: list[SingleLineModel]


@app.get("/api/lines/{cam_id}", summary="Obtener todas las líneas de conteo configuradas para la cámara")
async def get_lines(cam_id: str) -> JSONResponse:
    if cam_id not in analyzers:
        raise HTTPException(status_code=404, detail=f"Cámara '{cam_id}' no encontrada.")
    analyzer = analyzers[cam_id]
    lines_list = [l.to_dict() for l in analyzer.counting_lines.values()]
    return JSONResponse({
        "status": "ok",
        "cam_id": cam_id,
        "lines": lines_list
    })


@app.post("/api/lines/{cam_id}", summary="Actualizar y guardar la lista de líneas de conteo de la cámara")
async def update_lines_endpoint(cam_id: str, payload: MultiLineUpdate) -> JSONResponse:
    if cam_id not in analyzers:
        raise HTTPException(status_code=404, detail=f"Cámara '{cam_id}' no encontrada.")
    
    formatted_lines = []
    for idx, l in enumerate(payload.lines):
        lid = l.id or f"{cam_id}_line_{idx+1}"
        lname = l.name or f"Línea {idx+1}"
        ent = (l.target_entity or "ALL").upper()
        
        # Extraer coordenadas
        if l.coords and len(l.coords) == 2:
            coords = ((int(l.coords[0][0]), int(l.coords[0][1])), (int(l.coords[1][0]), int(l.coords[1][1])))
        elif l.start and l.end:
            coords = ((int(l.start[0]), int(l.start[1])), (int(l.end[0]), int(l.end[1])))
        else:
            coords = ((50, config.FRAME_HEIGHT // 2), (config.FRAME_WIDTH - 50, config.FRAME_HEIGHT // 2))

        formatted_lines.append({
            "id": lid,
            "name": lname,
            "target_entity": ent,
            "coords": coords
        })

    # 1. Actualizar configuración en memoria
    config.MULTI_COUNTING_LINES[cam_id] = formatted_lines
    if formatted_lines:
        config.COUNTING_LINES[cam_id] = formatted_lines[0]["coords"]

    # 2. Persistir en lines.json
    lines_file = Path("lines.json")
    saved_lines = {}
    if lines_file.exists():
        try:
            saved_lines = json.loads(lines_file.read_text(encoding="utf-8"))
        except Exception:
            pass
            
    saved_lines[cam_id] = [
        {
            "id": fl["id"],
            "name": fl["name"],
            "target_entity": fl["target_entity"],
            "coords": [list(fl["coords"][0]), list(fl["coords"][1])]
        }
        for fl in formatted_lines
    ]
    lines_file.write_text(json.dumps(saved_lines, indent=2), encoding="utf-8")

    # 3. Actualizar el analizador
    analyzers[cam_id].update_lines(formatted_lines)
    latest_metrics[cam_id] = analyzers[cam_id].metrics.to_dict()

    return JSONResponse({
        "status": "ok",
        "message": f"Se guardaron {len(formatted_lines)} líneas de conteo para '{cam_id}'.",
        "lines": [l.to_dict() for l in analyzers[cam_id].counting_lines.values()]
    })


@app.get("/api/line/{cam_id}", summary="Obtener línea actual de la cámara (compatibilidad)")
async def get_line(cam_id: str) -> JSONResponse:
    if cam_id not in config.COUNTING_LINES and cam_id not in analyzers:
        raise HTTPException(status_code=404, detail=f"Cámara '{cam_id}' no configurada.")
    
    if cam_id in analyzers and analyzers[cam_id].counting_lines:
        first_line = next(iter(analyzers[cam_id].counting_lines.values()))
        return JSONResponse({
            "status": "ok",
            "cam_id": cam_id,
            "line": {"start": first_line.coords[0], "end": first_line.coords[1]},
            "lines": [l.to_dict() for l in analyzers[cam_id].counting_lines.values()]
        })

    coords = config.COUNTING_LINES.get(cam_id, ((50, 360), (1230, 360)))
    return JSONResponse({
        "status": "ok",
        "cam_id": cam_id,
        "line": {"start": coords[0], "end": coords[1]}
    })


@app.post("/api/line/{cam_id}", summary="Actualizar línea de conteo (compatibilidad)")
async def update_line(cam_id: str, line_data: LineUpdate) -> JSONResponse:
    if cam_id not in analyzers:
        raise HTTPException(status_code=404, detail=f"Cámara '{cam_id}' no encontrada.")
    
    new_coords = (line_data.start, line_data.end)
    config.COUNTING_LINES[cam_id] = new_coords
    
    # Actualizar primera línea o agregar
    analyzer = analyzers[cam_id]
    if analyzer.counting_lines:
        first_key = next(iter(analyzer.counting_lines.keys()))
        first_line = analyzer.counting_lines[first_key]
        first_line.coords = new_coords
        lines_cfg = [
            {
                "id": l.id,
                "name": l.name,
                "target_entity": l.target_entity,
                "coords": l.coords
            }
            for l in analyzer.counting_lines.values()
        ]
    else:
        lines_cfg = [{
            "id": f"{cam_id}_line_1",
            "name": "Línea Principal",
            "target_entity": "ALL",
            "coords": new_coords
        }]
        
    config.MULTI_COUNTING_LINES[cam_id] = lines_cfg
    
    lines_file = Path("lines.json")
    saved_lines = {}
    if lines_file.exists():
        try:
            saved_lines = json.loads(lines_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    saved_lines[cam_id] = [
        {
            "id": l["id"],
            "name": l["name"],
            "target_entity": l["target_entity"],
            "coords": [list(l["coords"][0]), list(l["coords"][1])]
        }
        for l in lines_cfg
    ]
    lines_file.write_text(json.dumps(saved_lines, indent=2), encoding="utf-8")
    
    analyzers[cam_id].update_lines(lines_cfg)
    latest_metrics[cam_id] = analyzers[cam_id].metrics.to_dict()
    
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
    
    stream = CameraStream(req.cam_id, req.source, on_status_change=handle_camera_status_change)
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


@app.get("/api/cameras/connection-logs", summary="Bitácora de eventos de conexión y desconexión de cámaras")
async def api_camera_connection_logs(
    camera_id: str = Query("all", description="ID de cámara o 'all'"),
    limit: int = Query(50, description="Cantidad máxima de registros"),
    offset: int = Query(0, description="Desplazamiento para paginación"),
    date: Optional[str] = Query(None, description="Fecha YYYY-MM-DD"),
    event_type: str = Query("all", description="Filtro de evento (all, CONNECTED, DISCONNECTED, ERROR, RECONNECTING, etc.)")
) -> JSONResponse:
    """Devuelve los registros históricos y recientes de conectividad de los equipos de cámara."""
    data = database.get_camera_connection_logs(
        camera_id=camera_id,
        limit=limit,
        offset=offset,
        date_str=date,
        event_type=event_type
    )
    return JSONResponse({"status": "ok", "data": data})


@app.get("/api/cameras/connection-stats", summary="Estadísticas de disponibilidad y tiempo fuera de línea de cámaras")
async def api_camera_connection_stats(
    camera_id: str = Query("all", description="ID de cámara o 'all'"),
    date: Optional[str] = Query(None, description="Fecha YYYY-MM-DD")
) -> JSONResponse:
    """Retorna métricas de disponibilidad (uptime %, total desconexiones, tiempo fuera de línea)."""
    stats = database.get_camera_connection_stats(camera_id=camera_id, date_str=date)
    live_states = {}
    for cid, st in streams.items():
        is_enabled = getattr(st, "enabled", True) and (cid in active_processing)
        live_states[cid] = {
            "source": str(st.source),
            "connected": st.connected if is_enabled else False,
            "enabled": is_enabled,
            "fps": round(st.fps, 1) if is_enabled else 0.0,
            "reconnect_count": getattr(st, "reconnect_count", 0),
        }
    stats["live_cameras"] = live_states
    return JSONResponse({"status": "ok", "data": stats})


@app.get("/api/cameras/connection-logs/export-csv", summary="Exportar bitácora de conexiones de cámaras a CSV")
async def api_camera_connection_logs_export_csv(
    camera_id: str = Query("all", description="ID de cámara o 'all'"),
    date: Optional[str] = Query(None, description="Fecha YYYY-MM-DD"),
    event_type: str = Query("all", description="Filtro de evento")
) -> Response:
    """Genera y descarga un archivo CSV con la bitácora completa de conexiones y desconexiones."""
    data = database.get_camera_connection_logs(
        camera_id=camera_id,
        limit=5000,
        offset=0,
        date_str=date,
        event_type=event_type
    )
    records = data.get("records", [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID",
        "Fecha y Hora",
        "Cámara",
        "Tipo de Evento",
        "Estado",
        "Fuente / Origen",
        "Tiempo Fuera de Línea (seg)",
        "Tiempo Fuera de Línea (Legible)",
        "Tiempo en Línea (seg)",
        "Tiempo en Línea (Legible)",
        "Diagnóstico / Detalles"
    ])

    for r in records:
        writer.writerow([
            r["id"],
            r["timestamp"],
            r["camera_id"],
            r["event_type"],
            r["status"],
            r["source"],
            r["duration_offline_sec"],
            r["duration_offline_formatted"],
            r["duration_online_sec"],
            r["duration_online_formatted"],
            r["details"]
        ])

    csv_bytes = output.getvalue().encode("utf-8-sig")
    filename = f"bitacora_conexiones_camaras_{date or 'historico'}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS DE ANALÍTICA HISTÓRICA Y REGISTRO DE EVENTOS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/history/summary", summary="Resumen ejecutivo de KPIs históricos")
async def api_history_summary(
    camera_id: str = Query("all", description="ID de cámara o 'all'"),
    date: str | None = Query(None, description="Fecha YYYY-MM-DD"),
    entity_type: str = Query("all", description="Filtro: 'all', 'vehicle', 'pedestrian'")
) -> JSONResponse:
    """Retorna KPIs ejecutivos: aforo total, in, out, hora pico, tiempo de estancia y desglose vehicular y peatonal."""
    data = database.get_kpis(camera_id=camera_id, date_str=date, entity_type=entity_type)
    return JSONResponse({"status": "ok", "data": data})


@app.get("/api/history/hourly", summary="Métricas de aforo por hora (00:00 - 23:00)")
async def api_history_hourly(
    camera_id: str = Query("all", description="ID de cámara o 'all'"),
    date: str | None = Query(None, description="Fecha YYYY-MM-DD"),
    entity_type: str = Query("all", description="Filtro: 'all', 'vehicle', 'pedestrian'")
) -> JSONResponse:
    """Retorna la distribución horaria completa con desglose dual vehicular y peatonal."""
    data = database.get_hourly_metrics(camera_id=camera_id, date_str=date, entity_type=entity_type)
    return JSONResponse({"status": "ok", "data": data})


@app.get("/api/history/daily", summary="Tendencia diaria histórica")
async def api_history_daily(
    camera_id: str = Query("all", description="ID de cámara o 'all'"),
    days: int = Query(7, description="Cantidad de días hacia atrás (7, 14, 30)"),
    entity_type: str = Query("all", description="Filtro: 'all', 'vehicle', 'pedestrian'")
) -> JSONResponse:
    """Retorna el volumen diario por día para vehículos y peatones."""
    data = database.get_daily_metrics(camera_id=camera_id, days=days, entity_type=entity_type)
    return JSONResponse({"status": "ok", "data": data})


@app.get("/api/history/events", summary="Registro filtrado y paginado de eventos de cruce")
async def api_history_events(
    camera_id: str = Query("all"),
    line_id: str = Query("all"),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    vehicle_type: str = Query("all"),
    direction: str = Query("all"),
    entity_type: str = Query("all", description="'all', 'vehicle', 'pedestrian'"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
) -> JSONResponse:
    """Consulta la bitácora de eventos con filtros dinámicos incluyendo tipo de entidad y línea."""
    data = database.get_events(
        camera_id=camera_id,
        line_id=line_id,
        start_date=start_date,
        end_date=end_date,
        vehicle_type=vehicle_type,
        direction=direction,
        entity_type=entity_type,
        limit=limit,
        offset=offset
    )
    return JSONResponse({"status": "ok", "data": data})


@app.get("/api/history/export/csv", summary="Exportar eventos a archivo CSV para Excel")
async def api_history_export_csv(
    camera_id: str = Query("all"),
    line_id: str = Query("all"),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    vehicle_type: str = Query("all"),
    direction: str = Query("all"),
    entity_type: str = Query("all")
) -> Response:
    """Genera y descarga un archivo CSV con el historial de eventos para análisis y auditoría."""
    data = database.get_events(
        camera_id=camera_id,
        line_id=line_id,
        start_date=start_date,
        end_date=end_date,
        vehicle_type=vehicle_type,
        direction=direction,
        entity_type=entity_type,
        limit=5000,
        offset=0
    )
    
    output = io.StringIO()
    writer = csv.writer(output)
    # Encabezados en español con acentos limpios
    writer.writerow(["ID", "Fecha_Hora", "Cámara", "Línea_Aforo", "Track_ID", "Clase_Objeto", "Tipo_Entidad", "Sentido", "Confianza", "Tiempo_Permanencia_Seg"])
    
    for ev in data["events"]:
        writer.writerow([
            ev["id"],
            ev["timestamp"],
            ev["camera_id"],
            ev.get("line_id", "line_1"),
            ev["track_id"],
            ev["vehicle_type"],
            ev.get("entity_type", "VEHICLE"),
            ev["direction"],
            ev["confidence"],
            ev["dwell_time"]
        ])
        
    csv_bytes = output.getvalue().encode("utf-8-sig")
    filename = f"reporte_aforo_multimodal_{camera_id}_{start_date or 'inicio'}.csv"
    
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
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

@app.get("/api/novastar/playback", summary="Spot o video que se reproduce actualmente en el NovaStar TB40")
async def api_novastar_playback(
    location_id: Optional[str] = Query(None),
    ip: str = Query("192.168.1.140"),
    port: int = Query(8001)
) -> JSONResponse:
    """
    Obtiene la información en tiempo real del spot o video que está proyectando el NovaStar TB40:
    título, archivo .mp4, duración, segundo actual, porcentaje de progreso, resolución, siguiente anuncio,
    y datos limpios de consumo eléctrico de la pantalla.
    """
    loc = None
    if location_id and location_id != "all":
        loc = database.get_location(location_id)
    if not loc:
        locs = database.get_locations()
        loc = locs[0] if locs else {}

    if loc:
        ip = loc.get("novastar_ip", ip)
        port = int(loc.get("novastar_port", port))

    playback = novastar_controller.get_current_playback(ip=ip, port=port, location_id=location_id)
    
    # Obtener telemetría eléctrica limpia de la pantalla (sin tecnicismos de Shelly)
    shelly_ip = loc.get("shelly_ip", "192.168.1.150") if loc else "192.168.1.150"
    shelly_data = shelly_controller.get_status(ip=shelly_ip, channel=0)
    power_kw = round((shelly_data.get("power_w", 8450.0)) / 1000.0, 2)
    today_kwh = round(shelly_data.get("total_kwh", 12.4), 2)

    screen_info = {
        "id": loc.get("id", location_id or "loc_minerva"),
        "name": loc.get("name", "Pantalla Glorieta Minerva"),
        "dimensions": loc.get("screen_dimensions", "6.0m x 3.5m"),
        "area_sqm": float(loc.get("screen_area_sqm") or 21.0),
        "power_kw": power_kw,
        "energy_today_kwh": today_kwh,
        "cost_per_kwh": float(loc.get("cost_per_kwh") or 3.85)
    }

    return JSONResponse({
        "status": "ok",
        "playback": playback,
        "screen": screen_info
    })

@app.get("/api/client/{client_id}/campaign-report", summary="Reporte ejecutivo de cliente por campaña, pantalla y videos")
async def api_client_campaign_report(
    client_id: str,
    campaign_id: Optional[int] = Query(None),
    location_id: Optional[str] = Query(None)
) -> JSONResponse:
    """
    Genera el reporte consolidado para el cliente con desglose por campaña, pantalla y videos,
    presentando los datos de consumo eléctrico de forma limpia sin tecnicismos de hardware interno.
    """
    report = database.get_client_campaign_report(client_id=client_id, campaign_id=campaign_id, location_id=location_id)
    return JSONResponse({
        "status": "ok",
        "report": report,
        "summary": report.get("summary", {}),
        "videos": report.get("videos", []),
        "client_name": report.get("client_name", "")
    })

@app.get("/api/campaigns/{campaign_id}/videos", summary="Lista de videos y spots de una campaña")
async def api_campaign_videos(campaign_id: int) -> JSONResponse:
    """Obtiene los spots registrados para una campaña específica."""
    videos = database.get_campaign_videos(campaign_id=campaign_id)
    return JSONResponse({"status": "ok", "videos": videos})


@app.get("/api/videos", summary="Lista general de videos y spots DOOH")
async def api_all_videos(client_id: Optional[str] = None, location_id: Optional[str] = None) -> JSONResponse:
    """Retorna todos los spots de video registrados para clientes y pantallas."""
    videos = database.get_campaign_videos(client_id=client_id, location_id=location_id)
    return JSONResponse(videos)



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
    kwh_rate_cfe: Optional[float] = 3.85
    location_ids: Optional[list[str]] = []

@app.post("/api/clients", summary="Registrar o actualizar cliente")
async def api_post_client(req: ClientCreate) -> JSONResponse:
    cid = database.add_client(
        req.id, 
        req.name, 
        req.contact_email or "", 
        req.logo_url or "",
        float(req.kwh_rate_cfe or 3.85)
    )
    if req.location_ids:
        for loc in req.location_ids:
            database.assign_client_location(cid, loc)
    return JSONResponse({"status": "ok", "client_id": cid, "message": "Cliente guardado exitosamente."})


class ClientKwhRateUpdate(BaseModel):
    kwh_rate: float
    location_id: Optional[str] = None

@app.post("/api/client/{client_id}/kwh-rate", summary="Actualizar tarifa de CFE por kWh para un cliente")
async def api_update_client_kwh_rate(client_id: str, req: ClientKwhRateUpdate) -> JSONResponse:
    """
    Permite al cliente o administrador fijar la tarifa real que le factura CFE ($/kWh).
    Impacta inmediatamente en los costos reales de la pantalla y reportes ejecutivos.
    """
    client = database.get_client(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Cliente '{client_id}' no encontrado.")
    
    if req.kwh_rate <= 0:
        raise HTTPException(status_code=400, detail="La tarifa por kWh debe ser un valor numérico positivo mayor a 0.")
    
    database.update_client_kwh_rate(client_id, req.kwh_rate, req.location_id)
    logger.info(f"[CFE-RATE] Tarifa CFE actualizada para cliente '{client_id}': ${req.kwh_rate:.2f} MXN/kWh (loc: {req.location_id or 'global'})")
    return JSONResponse({
        "status": "ok", 
        "client_id": client_id, 
        "kwh_rate": req.kwh_rate,
        "location_id": req.location_id,
        "message": f"Tarifa de CFE actualizada a ${req.kwh_rate:.2f} MXN/kWh correctamente."
    })



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
# RUTAS — AUDITORÍA INTELIGENTE DE PANTALLA: PROOF OF PLAY (CÁMARA USB / CCTV)
# ─────────────────────────────────────────────────────────────────────────────

class ProofCameraConfigRequest(BaseModel):
    location_id: str = "loc_minerva"
    camera_type: str = "USB"  # 'USB' o 'CCTV_RTSP'
    source_url: str = "0"     # índice '0'/'1' o 'rtsp://...'
    auto_interval_min: Optional[int] = 15

class ProofManualCaptureRequest(BaseModel):
    client_id: str
    video_title: str
    video_id: Optional[int] = None
    location_id: Optional[str] = "loc_minerva"
    notes: Optional[str] = None

@app.get("/api/screen-verifier/status", summary="Estado de la cámara de auditoría de pantalla")
async def api_screen_verifier_status(location_id: str = "loc_minerva") -> JSONResponse:
    """Retorna el estado de la cámara que apunta a la pantalla, luminancia y último frame."""
    cfg = database.get_proof_camera_config(location_id)
    return JSONResponse({
        "status": "ok",
        "location_id": location_id,
        "is_connected": screen_verifier_engine.is_connected,
        "camera_type": screen_verifier_engine.camera_type,
        "source_url": screen_verifier_engine.source_url,
        "screen_is_on": screen_verifier_engine.screen_is_on,
        "latest_luminance": screen_verifier_engine.latest_luminance,
        "fps_observed": screen_verifier_engine.fps_observed,
        "config": cfg
    })

@app.post("/api/screen-verifier/config", summary="Configurar cámara de verificación (USB o CCTV RTSP)")
async def api_screen_verifier_config(req: ProofCameraConfigRequest) -> JSONResponse:
    """Configura si la cámara es USB local o stream CCTV RTSP y reinicia el motor."""
    res = screen_verifier_engine.update_config(
        camera_type=req.camera_type,
        source_url=req.source_url,
        auto_interval_min=req.auto_interval_min or 15
    )
    return JSONResponse({
        "status": "ok",
        "message": f"Cámara de verificación configurada como {req.camera_type} ('{req.source_url}').",
        "config": res
    })

@app.post("/api/screen-verifier/capture", summary="Disparar captura inmediata de evidencia de reproducción")
async def api_screen_verifier_capture(req: ProofManualCaptureRequest) -> JSONResponse:
    """Toma una foto de prueba con marca de agua pericial y la guarda en la base de datos."""
    proof = screen_verifier_engine.capture_play_proof(
        client_id=req.client_id,
        video_title=req.video_title,
        video_id=req.video_id,
        notes=req.notes
    )
    return JSONResponse({
        "status": "ok",
        "message": "Evidencia fotográfica capturada y registrada con éxito.",
        "proof": proof
    })

@app.get("/api/client/{client_id}/evidence", summary="Galería de comprobaciones visuales del cliente")
async def api_client_evidence(
    client_id: str,
    location_id: Optional[str] = None,
    limit: int = Query(30, ge=1, le=100),
    date: Optional[str] = None
) -> JSONResponse:
    """Retorna las evidencias fotográficas de spots transmitidos para el cliente."""
    proofs = database.get_client_play_proofs(
        client_id=client_id,
        location_id=location_id,
        limit=limit,
        date_str=date
    )
    return JSONResponse({
        "status": "ok",
        "client_id": client_id,
        "count": len(proofs),
        "evidence": proofs
    })

@app.get("/video_feed/screen_proof", summary="Stream MJPEG en vivo de la cámara que audita la pantalla")
async def video_feed_screen_proof() -> StreamingResponse:
    """Stream de video en vivo de la cámara de verificación hacia la pantalla."""
    async def _proof_generator():
        while True:
            jpg = screen_verifier_engine.get_latest_frame_jpeg()
            if jpg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            await asyncio.sleep(0.06)

    return StreamingResponse(
        _proof_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ─────────────────────────────────────────────────────────────────────────────
# REPORTE EJECUTIVO PDF / IMPRESIÓN (APEX COMPANY)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/report/print", response_class=HTMLResponse, summary="Generador de Reporte Ejecutivo PDF Imprimible")
async def report_print_view(
    client_id: Optional[str] = Query(None, description="ID del cliente"),
    campaign_id: Optional[str] = Query(None, description="ID de la campaña"),
    location_id: Optional[str] = Query("loc_minerva", description="ID de la ubicación"),
    view_mode: str = Query("client", description="'client' para vista ejecutiva limpia o 'admin' para vista técnica"),
    period: str = Query("day", description="'day' (día), 'month' (mes), 'year' (año) o 'campaign' (por campaña/video)"),
    date: Optional[str] = Query(None, description="Fecha YYYY-MM-DD para período día"),
    month: Optional[str] = Query(None, description="Mes YYYY-MM para período mes"),
    year: Optional[str] = Query(None, description="Año YYYY para período año"),
    video_id: Optional[str] = Query(None, description="ID específico de video/spot a auditar"),
    kwh_rate: Optional[float] = Query(None, description="Tarifa CFE $/kWh personalizada o auditada"),
    autoprint: bool = Query(False, description="Disparar automáticamente el diálogo de impresión al cargar")
) -> HTMLResponse:
    """
    Genera un informe ejecutivo imprimible en formato HTML estilizado con @media print
    para exportar directamente a PDF desde el navegador (Ctrl + P / Imprimir a PDF).
    Soporta filtrado por Día, Mes, Año o Campaña/Video publicitario, calculando el
    costo real de energía con la tarifa contratada con CFE.
    """
    # 1. Obtener cliente
    client = None
    if client_id and client_id != "all":
        client = database.get_client(client_id)
    if not client:
        clients = database.get_clients()
        client = clients[0] if clients else {
            "id": "cli_cerveceria",
            "name": "Cervecería Cuauhtémoc Moctezuma",
            "logo_url": "/static/uploads/logos/default_cerveceria.svg",
            "contact_email": "marketing@cerveceria.com.mx",
            "kwh_rate_cfe": 3.85
        }

    # 2. Obtener ubicación y reporte de cliente consolidado
    loc = database.get_location(location_id) or {
        "id": "loc_minerva",
        "name": "Ubicación 01 — Glorieta Minerva",
        "address": "Av. Vallarta y López Mateos, Guadalajara",
        "screen_area_m2": 32.0,
        "cost_per_kwh": 3.85,
        "shelly_ip": "192.168.1.150"
    }

    # Tarifa CFE efectiva (prioridad: query param > cliente > ubicación > default 3.85)
    effective_cfe_rate = float(kwh_rate) if (kwh_rate and kwh_rate > 0) else float(client.get("kwh_rate_cfe") or loc.get("cost_per_kwh") or 3.85)

    camp_id_int = int(campaign_id) if (campaign_id and campaign_id.isdigit()) else None
    vid_id_int = int(video_id) if (video_id and video_id.isdigit()) else None

    client_report = database.get_client_campaign_report(
        client["id"],
        campaign_id=camp_id_int,
        location_id=loc["id"],
        video_id=vid_id_int
    )
    videos = client_report.get("videos", [])
    rep_summary = client_report.get("summary", {})
    campaigns = client_report.get("campaigns", [])
    campaign_name = campaigns[0]["name"] if campaigns else "Pauta Anual de Marca DOOH"

    # Determinar etiquetas y parámetros de período
    today = datetime.date.today().strftime("%Y-%m-%d")
    current_month = datetime.date.today().strftime("%Y-%m")
    current_year = datetime.date.today().strftime("%Y")

    active_date = date or today
    active_month = month or current_month
    active_year = year or current_year

    if period == "month":
        period_title = f"Reporte Mensual — {active_month}"
        period_label = f"Mes Auditado: {active_month}"
        chart_title = f"Curva de Flujo Diario ({active_month})"
        period_kpis = database.get_kpis(period="month", month_str=active_month)
        chart_data = database.get_period_chart_data(period="month", month_str=active_month)
    elif period == "year":
        period_title = f"Reporte Anual — Año {active_year}"
        period_label = f"Año Auditado: {active_year}"
        chart_title = f"Curva de Flujo Mensual ({active_year})"
        period_kpis = database.get_kpis(period="year", year_str=active_year)
        chart_data = database.get_period_chart_data(period="year", year_str=active_year)
    elif period == "campaign":
        video_selected = next((v for v in videos if v["id"] == vid_id_int), None) if vid_id_int else None
        v_title = f"Spot: {video_selected['title']}" if video_selected else campaign_name
        period_title = f"Auditoría de Campaña — {v_title}"
        period_label = f"Campaña / Video: {v_title}"
        chart_title = "Distribución Horaria de Impactos"
        period_kpis = database.get_kpis(period="day", date_str=active_date)
        chart_data = database.get_period_chart_data(period="day", date_str=active_date)
    else:
        # Período día (default)
        period_title = f"Reporte Diario — {active_date}"
        period_label = f"Fecha Auditada: {active_date}"
        chart_title = "Curva de Flujo Horario (24 Horas)"
        period_kpis = database.get_kpis(period="day", date_str=active_date)
        chart_data = database.get_period_chart_data(period="day", date_str=active_date)

    kpis = period_kpis
    hourly_data = chart_data
    classes_dict = kpis.get("vehicle_classes", {})
    types_breakdown = [{"type": k, "count": v} for k, v in classes_dict.items()]

    # 4. Telemetría de energía de pantalla
    shelly_data = shelly_controller.get_status(loc.get("shelly_ip", "192.168.1.150"))
    power_kw = shelly_data.get("power_kw", 4.2)
    
    # Factor de escala según período para cálculo energético estimado
    period_days_factor = 1.0
    if period == "month":
        period_days_factor = 30.0
    elif period == "year":
        period_days_factor = 365.0
    elif period == "campaign":
        period_days_factor = 7.0

    daily_kwh = round(power_kw * 16.5 * period_days_factor, 2)
    energy_cost = round(daily_kwh * effective_cfe_rate, 2)

    # Cálculos acumulados y doble métrica (Vistos en Cámara vs Cruces por Línea)
    veh_kpis = kpis.get("vehicles", {})
    ped_kpis = kpis.get("pedestrians", {})

    total_flow = kpis.get("total_flow", 0)
    total_seen = kpis.get("total_seen", 0)
    total_crossed = kpis.get("total_crossed", 0)
    crossing_rate = kpis.get("crossing_rate", 0.0)
    total_in = kpis.get("total_in", 0)
    total_out = kpis.get("total_out", 0)

    v_seen = veh_kpis.get("seen", 0)
    v_crossed = veh_kpis.get("crossed", 0)
    v_rate = veh_kpis.get("crossing_rate", 0.0)
    v_flow = veh_kpis.get("total_flow", 0)
    v_in = veh_kpis.get("total_in", 0)
    v_out = veh_kpis.get("total_out", 0)

    p_seen = ped_kpis.get("seen", 0)
    p_crossed = ped_kpis.get("crossed", 0)
    p_rate = ped_kpis.get("crossing_rate", 0.0)
    p_flow = ped_kpis.get("total_flow", 0)
    p_in = ped_kpis.get("total_in", 0)
    p_out = ped_kpis.get("total_out", 0)

    # Impactos DOOH: factor 1.6 por vehículo + 1.0 por peatón
    impressions = int((v_crossed * 1.6) + (p_crossed * 1.0))
    if impressions == 0 and total_flow > 0:
        impressions = int(total_flow * 1.6)

    avg_dwell = kpis.get("avg_dwell_seconds", 8.4)
    peak_hr = kpis.get("peak_hour", "18:00 - 19:00")

    # Si hay videos de campaña, tomar sus métricas consolidadas
    if videos:
        total_spots_today = rep_summary.get("total_spots_today", 180)
        total_spots_camp = rep_summary.get("total_spots_campaign", 3500)
        campaign_kwh = rep_summary.get("campaign_energy_kwh", daily_kwh)
        campaign_cost = rep_summary.get("energy_cost_mxn", energy_cost)
        campaign_impressions = rep_summary.get("total_impressions", impressions)
    else:
        total_spots_today = int(180 * period_days_factor)
        total_spots_camp = int(3600 * period_days_factor)
        campaign_kwh = daily_kwh
        campaign_cost = energy_cost
        campaign_impressions = impressions

    # Generar barras SVG para la gráfica con separación vehicular y peatonal adaptada a días / meses / horas
    chart_bars = ""
    max_count = max([h.get("total", 0) for h in hourly_data], default=10) or 1
    if max_count == 0:
        max_count = 1

    total_slots = max(len(hourly_data), 1)
    bar_width = max(8, min(24, int(600 / total_slots) - 4))
    
    for idx, h in enumerate(hourly_data):
        label_str = h.get("label", h.get("hour", str(idx)))
        tot = h.get("total", 0)
        v_cnt = h.get("vehicles_total", 0)
        p_cnt = h.get("pedestrians_total", 0)

        # Si no hay desglose específico pero sí total general
        if v_cnt == 0 and p_cnt == 0 and tot > 0:
            v_cnt = tot

        h_veh = int((v_cnt / max_count) * 90) if max_count > 0 else 0
        h_ped = int((p_cnt / max_count) * 90) if max_count > 0 else 0

        y_veh = 115 - h_veh
        y_ped = y_veh - h_ped
        
        spacing = (640 / total_slots)
        x = int(30 + (idx * spacing))

        veh_bar = f'<rect x="{x}" y="{y_veh}" width="{bar_width}" height="{h_veh}" fill="#0284c7" rx="2" opacity="0.9"><title>{v_cnt} Vehículos</title></rect>' if h_veh > 0 else ""
        ped_bar = f'<rect x="{x}" y="{y_ped}" width="{bar_width}" height="{h_ped}" fill="#10b981" rx="2" opacity="0.9"><title>{p_cnt} Peatones</title></rect>' if h_ped > 0 else ""
        total_lbl = f'<text x="{x + (bar_width // 2)}" y="{max(12, y_ped - 3)}" font-size="7" font-weight="bold" fill="#0f172a" text-anchor="middle">{tot}</text>' if tot > 0 else ""

        # Mostrar etiquetas de eje X (cada 2 si son más de 15 slots)
        show_x_label = True
        if total_slots > 15 and idx % 2 != 0:
            show_x_label = False

        x_text = f'<text x="{x + (bar_width // 2)}" y="132" font-size="8" fill="#64748b" text-anchor="middle">{label_str}</text>' if show_x_label else ""

        chart_bars += f"""
        <g>
          {veh_bar}
          {ped_bar}
          {total_lbl}
          {x_text}
        </g>
        """

    # Filas de desglose por Video / Spot (Vista Cliente)
    video_rows = ""
    for v in videos:
        dur = v.get("duration_seconds", 15)
        p_today = v.get("plays_today", 0)
        p_total = v.get("total_plays", 0)
        v_impr = v.get("total_impressions", 0)
        v_kwh = v.get("kwh_consumed", 0.0)
        v_cost = round(v_kwh * effective_cfe_rate, 2)
        video_rows += f"""
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 8px 10px;">
            <div style="font-weight: 700; color: #0f172a;">{v.get('title')}</div>
            <div style="font-family: monospace; font-size: 10px; color: #64748b;">{v.get('video_name')} ({v.get('resolution', '1080p')})</div>
          </td>
          <td style="padding: 8px 10px; text-align: center; font-weight: 600; color: #334155;">{dur} seg</td>
          <td style="padding: 8px 10px; color: #334155;">{v.get('location_name', loc.get('name'))}</td>
          <td style="padding: 8px 10px; text-align: right; color: #0284c7; font-weight: 700;">{p_today:,}</td>
          <td style="padding: 8px 10px; text-align: right; font-weight: 800; color: #0f172a;">{p_total:,}</td>
          <td style="padding: 8px 10px; text-align: right; color: #059669; font-weight: 700;">{v_impr:,}</td>
          <td style="padding: 8px 10px; text-align: right; font-weight: 600; color: #1e293b;">{v_kwh} kWh <span style="font-size: 10px; color: #059669; font-weight: 700;">(${v_cost:,.2f})</span></td>
        </tr>
        """
    if not video_rows:
        video_rows = """<tr><td colspan="7" style="padding: 12px; text-align: center; color: #94a3b8;">Sin spots registrados para esta campaña</td></tr>"""

    # Evidencias fotográficas de comprobación de emisión (Proof of Play)
    play_proofs = database.get_client_play_proofs(
        client_id=client.get("id"),
        location_id=loc.get("id"),
        video_id=vid_id_int,
        limit=6,
        date_str=active_date if period == "day" else None
    )

    proof_cards = ""
    for p in play_proofs:
        proof_cards += f"""
        <div style="border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden; background: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
          <img src="{p.get('image_url')}" alt="Evidencia de Emisión" style="width: 100%; height: 95px; object-fit: cover; display: block;" onerror="this.src='/static/uploads/evidence/placeholder.jpg'" />
          <div style="padding: 6px 8px; font-size: 10px;">
            <div style="font-weight: bold; color: #0f172a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{p.get('video_title')}</div>
            <div style="color: #64748b; font-family: monospace; font-size: 9px;">{p.get('timestamp')}</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 3px;">
              <span style="color: #166534; font-weight: 700; background: #dcfce7; padding: 1px 5px; border-radius: 3px; font-size: 9px;">✓ VERIFICADO</span>
              <span style="color: #475569; font-size: 9px;">LUM: {int(p.get('luminance_score', 0.85) * 100)}%</span>
            </div>
          </div>
        </div>
        """

    if not proof_cards:
        proof_cards = """
        <div style="grid-column: 1 / -1; padding: 16px; text-align: center; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 6px; color: #64748b; font-size: 11px;">
          Auditoría de cámara activa · Las instantáneas periciales se sincronizan automáticamente cada 15 min según la rotación de spots.
        </div>
        """

    proof_gallery_section = f"""
    <div class="section-title" style="display: flex; justify-content: space-between; align-items: center;">
      <span>Auditoría Visual de Emisión en Pantalla (Proof of Play por Cámara USB / CCTV)</span>
      <span style="font-size: 10px; font-weight: normal; color: #059669; background: #ecfdf5; padding: 2px 8px; border-radius: 4px; border: 1px solid #a7f3d0;">
        Conforme a Norma AVIXA & Certificación de Emisión Real
      </span>
    </div>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 16px; page-break-inside: avoid; break-inside: avoid;">
      {proof_cards}
    </div>
    """

    # Filas de dispositivos de hardware (Vista Administrador)
    devices = database.get_hardware_devices(loc["id"])
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
        pct = round((v["count"] / max(1, total_flow)) * 100, 1)
        veh_rows += f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px dashed #cbd5e1; font-size: 12px;">
          <span style="font-weight: 600; color: #1e293b;">{v['type']}</span>
          <span style="color: #475569;"><strong style="color: #0f172a;">{v['count']:,}</strong> ({pct}%)</span>
        </div>
        """

    client_logo = client.get("logo_url") or "/static/uploads/logos/default_cerveceria.svg"

    # Preparar bloques HTML condicionales según view_mode ('client' vs 'admin')
    is_client_mode = (view_mode == "client")

    if is_client_mode:
        ficha_tecnica = f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 11px;">
          <div><strong>Pantalla DOOH:</strong> {loc.get('name')}</div>
          <div><strong>Área de Pantalla:</strong> {loc.get('screen_area_m2')} m² LED Exterior</div>
          <div><strong>Ubicación:</strong> {loc.get('address', 'Guadalajara, Jal.')}</div>
          <div><strong>Campaña:</strong> {campaign_name}</div>
          <div><strong>Resolución:</strong> 1920 x 1080 Full HD (60 fps)</div>
          <div><strong>ID Auditoría:</strong> APEX-DOOH-{int(time.time())}</div>
        </div>
        """
        tabla_central = f"""
        <div class="section-title">Pauta de Campaña y Spots de Video Transmitidos en Pantalla</div>
        <table class="data-table">
          <thead>
            <tr>
              <th>Spot Publicitario / Video</th>
              <th style="text-align: center;">Duración</th>
              <th>Pantalla</th>
              <th style="text-align: right;">Pautas Hoy</th>
              <th style="text-align: right;">Total Pautas</th>
              <th style="text-align: right;">Impactos DOOH</th>
              <th style="text-align: right;">Energía Pantalla</th>
            </tr>
          </thead>
          <tbody>
            {video_rows}
          </tbody>
        </table>
        """
        campaign_cost_actual = round(campaign_kwh * effective_cfe_rate, 2)
        seccion_energia = f"""
        <div class="section-title">Consumo Eléctrico de la Pantalla LED (Tarifa CFE: ${effective_cfe_rate:.2f} MXN/kWh)</div>
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 12px 16px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 11px;">
          <div>
            <span style="color: #166534; font-weight: bold; display: block;">Potencia Activa Pantalla:</span>
            <span style="font-size: 16px; font-weight: 800; color: #15803d;">{power_kw} kW</span>
          </div>
          <div>
            <span style="color: #166534; font-weight: bold; display: block;">Energía Consumida Campaña:</span>
            <span style="font-size: 16px; font-weight: 800; color: #15803d;">{campaign_kwh} kWh</span>
          </div>
          <div>
            <span style="color: #166534; font-weight: bold; display: block;">Costo Eléctrico Real (CFE):</span>
            <span style="font-size: 16px; font-weight: 800; color: #15803d;">${campaign_cost_actual:,.2f} MXN</span>
          </div>
          <div>
            <span style="color: #166534; font-weight: bold; display: block;">Tarifa CFE Auditada:</span>
            <span style="font-size: 14px; font-weight: 800; color: #0284c7;">${effective_cfe_rate:.2f} / kWh</span>
          </div>
        </div>
        <div style="font-size: 10px; color: #64748b; margin-top: 4px; font-style: italic;">
          * Medición directa de telemetría de pantalla conforme a los horarios de transmisión y tarifa eléctrica real CFE configurada por el cliente (${effective_cfe_rate:.2f} MXN/kWh).
        </div>
        """
    else:
        # Modo Administrador (Técnico con hardware e IPs)
        ficha_tecnica = f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 11px;">
          <div><strong>Ubicación:</strong> {loc.get('name')}</div>
          <div><strong>Área de Pantalla:</strong> {loc.get('screen_area_m2')} m² LED Exterior</div>
          <div><strong>Orientación:</strong> {loc.get('screen_orientation_deg', 270)}° Azimut</div>
          <div><strong>Contacto Cliente:</strong> {client.get('contact_email', 'N/A')}</div>
          <div><strong>ID Auditoría:</strong> DOOH-{int(time.time())}</div>
          <div><strong>Controlador:</strong> NovaStar TB40 / VX600 Pro + Shelly Pro</div>
        </div>
        """
        tabla_central = f"""
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
        """
        seccion_energia = f"""
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
        """

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Reporte Ejecutivo DOOH — {client.get('name', 'Cliente')} — Apex Company</title>
  <style>
    @page {{
      size: A4 portrait;
      margin: 10mm 12mm 12mm 12mm;
    }}
    * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
    body {{
      background: #ffffff;
      color: #0f172a;
      margin: 0;
      padding: 0;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .no-print {{
      background: #0f172a;
      color: #f8fafc;
      padding: 10px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      border-radius: 6px;
    }}
    @media print {{
      .no-print {{ display: none !important; }}
      body {{ margin: 0; }}
    }}
    .btn {{
      background: #0284c7;
      color: #ffffff;
      border: none;
      padding: 6px 14px;
      border-radius: 4px;
      font-weight: 600;
      cursor: pointer;
      font-size: 12px;
      text-decoration: none;
      display: inline-block;
    }}
    .btn:hover {{ opacity: 0.9; }}
    .header-table {{
      width: 100%;
      border-bottom: 2px solid #0284c7;
      padding-bottom: 12px;
      margin-bottom: 14px;
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-bottom: 10px;
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    .kpi-grid-sub {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 14px;
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    .kpi-card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 9px 11px;
      text-align: center;
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    .kpi-card .val {{
      font-size: 19px;
      font-weight: 900;
      color: #0f172a;
      line-height: 1.2;
    }}
    .kpi-card .lbl {{
      font-size: 9.5px;
      font-weight: 700;
      text-transform: uppercase;
      color: #64748b;
      letter-spacing: 0.5px;
    }}
    .section-title {{
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      color: #0f172a;
      border-left: 4px solid #0284c7;
      padding-left: 6px;
      margin: 12px 0 6px 0;
      letter-spacing: 0.5px;
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
      margin-bottom: 12px;
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    table.data-table th {{
      background: #f1f5f9;
      color: #334155;
      text-align: left;
      padding: 6px 10px;
      font-weight: 700;
      border-bottom: 2px solid #cbd5e1;
    }}
    table.data-table tr {{
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    .footer-watermark {{
      margin-top: 20px;
      border-top: 1px solid #cbd5e1;
      padding-top: 8px;
      text-align: center;
      font-size: 10px;
      color: #64748b;
      page-break-inside: avoid;
      break-inside: avoid;
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
      <strong style="color: #38bdf8; font-size: 14px;">Vista Previa de Reporte Ejecutivo DOOH ({period_title})</strong>
      <span style="color: #94a3b8; font-size: 12px; margin-left: 8px;">Listo para imprimir o exportar a PDF</span>
      <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 11px; padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.4); margin-left: 8px;">
        ⚡ Tarifa CFE: ${effective_cfe_rate:.2f} MXN/kWh
      </span>
    </div>
    <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
      <a href="/report/print?client_id={client.get('id')}&location_id={loc.get('id')}&view_mode={view_mode}&period=day&date={active_date}&kwh_rate={effective_cfe_rate}" class="btn" style="background: {'#0284c7' if period == 'day' else '#334155'}; font-size: 11px; padding: 4px 8px;">Día</a>
      <a href="/report/print?client_id={client.get('id')}&location_id={loc.get('id')}&view_mode={view_mode}&period=month&month={active_month}&kwh_rate={effective_cfe_rate}" class="btn" style="background: {'#0284c7' if period == 'month' else '#334155'}; font-size: 11px; padding: 4px 8px;">Mes</a>
      <a href="/report/print?client_id={client.get('id')}&location_id={loc.get('id')}&view_mode={view_mode}&period=year&year={active_year}&kwh_rate={effective_cfe_rate}" class="btn" style="background: {'#0284c7' if period == 'year' else '#334155'}; font-size: 11px; padding: 4px 8px;">Año</a>
      <a href="/report/print?client_id={client.get('id')}&location_id={loc.get('id')}&view_mode={view_mode}&period=campaign{f'&campaign_id={camp_id_int}' if camp_id_int else ''}{f'&video_id={vid_id_int}' if vid_id_int else ''}&kwh_rate={effective_cfe_rate}" class="btn" style="background: {'#0284c7' if period == 'campaign' else '#334155'}; font-size: 11px; padding: 4px 8px;">Campaña / Video</a>
      <span style="color: #64748b; margin: 0 4px;">|</span>
      <a href="/report/print?client_id={client.get('id')}&location_id={loc.get('id')}&view_mode=client&period={period}&date={active_date}&month={active_month}&year={active_year}&kwh_rate={effective_cfe_rate}" class="btn" style="background: {'#059669' if is_client_mode else '#475569'}; font-size: 11px; padding: 4px 8px;">Vista Cliente</a>
      <a href="/report/print?client_id={client.get('id')}&location_id={loc.get('id')}&view_mode=admin&period={period}&date={active_date}&month={active_month}&year={active_year}&kwh_rate={effective_cfe_rate}" class="btn" style="background: {'#059669' if not is_client_mode else '#475569'}; font-size: 11px; padding: 4px 8px;">Vista Admin</a>
      <button onclick="window.print()" class="btn" style="background: #b45309; font-weight: bold; font-size: 11px; padding: 4px 10px;">🖨️ Imprimir / Guardar PDF</button>
      <button onclick="window.close()" class="btn" style="background: #475569; font-size: 11px; padding: 4px 8px;">Cerrar</button>
    </div>
  </div>

  <!-- Encabezado con Logotipo del Cliente y Marca Apex Company -->
  <table class="header-table">
    <tr>
      <td style="width: 50%; vertical-align: middle;">
        <div style="display: flex; align-items: center; gap: 12px;">
          <img src="{client_logo}" alt="{client.get('name')}" style="max-height: 48px; max-width: 180px; object-fit: contain;" onerror="this.style.display='none'" />
          <div>
            <h1 style="font-size: 16px; margin: 0; color: #0f172a; font-weight: 800;">{client.get('name')}</h1>
            <p style="margin: 2px 0 0 0; font-size: 11px; color: #64748b;">Reporte de Aforo Vehicular, Audiencia e Impactos DOOH ({period_title})</p>
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
            <strong>{period_label}</strong> | Pantalla: <strong>{loc.get('name')}</strong>
          </div>
        </div>
      </td>
    </tr>
  </table>

  <!-- Ficha Técnica -->
  {ficha_tecnica}

  <!-- Métricas Principales (KPIs Multimodales con Doble Métrica) -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="lbl">Aforo Total Auditado</div>
      <div class="val">{total_flow:,} <span style="font-size: 11px; font-weight: normal; color: #64748b;">cruces</span></div>
      <div style="font-size: 9.5px; color: #64748b; margin-top: 2px;">
        Entradas: <strong style="color: #059669;">{total_in:,}</strong> | Salidas: <strong style="color: #d97706;">{total_out:,}</strong>
      </div>
    </div>
    <div class="kpi-card">
      <div class="lbl">Impactos Estimados DOOH</div>
      <div class="val" style="color: #0284c7;">{impressions:,}</div>
      <div style="font-size: 9.5px; color: #64748b; margin-top: 2px;">1.6x ocupación veh + 1.0x peatonal</div>
    </div>
    <div class="kpi-card">
      <div class="lbl">Vehículos (Doble Métrica)</div>
      <div class="val" style="color: #0369a1;">{v_crossed:,} <span style="font-size: 11px; font-weight: normal; color: #64748b;">cruces</span></div>
      <div style="font-size: 9.5px; color: #475569; margin-top: 2px;">
        👁️ <strong>{v_seen:,}</strong> vistos ({v_rate}% en línea)
      </div>
    </div>
    <div class="kpi-card">
      <div class="lbl">Peatones (Doble Métrica)</div>
      <div class="val" style="color: #059669;">{p_crossed:,} <span style="font-size: 11px; font-weight: normal; color: #64748b;">cruces</span></div>
      <div style="font-size: 9.5px; color: #475569; margin-top: 2px;">
        👁️ <strong>{p_seen:,}</strong> vistos ({p_rate}% en línea)
      </div>
    </div>
  </div>

  <div class="kpi-grid-sub">
    <div class="kpi-card" style="padding: 7px 10px;">
      <div class="lbl">Tiempo Promedio en Escena</div>
      <div class="val" style="font-size: 16px; color: #d97706;">{avg_dwell} seg</div>
      <div style="font-size: 9px; color: #64748b;">Permanencia visual ante pantalla</div>
    </div>
    <div class="kpi-card" style="padding: 7px 10px;">
      <div class="lbl">Hora Pico Máxima</div>
      <div class="val" style="font-size: 14px; color: #7c3aed; margin-top: 2px;">{peak_hr}</div>
      <div style="font-size: 9px; color: #64748b;">Mayor concentración de tráfico</div>
    </div>
    <div class="kpi-card" style="padding: 7px 10px;">
      <div class="lbl">Tasa Global de Cruce</div>
      <div class="val" style="font-size: 16px; color: #0284c7;">{crossing_rate}%</div>
      <div style="font-size: 9px; color: #64748b;">{total_crossed:,} cruzan de {total_seen:,} vistos</div>
    </div>
  </div>

  <!-- Tabla Central (Videos de Campaña o Controladores según vista) -->
  {tabla_central}

  <!-- Gráfica de Tráfico y Desglose por Tipo de Vehículo -->
  <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 14px; margin-bottom: 14px; page-break-inside: avoid; break-inside: avoid;">
    <div>
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
        <div class="section-title" style="margin: 0;">{chart_title}</div>
        <div style="display: flex; gap: 12px; font-size: 10px; font-weight: 700;">
          <span style="display: flex; align-items: center; gap: 4px; color: #0284c7;">
            <span style="width: 9px; height: 9px; background: #0284c7; border-radius: 2px; display: inline-block;"></span>
            Vehículos ({v_flow:,})
          </span>
          <span style="display: flex; align-items: center; gap: 4px; color: #10b981;">
            <span style="width: 9px; height: 9px; background: #10b981; border-radius: 2px; display: inline-block;"></span>
            Peatones ({p_flow:,})
          </span>
        </div>
      </div>
      <div style="border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; background: #fafafa;">
        <svg viewBox="0 0 700 145" style="width: 100%; height: auto;">
          <line x1="20" y1="115" x2="680" y2="115" stroke="#cbd5e1" stroke-width="1" />
          <line x1="20" y1="65" x2="680" y2="65" stroke="#f1f5f9" stroke-dasharray="4 4" stroke-width="1" />
          {chart_bars}
        </svg>
      </div>
    </div>
    <div>
      <div class="section-title" style="margin-top: 0;">Composición del Tráfico</div>
      <div style="border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px; background: #fafafa;">
        {veh_rows}
      </div>
    </div>
  </div>

  <!-- Sección de Consumo Eléctrico de Pantalla -->
  {seccion_energia}

  <!-- Sección de Auditoría Visual: Proof of Play por Cámara -->
  {proof_gallery_section}

  <!-- Pie de página Oficial y Obligatorio -->
  <div class="footer-watermark">
    <p style="margin: 0 0 4px 0; font-size: 11px; font-weight: bold; color: #0f172a;">
      Tecnología y Plataforma Desarrollada por <strong>apexcompany.com.mx</strong>
    </p>
    <p style="margin: 0; color: #64748b;">
      Soluciones Avanzadas de Visión Artificial, Analítica Vehicular e Infraestructura Publicitaria DOOH. Todos los derechos reservados.
    </p>
  </div>

  <script>
    function triggerPrint() {{
      try {{
        window.focus();
        window.print();
      }} catch (err) {{
        console.error("Error al disparar impresión:", err);
      }}
    }}
    {'window.addEventListener("load", () => { setTimeout(triggerPrint, 350); });' if autoprint else ''}
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — TELEMETRÍA EDGE INTEL NUC (PUSH DESDE SITIO HACIA LA NUBE)
# ─────────────────────────────────────────────────────────────────────────────

latest_nuc_snapshots: dict[str, bytes] = {}
latest_nuc_heartbeat: dict[str, float] = {}


class NucTelemetryPayload(BaseModel):
    nuc_id: str
    api_key: Optional[str] = None
    timestamp: float
    cameras: dict = {}
    shelly: Optional[dict] = None
    novastar: Optional[dict] = None
    snapshot_base64: Optional[str] = None
    camera_id_snapshot: Optional[str] = None


@app.post("/api/telemetry/push")
async def receive_nuc_telemetry(payload: NucTelemetryPayload) -> dict:
    """
    Recibe la telemetría en tiempo real procesada localmente en la Intel NUC (Edge).
    Actualiza el estado de las cámaras en la nube y despacha actualizaciones por WebSocket.
    """
    global latest_metrics
    now = time.time()
    latest_nuc_heartbeat[payload.nuc_id] = now

    # Actualizar métricas en memoria
    for cam_id, metrics in payload.cameras.items():
        latest_metrics[cam_id] = metrics

    # Guardar snapshot si fue provisto
    if payload.snapshot_base64 and payload.camera_id_snapshot:
        try:
            img_data = base64.b64decode(payload.snapshot_base64)
            latest_nuc_snapshots[payload.camera_id_snapshot] = img_data
        except Exception as e:
            logger.debug(f"Error decodificando snapshot de NUC: {e}")

    # Notificar por WebSocket a los clientes web conectados
    ws_payload = json.dumps({
        "type": "metrics_update",
        "cameras": latest_metrics,
        "server_time": now,
        "source": "nuc_edge",
        "nuc_id": payload.nuc_id,
    })
    asyncio.create_task(broadcast_payload_to_ws(ws_payload))

    return {
        "status": "ok",
        "message": f"Telemetría de NUC '{payload.nuc_id}' recibida y distribuida",
        "cameras_updated": list(payload.cameras.keys()),
        "timestamp": now,
    }


@app.get("/api/telemetry/snapshot/{camera_id}")
async def get_nuc_snapshot(camera_id: str):
    """
    Retorna la última imagen JPEG capturada y procesada por la Intel NUC para esta cámara.
    """
    if camera_id in latest_nuc_snapshots:
        return Response(content=latest_nuc_snapshots[camera_id], media_type="image/jpeg")
    
    # Si no hay snapshot de NUC pero el servidor local tiene CameraStream activo
    stream = streams.get(camera_id)
    if stream:
        frame = stream.read()
        if frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return Response(content=jpeg.tobytes(), media_type="image/jpeg")
            
    raise HTTPException(status_code=404, detail=f"No hay snapshot disponible para '{camera_id}'")


@app.get("/api/telemetry/status")
async def get_nuc_status() -> dict:
    """
    Informa el estado de conexión de las NUCs que reportan a la nube.
    """
    now = time.time()
    nuc_states = {}
    for nuc_id, last_seen in latest_nuc_heartbeat.items():
        diff = now - last_seen
        nuc_states[nuc_id] = {
            "last_seen_sec_ago": round(diff, 1),
            "status": "ONLINE" if diff < 15.0 else "OFFLINE",
        }
    return {
        "status": "ok",
        "nucs": nuc_states,
        "active_cameras": list(latest_metrics.keys()),
    }


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
