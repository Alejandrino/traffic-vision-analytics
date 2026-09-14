"""
camera_stream.py — Módulo de captura de video en hilo independiente.

Responsabilidades:
  - Abrir y mantener la conexión a cada fuente de video (RTSP o USB).
  - Leer frames continuamente en un Thread dedicado para no bloquear la inferencia.
  - Reconectarse automáticamente ante desconexiones de red (crucial para RTSP).
  - Exponer el último frame disponible de forma thread-safe mediante un Lock.

Dependencias: opencv-python, config.py
"""

import cv2
import time
import logging
import threading
from typing import Optional, Union

import config

# Configurar logger específico de este módulo
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class CameraStream:
    """
    Captura frames de una cámara (RTSP o USB) en un hilo de fondo.

    Uso básico:
        stream = CameraStream(cam_id="cam1", source="rtsp://...")
        stream.start()
        frame = stream.get_frame()   # No bloqueante; devuelve None si aún no hay frame
        stream.stop()
    """

    def __init__(self, cam_id: str, source: Union[str, int]) -> None:
        """
        Inicializa la instancia de captura SIN abrir la cámara todavía.

        Args:
            cam_id  : Identificador textual de la cámara (ej. "cam1").
            source  : URL RTSP (str) o índice de cámara USB (int).
        """
        self.cam_id: str = cam_id
        self.source: Union[str, int] = source

        # Estado interno
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[object] = None        # Último frame capturado (numpy array)
        self._frame_lock: threading.Lock = threading.Lock()

        # Control del hilo
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False                  # Señal de parada limpia

        # Métricas de operación
        self.fps: float = config.ASSUMED_FPS        # Se actualiza con el FPS real del stream
        self.connected: bool = False
        self.enabled: bool = True                   # Control de activación/desactivación
        self.reconnect_count: int = 0               # Contador de reconexiones acumuladas
        self.last_frame_time: float = 0.0           # Timestamp del último frame recibido

        logger.info(f"[{self.cam_id}] CameraStream creado para fuente: {self.source}")

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PÚBLICOS
    # ──────────────────────────────────────────────────────────────────────────

    def set_enabled(self, enabled: bool) -> bool:
        """
        Activa o desactiva la captura de la cámara.
        Si se desactiva, detiene el hilo de lectura y libera los recursos de hardware (OpenCV).
        Si se activa, vuelve a iniciar el hilo de captura en segundo plano.
        """
        if self.enabled == enabled:
            return self.enabled

        self.enabled = enabled
        if enabled:
            logger.info(f"[{self.cam_id}] Reactivando captura de cámara...")
            self.start()
        else:
            logger.info(f"[{self.cam_id}] Desactivando captura de cámara (pausa de hardware)...")
            self.stop()
            self.connected = False
            with self._frame_lock:
                self._frame = None
        return self.enabled

    def start(self) -> "CameraStream":
        """
        Inicia el hilo de captura en segundo plano.
        Retorna self para permitir encadenamiento: stream = CameraStream(...).start()
        """
        if not self.enabled:
            logger.info(f"[{self.cam_id}] No se inicia porque la cámara está desactivada.")
            return self
        if self._running:
            logger.warning(f"[{self.cam_id}] El hilo de captura ya está en ejecución.")
            return self

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"CameraStream-{self.cam_id}",
            daemon=True,    # El hilo muere cuando el proceso principal termina
        )
        self._thread.start()
        logger.info(f"[{self.cam_id}] Hilo de captura iniciado (Thread: {self._thread.name}).")
        return self

    def stop(self) -> None:
        """
        Detiene el hilo de captura y libera los recursos de OpenCV.
        Bloquea hasta que el hilo termina (máx. 5 seg).
        """
        logger.info(f"[{self.cam_id}] Solicitando detención del hilo de captura...")
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(f"[{self.cam_id}] El hilo no se detuvo en 5 seg.")

        self._release_capture()
        logger.info(f"[{self.cam_id}] Hilo de captura detenido y recursos liberados.")

    def get_frame(self) -> Optional[object]:
        """
        Retorna el último frame capturado de forma thread-safe.

        Returns:
            numpy.ndarray con el frame BGR, o None si aún no hay frame disponible.
        """
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def is_running(self) -> bool:
        """Indica si el hilo de captura está activo y habilitado."""
        return self.enabled and self._running and (self._thread is not None) and self._thread.is_alive()

    # ──────────────────────────────────────────────────────────────────────────
    # LÓGICA INTERNA DEL HILO
    # ──────────────────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Bucle principal del hilo de captura.

        1. Intenta conectarse a la fuente de video.
        2. Lee frames continuamente y los almacena en self._frame.
        3. Ante fallos de lectura, espera y reintenta la conexión (solo RTSP).
        4. Termina limpiamente cuando self._running == False.
        """
        retries: int = 0
        max_retries: int = config.RTSP_MAX_RETRIES    # -1 = infinito

        while self._running:
            # ── Intentar abrir la fuente ────────────────────────────────────
            if not self._open_capture():
                retries += 1
                if max_retries != -1 and retries > max_retries:
                    logger.error(
                        f"[{self.cam_id}] Se alcanzó el máximo de reintentos "
                        f"({max_retries}). Deteniendo captura."
                    )
                    self._running = False
                    break

                logger.warning(
                    f"[{self.cam_id}] Reintento {retries} en "
                    f"{config.RTSP_RECONNECT_DELAY} seg..."
                )
                time.sleep(config.RTSP_RECONNECT_DELAY)
                continue

            # ── Lectura continua de frames ───────────────────────────────────
            retries = 0    # Reiniciar contador al conectar exitosamente
            self.connected = True
            self.reconnect_count += (1 if self.reconnect_count > 0 else 0)
            logger.info(f"[{self.cam_id}] Leyendo frames del stream...")

            while self._running:
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    # Frame inválido: posible desconexión o fin del stream
                    logger.warning(
                        f"[{self.cam_id}] Frame inválido recibido. "
                        "Posible desconexión del stream."
                    )
                    self.connected = False
                    self._release_capture()
                    break   # Salir al bucle externo para reconectar

                # === ESTANDARIZAR RESOLUCIÓN ===
                # Redimensionar el frame para que las coordenadas (YOLO, tracking, LineZone, UI)
                # coincidan siempre, sin importar la resolución original de la cámara.
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

                # ── Detectar timeout por frame inactivo ─────────────────────
                now = time.time()
                if self.last_frame_time > 0:
                    elapsed = now - self.last_frame_time
                    if elapsed > config.RTSP_FRAME_TIMEOUT:
                        logger.warning(f"[{self.cam_id}] Timeout de stream detectado ({elapsed:.1f}s)")
                        logger.warning(
                            f"[{self.cam_id}] Sin frames por {elapsed:.1f} seg "
                            f"(timeout={config.RTSP_FRAME_TIMEOUT} seg). Reconectando..."
                        )
                        self.connected = False
                        self._release_capture()
                        break

                # ── Frame válido: almacenar thread-safe ──────────────────────
                self.last_frame_time = now
                with self._frame_lock:
                    self._frame = frame

        # Limpieza al salir del bucle principal
        self.connected = False
        self._release_capture()
        logger.info(f"[{self.cam_id}] Bucle de captura finalizado.")

    def _open_capture(self) -> bool:
        """
        Abre la fuente de video con OpenCV y configura resolución y transporte.

        Returns:
            True si la apertura fue exitosa, False en caso contrario.
        """
        logger.info(f"[{self.cam_id}] Conectando a: {self.source}")

        try:
            # Para fuentes RTSP se agregan parámetros FFmpeg
            if isinstance(self.source, str) and self.source.lower().startswith("rtsp"):
                # Forzar transporte TCP para mayor estabilidad en LAN
                source_with_options = (
                    f"{self.source}"
                    f"?rtsp_transport={config.RTSP_TRANSPORT}"
                    f"&stimeout=5000000"   # 5 seg de timeout en micro-segundos
                )
                self._cap = cv2.VideoCapture(source_with_options, cv2.CAP_FFMPEG)
            else:
                # Dispositivo USB o ruta de archivo local
                self._cap = cv2.VideoCapture(self.source)

            # Verificar que se abrió correctamente
            if not self._cap.isOpened():
                logger.error(f"[{self.cam_id}] No se pudo abrir la fuente: {self.source}")
                self._cap = None
                return False

            # ── Configurar resolución preferida ─────────────────────────────
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

            # ── Leer FPS real del stream ─────────────────────────────────────
            reported_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if reported_fps and reported_fps > 0:
                self.fps = reported_fps
                logger.info(f"[{self.cam_id}] FPS reportado por el stream: {self.fps:.1f}")
            else:
                self.fps = config.ASSUMED_FPS
                logger.warning(
                    f"[{self.cam_id}] FPS no reportado. Usando valor asumido: {self.fps}"
                )

            # Registrar el momento de conexión
            self.last_frame_time = time.time()
            self.reconnect_count += 1
            logger.info(f"[{self.cam_id}] Conexión exitosa. Reconexiones totales: {self.reconnect_count}")
            return True

        except Exception as exc:
            logger.exception(f"[{self.cam_id}] Excepción al abrir la fuente: {exc}")
            self._cap = None
            return False

    def _release_capture(self) -> None:
        """
        Libera el objeto VideoCapture de OpenCV de forma segura.
        Evita llamar a release() si ya fue liberado (previene crash en OpenCV).
        """
        if self._cap is not None:
            try:
                self._cap.release()
                logger.debug(f"[{self.cam_id}] VideoCapture liberado.")
            except Exception as exc:
                logger.warning(f"[{self.cam_id}] Error al liberar VideoCapture: {exc}")
            finally:
                self._cap = None

    # ──────────────────────────────────────────────────────────────────────────
    # REPRESENTACIÓN
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"CameraStream(cam_id={self.cam_id!r}, source={self.source!r}, "
            f"connected={self.connected}, fps={self.fps:.1f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FÁBRICA DE CÁMARAS
# ─────────────────────────────────────────────────────────────────────────────

def create_all_streams() -> dict[str, CameraStream]:
    """
    Lee CAMERA_SOURCES desde config.py y crea + arranca todos los streams.

    Returns:
        Dict cam_id → CameraStream (ya iniciado).
    """
    streams: dict[str, CameraStream] = {}
    for cam_id, source in config.CAMERA_SOURCES.items():
        stream = CameraStream(cam_id=cam_id, source=source)
        stream.start()
        streams[cam_id] = stream
        logger.info(f"Stream iniciado para cámara '{cam_id}'.")
    return streams
