"""
traffic_analyzer.py — Motor de inferencia y análisis de tráfico vehicular.

Responsabilidades:
  - Cargar el modelo YOLOv8 (con soporte OpenVINO/NPU via Ultralytics).
  - Ejecutar detección e inicializar ByteTrack para seguimiento multi-objeto.
  - Manejar LineZone para conteo de cruces entrada/salida.
  - Calcular dwell time por track_id para clasificar Lead vs Bounced.
  - Devolver el frame anotado + métricas estructuradas por cámara.

Dependencias: ultralytics, supervision, opencv-python, config.py
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

import config
import database

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleMetrics:
    """
    Métricas de tráfico acumuladas para UNA cámara.
    Se serializa a dict para enviarla por WebSocket / REST.
    """
    cam_id: str
    total_in: int = 0                           # Total de vehículos que cruzaron hacia adentro
    total_out: int = 0                          # Total de vehículos que cruzaron hacia afuera
    active_tracks: int = 0                      # Objetos actualmente en pantalla
    leads: int = 0                              # Tracks con dwell >= DWELL_LEAD_SECONDS
    bounced: int = 0                            # Tracks con dwell < DWELL_LEAD_SECONDS (ya salieron)
    class_counts: dict = field(default_factory=dict)   # { "Automovil": 5, "Camion": 2, ... }
    fps_processing: float = 0.0                 # FPS de inferencia (para monitoreo)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Serializa las métricas a un diccionario JSON-compatible."""
        return {
            "cam_id": self.cam_id,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "total_flow": self.total_in + self.total_out,
            "active_tracks": self.active_tracks,
            "leads": self.leads,
            "bounced": self.bounced,
            "class_counts": self.class_counts,
            "fps_processing": round(self.fps_processing, 1),
            "timestamp": self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL: TrafficAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class TrafficAnalyzer:
    """
    Encapsula YOLOv8 + ByteTrack + LineZone para analizar tráfico en una cámara.

    Cada instancia de TrafficAnalyzer corresponde a UNA cámara.
    El método `process_frame()` es el punto de entrada para cada ciclo de inferencia.
    """

    def __init__(self, cam_id: str) -> None:
        """
        Inicializa el analizador para una cámara específica.

        Args:
            cam_id: Identificador de la cámara (debe existir en config.COUNTING_LINES).
        """
        self.cam_id = cam_id
        self.metrics = VehicleMetrics(cam_id=cam_id)

        # Diccionario: track_id → timestamp del primer frame en que se vio
        self._track_first_seen: dict[int, float] = {}

        # Diccionario: track_id → clase de vehículo detectada (str)
        self._track_class: dict[int, str] = {}

        # Conjunto de tracks que ya salieron de pantalla (para calcular bounced/leads)
        self._finished_tracks: set[int] = set()

        # Conteo de clase acumulado para vehículos que cruzaron la línea
        self._class_counts: dict[str, int] = defaultdict(int)

        # Timestamps para medir FPS de inferencia
        self._last_inference_time: float = time.time()

        # ── Cargar modelo YOLO ───────────────────────────────────────────────
        logger.info(f"[{cam_id}] Cargando modelo YOLO desde: {config.YOLO_MODEL_PATH}")
        try:
            self._model = YOLO(config.YOLO_MODEL_PATH)
            # Warm-up: inferencia con frame vacío para inicializar el runtime
            dummy = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            self._model(dummy, device=config.INFERENCE_DEVICE, verbose=False)
            logger.info(f"[{cam_id}] Modelo cargado en dispositivo: {config.INFERENCE_DEVICE}")
        except Exception as exc:
            logger.exception(f"[{cam_id}] Error al cargar el modelo YOLO: {exc}")
            raise

        # ── Inicializar LineZone ─────────────────────────────────────────────
        line_coords = config.COUNTING_LINES.get(cam_id)
        if line_coords is None:
            # Si no hay línea definida para esta cámara, usar una línea central por defecto
            logger.warning(
                f"[{cam_id}] No hay COUNTING_LINE definida. "
                "Usando línea horizontal central como fallback."
            )
            line_coords = (
                (50, config.FRAME_HEIGHT // 2),
                (config.FRAME_WIDTH - 50, config.FRAME_HEIGHT // 2),
            )

        self._line_zone = sv.LineZone(
            start=sv.Point(*line_coords[0]),
            end=sv.Point(*line_coords[1]),
        )

        # Offsets para preservar conteos al mover la línea
        self._in_offset: int = 0
        self._out_offset: int = 0

        # ── Anotadores visuales de supervision ──────────────────────────────
        self._box_annotator = sv.BoxAnnotator(thickness=2)
        self._label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=4,
        )
        self._trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=30,    # Largo de la estela del track (en frames)
        )
        self._line_zone_annotator = sv.LineZoneAnnotator(
            thickness=3,
            text_scale=1.0,
            text_thickness=2,
        )

        logger.info(f"[{cam_id}] TrafficAnalyzer listo.")

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL: process_frame
    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, stream_fps: float = config.ASSUMED_FPS) -> tuple[np.ndarray, VehicleMetrics]:
        """
        Ejecuta el pipeline completo de inferencia y análisis sobre un frame.

        Pipeline:
          1. Detección YOLO (filtrada por clase y confianza).
          2. Seguimiento ByteTrack (asigna track_id persistentes).
          3. Conteo LineZone (detecta cruces IN/OUT).
          4. Cálculo dwell time (Lead vs Bounced).
          5. Anotación visual del frame.

        Args:
            frame      : Frame BGR (numpy array) de la cámara.
            stream_fps : FPS reportado por el stream (para cálculos de tiempo).

        Returns:
            Tuple (frame_anotado, métricas_actualizadas).
        """
        t_start = time.time()

        # ── 1. Inferencia YOLO y Tracking (ByteTrack interno) ────────────────
        results = self._model.track(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=config.VEHICLE_CLASS_IDS,
            device=config.INFERENCE_DEVICE,
            verbose=False,
            persist=True,
            tracker="bytetrack.yaml"
        )[0]  # Tomar el primer resultado (batch_size=1)

        # Convertir resultados YOLO → formato supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Filtrar solo las clases de vehículos (doble filtro de seguridad)
        if detections.class_id is not None:
            mask = np.isin(detections.class_id, config.VEHICLE_CLASS_IDS)
            detections = detections[mask]

        # ── 3. Registro de primer avistamiento y clase por track ─────────────
        current_time = time.time()
        if detections.tracker_id is not None:
            for i, track_id in enumerate(detections.tracker_id):
                track_id = int(track_id)

                # Registrar primera vez que se ve este track
                if track_id not in self._track_first_seen:
                    self._track_first_seen[track_id] = current_time

                # Registrar la clase del vehículo (si aún no se conoce)
                if track_id not in self._track_class and detections.class_id is not None:
                    class_id = int(detections.class_id[i])
                    self._track_class[track_id] = config.CLASS_LABELS.get(class_id, "Desconocido")

        # ── 4. Conteo LineZone y Registro de Eventos ────────────────────────
        if detections.tracker_id is not None:
            crossed_in, crossed_out = self._line_zone.trigger(detections)
            for i, tid in enumerate(detections.tracker_id):
                tid = int(tid)
                v_type = self._track_class.get(tid, "Automovil")
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.85
                dwell = current_time - self._track_first_seen.get(tid, current_time)

                if crossed_in[i]:
                    try:
                        database.record_event(
                            camera_id=self.cam_id,
                            track_id=tid,
                            vehicle_type=v_type,
                            direction="IN",
                            confidence=conf,
                            dwell_time=dwell
                        )
                    except Exception as e:
                        logger.error(f"[{self.cam_id}] Error guardando evento IN: {e}")
                elif crossed_out[i]:
                    try:
                        database.record_event(
                            camera_id=self.cam_id,
                            track_id=tid,
                            vehicle_type=v_type,
                            direction="OUT",
                            confidence=conf,
                            dwell_time=dwell
                        )
                    except Exception as e:
                        logger.error(f"[{self.cam_id}] Error guardando evento OUT: {e}")

        # Leer conteos acumulados directamente del LineZone más los offsets
        # supervision 0.21+ almacena los conteos en .in_count y .out_count
        self.metrics.total_in = int(self._line_zone.in_count) + self._in_offset
        self.metrics.total_out = int(self._line_zone.out_count) + self._out_offset

        # ── 5. Cálculo dwell time y clasificación Lead/Bounced ───────────────
        active_ids = set(int(tid) for tid in detections.tracker_id) if detections.tracker_id is not None else set()
        self._update_dwell_metrics(active_ids, current_time)

        # ── 6. Conteo de clases activas ──────────────────────────────────────
        active_class_counts: dict[str, int] = defaultdict(int)
        if detections.tracker_id is not None and detections.class_id is not None:
            for i in range(len(detections)):
                class_id = int(detections.class_id[i])
                label = config.CLASS_LABELS.get(class_id, "Desconocido")
                active_class_counts[label] += 1
        self.metrics.class_counts = dict(active_class_counts)

        # ── 7. Métricas de rendimiento ───────────────────────────────────────
        self.metrics.active_tracks = len(active_ids)
        self.metrics.fps_processing = 1.0 / max(time.time() - t_start, 1e-6)
        self.metrics.timestamp = current_time

        # ── 8. Anotación visual del frame ─────────────────────────────────────
        annotated_frame = self._annotate_frame(frame.copy(), detections)

        return annotated_frame, self.metrics

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS AUXILIARES PRIVADOS
    # ──────────────────────────────────────────────────────────────────────────

    def _update_dwell_metrics(self, active_ids: set[int], current_time: float) -> None:
        """
        Calcula cuántos tracks activos son 'Lead' (dwell >= umbral) y
        clasifica los tracks que ya desaparecieron como Lead o Bounced.

        Args:
            active_ids   : Conjunto de track_ids presentes en el frame actual.
            current_time : Timestamp del frame actual.
        """
        # Identificar tracks que ya no están en pantalla (salieron)
        known_ids = set(self._track_first_seen.keys())
        disappeared_ids = known_ids - active_ids - self._finished_tracks

        for tid in disappeared_ids:
            # Calcular tiempo en pantalla
            dwell = current_time - self._track_first_seen[tid]
            if dwell >= config.DWELL_LEAD_SECONDS:
                self.metrics.leads += 1
            else:
                self.metrics.bounced += 1
            self._finished_tracks.add(tid)

        # Contar tracks activos que YA son Lead (llevan suficiente tiempo)
        active_leads = sum(
            1 for tid in active_ids
            if (current_time - self._track_first_seen.get(tid, current_time))
               >= config.DWELL_LEAD_SECONDS
        )
        # Nota: metrics.leads acumula los que ya salieron.
        # active_leads es informativo (se podría exponer por separado si se necesita).
        _ = active_leads  # Silenciar linter; usar si se requiere en el futuro

    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Dibuja bounding boxes, etiquetas, trazas y línea de conteo en el frame.

        Args:
            frame      : Frame original (copia, para no modificar el original).
            detections : Detecciones con track_ids asignados por ByteTrack.

        Returns:
            Frame con anotaciones visuales BGR.
        """
        # Construir etiquetas enriquecidas: "ID:42 | Automovil | 2.3s"
        labels = []
        if detections.tracker_id is not None:
            current_time = time.time()
            for i, track_id in enumerate(detections.tracker_id):
                track_id = int(track_id)
                clase = self._track_class.get(track_id, "?")
                dwell = current_time - self._track_first_seen.get(track_id, current_time)
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                etiqueta = f"#{track_id} {clase} {dwell:.1f}s [{conf:.0%}]"
                labels.append(etiqueta)

        # Dibujar estelas de movimiento solo si hay IDs de track
        if detections.tracker_id is not None:
            frame = self._trace_annotator.annotate(scene=frame, detections=detections)

        # Dibujar bounding boxes
        frame = self._box_annotator.annotate(scene=frame, detections=detections)

        # Dibujar etiquetas de texto
        if labels:
            frame = self._label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )

        # Dibujar línea de conteo con totales IN/OUT
        frame = self._line_zone_annotator.annotate(
            frame=frame, line_counter=self._line_zone
        )

        # ── HUD de estado en la esquina superior izquierda ───────────────────
        overlay_lines = [
            f"CAM: {self.cam_id}",
            f"Activos: {self.metrics.active_tracks}",
            f"IN: {self.metrics.total_in}  OUT: {self.metrics.total_out}",
            f"Leads: {self.metrics.leads}  Bounced: {self.metrics.bounced}",
            f"FPS: {self.metrics.fps_processing:.1f}",
        ]
        y_offset = 30
        for line in overlay_lines:
            cv2.putText(
                frame, line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),    # Verde
                2,
                cv2.LINE_AA,
            )
            y_offset += 25

        return frame

    def update_line(self, line_coords: tuple[tuple[int, int], tuple[int, int]]) -> None:
        """Actualiza las coordenadas de la línea de conteo manteniendo los contadores actuales."""
        import supervision as sv
        # Acumular conteos actuales en el offset antes de recrear
        self._in_offset += int(self._line_zone.in_count)
        self._out_offset += int(self._line_zone.out_count)
        
        # Guardar el estado del tracker para no perder cruces en progreso
        old_state = getattr(self._line_zone, 'crossing_state_history', None)
        # Recrear la zona de línea con las nuevas coordenadas
        self._line_zone = sv.LineZone(
            start=sv.Point(*line_coords[0]),
            end=sv.Point(*line_coords[1]),
        )
        if old_state is not None:
            self._line_zone.crossing_state_history = old_state
            
        logger.info(f"[{self.cam_id}] Línea actualizada a {line_coords}.")

    def reset_counts(self) -> None:
        """Reinicia todos los contadores (útil para turno/jornada nueva)."""
        self.metrics = VehicleMetrics(cam_id=self.cam_id)
        self._track_first_seen.clear()
        self._track_class.clear()
        self._finished_tracks.clear()
        self._class_counts.clear()
        # Reiniciar línea de conteo
        try:
            # supervision 0.21+ expone reset() en LineZone
            self._line_zone.reset()
        except AttributeError:
            # Fallback si no tiene reset (recreamos)
            import supervision as sv
            start = self._line_zone.vector.start
            end = self._line_zone.vector.end
            self._line_zone = sv.LineZone(start=start, end=end)
            
        self._in_offset = 0
        self._out_offset = 0
        logger.info(f"[{self.cam_id}] Contadores reiniciados.")
