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
from typing import Optional, Any
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
    Incluye desglose dual: Vehicular y Peatonal.
    Se serializa a dict para enviarla por WebSocket / REST.
    """
    cam_id: str
    total_in: int = 0                           # Total combinado de cruces hacia adentro
    total_out: int = 0                          # Total combinado de cruces hacia afuera
    active_tracks: int = 0                      # Objetos actualmente en pantalla
    leads: int = 0                              # Tracks con dwell >= DWELL_LEAD_SECONDS
    bounced: int = 0                            # Tracks con dwell < DWELL_LEAD_SECONDS (ya salieron)
    class_counts: dict = field(default_factory=dict)   # { "Automovil": 5, "Peaton": 8, ... }
    fps_processing: float = 0.0                 # FPS de inferencia (para monitoreo)
    timestamp: float = field(default_factory=time.time)

    # ── MÉTRICAS EXCLUSIVAS PARA VEHÍCULOS ─────────────────────────────────
    vehicles_seen: int = 0                      # Total único de vehículos vistos por la cámara (detección en encuadre)
    vehicles_crossed: int = 0                   # Total único de vehículos que cruzaron al menos una línea
    vehicles_crossing_rate: float = 0.0         # % de vehículos vistos que cruzaron una línea
    vehicles_in: int = 0
    vehicles_out: int = 0
    vehicles_active: int = 0
    vehicles_leads: int = 0
    vehicles_bounced: int = 0
    vehicle_class_counts: dict = field(default_factory=dict)

    # ── MÉTRICAS EXCLUSIVAS PARA PEATONES / TRANSEÚNTES ───────────────────
    pedestrians_seen: int = 0                   # Total único de peatones vistos por la cámara (detección en encuadre)
    pedestrians_crossed: int = 0                # Total único de peatones que cruzaron al menos una línea
    pedestrians_crossing_rate: float = 0.0      # % de peatones vistos que cruzaron una línea
    pedestrians_in: int = 0
    pedestrians_out: int = 0
    pedestrians_active: int = 0
    pedestrians_leads: int = 0
    pedestrians_bounced: int = 0
    pedestrians_dwell_avg: float = 0.0

    # ── TOTALES CONSOLIDADOS ──────────────────────────────────────────────
    total_seen: int = 0                         # Total único de objetos detectados (vehículos + peatones)

    # ── DESGLOSE POR LÍNEAS DE CONTEO INDIVIDUALES ────────────────────────
    lines: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serializa las métricas a un diccionario JSON-compatible con subpaneles separados y desglose de líneas."""
        return {
            "cam_id": self.cam_id,
            "total_seen": self.total_seen,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "total_flow": self.total_in + self.total_out,
            "active_tracks": self.active_tracks,
            "leads": self.leads,
            "bounced": self.bounced,
            "class_counts": self.class_counts,
            "fps_processing": round(self.fps_processing, 1),
            "timestamp": self.timestamp,
            # Desglose de cada línea de cruce activa
            "lines": self.lines,
            # Subpanel vehicular especializado
            "vehicles": {
                "seen": self.vehicles_seen,
                "crossed": self.vehicles_crossed,
                "crossing_rate": round(self.vehicles_crossing_rate, 1),
                "in": self.vehicles_in,
                "out": self.vehicles_out,
                "flow": self.vehicles_in + self.vehicles_out,
                "active": self.vehicles_active,
                "leads": self.vehicles_leads,
                "bounced": self.vehicles_bounced,
                "class_counts": self.vehicle_class_counts,
            },
            # Subpanel peatonal especializado
            "pedestrians": {
                "seen": self.pedestrians_seen,
                "crossed": self.pedestrians_crossed,
                "crossing_rate": round(self.pedestrians_crossing_rate, 1),
                "in": self.pedestrians_in,
                "out": self.pedestrians_out,
                "flow": self.pedestrians_in + self.pedestrians_out,
                "active": self.pedestrians_active,
                "leads": self.pedestrians_leads,
                "bounced": self.pedestrians_bounced,
                "dwell_avg": round(self.pedestrians_dwell_avg, 1),
            }
        }


@dataclass
class CountingLine:
    """Representa una línea de conteo activa y configurable dentro del encuadre de la cámara."""
    id: str
    name: str
    coords: tuple[tuple[int, int], tuple[int, int]]
    target_entity: str = "ALL"  # "ALL", "VEHICLE", "PEDESTRIAN"
    line_zone: Optional[Any] = None
    in_offset: int = 0
    out_offset: int = 0
    vehicles_in: int = 0
    vehicles_out: int = 0
    pedestrians_in: int = 0
    pedestrians_out: int = 0

    def to_dict(self) -> dict:
        if self.target_entity == "PEDESTRIAN":
            in_c = self.pedestrians_in + self.in_offset
            out_c = self.pedestrians_out + self.out_offset
        elif self.target_entity == "VEHICLE":
            in_c = self.vehicles_in + self.in_offset
            out_c = self.vehicles_out + self.out_offset
        else:
            in_c = self.vehicles_in + self.pedestrians_in + self.in_offset
            out_c = self.vehicles_out + self.pedestrians_out + self.out_offset
        return {
            "id": self.id,
            "name": self.name,
            "target_entity": self.target_entity,
            "coords": [list(self.coords[0]), list(self.coords[1])],
            "in": in_c,
            "out": out_c,
            "total": in_c + out_c,
            "vehicles_in": self.vehicles_in,
            "vehicles_out": self.vehicles_out,
            "pedestrians_in": self.pedestrians_in,
            "pedestrians_out": self.pedestrians_out,
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
        self._vehicles_finished_tracks: set[int] = set()
        self._pedestrians_finished_tracks: set[int] = set()

        # Conjuntos de IDs únicos para métrica dual en sesión activa: Vistos en Cámara vs Cruzaron Línea
        self._seen_vehicles: set[int] = set()
        self._seen_pedestrians: set[int] = set()
        self._crossed_vehicles: set[int] = set()
        self._crossed_pedestrians: set[int] = set()

        # Cargar conteos previos acumulados de hoy desde la base de datos
        try:
            today_data = database.get_today_counts(cam_id)
            self._initial_seen_veh: int = today_data.get("vehicles_seen", 0)
            self._initial_seen_ped: int = today_data.get("pedestrians_seen", 0)
            self._initial_crossed_veh: int = today_data.get("vehicles_crossed", 0)
            self._initial_crossed_ped: int = today_data.get("pedestrians_crossed", 0)
            self._vehicles_in_count: int = today_data.get("vehicles_in", 0)
            self._vehicles_out_count: int = today_data.get("vehicles_out", 0)
            self._pedestrians_in_count: int = today_data.get("pedestrians_in", 0)
            self._pedestrians_out_count: int = today_data.get("pedestrians_out", 0)
        except Exception as e:
            logger.warning(f"[{cam_id}] No se pudieron cargar conteos previos de hoy: {e}")
            self._initial_seen_veh = 0
            self._initial_seen_ped = 0
            self._initial_crossed_veh = 0
            self._initial_crossed_ped = 0
            self._vehicles_in_count = 0
            self._vehicles_out_count = 0
            self._pedestrians_in_count = 0
            self._pedestrians_out_count = 0

        self._vehicles_leads: int = 0
        self._vehicles_bounced: int = 0
        self._pedestrians_leads: int = 0
        self._pedestrians_bounced: int = 0

        # Conteo de clase acumulado para objetos que cruzaron la línea
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

        # ── Inicializar Líneas de Conteo Multi-Zona ──────────────────────────
        raw_lines = getattr(config, "MULTI_COUNTING_LINES", {}).get(cam_id)
        if not raw_lines:
            raw_lines = config.COUNTING_LINES.get(cam_id)
        norm_lines = config.normalize_counting_lines(cam_id, raw_lines)

        today_lines_data = {}
        try:
            today_lines_data = database.get_today_line_counts(cam_id)
        except Exception as e:
            logger.warning(f"[{cam_id}] No se pudieron cargar conteos por línea previos de hoy: {e}")

        self.counting_lines: dict[str, CountingLine] = {}
        for l_cfg in norm_lines:
            lid = l_cfg["id"]
            coords = l_cfg["coords"]
            lz = sv.LineZone(
                start=sv.Point(*coords[0]),
                end=sv.Point(*coords[1]),
                triggering_anchors=[sv.Position.CENTER]
            )
            ld = today_lines_data.get(lid, {})
            self.counting_lines[lid] = CountingLine(
                id=lid,
                name=l_cfg["name"],
                coords=coords,
                target_entity=l_cfg["target_entity"],
                line_zone=lz,
                vehicles_in=ld.get("vehicles_in", 0),
                vehicles_out=ld.get("vehicles_out", 0),
                pedestrians_in=ld.get("pedestrians_in", 0),
                pedestrians_out=ld.get("pedestrians_out", 0)
            )

        first_line = next(iter(self.counting_lines.values()))
        self._line_zone = first_line.line_zone
        self._in_offset: int = 0
        self._out_offset: int = 0
        self.metrics.lines = [l.to_dict() for l in self.counting_lines.values()]

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
            classes=config.TARGET_CLASS_IDS,
            device=config.INFERENCE_DEVICE,
            verbose=False,
            persist=True,
            tracker="bytetrack.yaml"
        )[0]  # Tomar el primer resultado (batch_size=1)

        # Convertir resultados YOLO → formato supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Filtrar solo las clases objetivo configuradas
        if detections.class_id is not None:
            mask = np.isin(detections.class_id, config.TARGET_CLASS_IDS)
            detections = detections[mask]

        # ── 3. Registro de primer avistamiento y clase por track ─────────────
        current_time = time.time()
        if detections.tracker_id is not None:
            for i, track_id in enumerate(detections.tracker_id):
                track_id = int(track_id)

                # Registrar primera vez que se ve este track
                if track_id not in self._track_first_seen:
                    self._track_first_seen[track_id] = current_time

                # Registrar la clase detectada
                if track_id not in self._track_class and detections.class_id is not None:
                    class_id = int(detections.class_id[i])
                    self._track_class[track_id] = config.CLASS_LABELS.get(class_id, "Desconocido")

                # Registrar en objetos vistos por la cámara (Métrica 1: Detección en encuadre)
                v_type = self._track_class.get(track_id, "Automovil")
                ent = config.ENTITY_TYPE_MAP.get(v_type, "VEHICLE")
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.85
                if ent == "PEDESTRIAN":
                    if track_id not in self._seen_pedestrians:
                        self._seen_pedestrians.add(track_id)
                        try:
                            database.record_sighting(self.cam_id, track_id, v_type, "PEDESTRIAN", conf)
                        except Exception:
                            pass
                else:
                    if track_id not in self._seen_vehicles:
                        self._seen_vehicles.add(track_id)
                        try:
                            database.record_sighting(self.cam_id, track_id, v_type, "VEHICLE", conf)
                        except Exception:
                            pass

        # ── 4. Conteo Multi-LineZone y Registro de Eventos ──────────────────
        for line_item in self.counting_lines.values():
            if detections.tracker_id is not None and len(detections) > 0:
                if line_item.target_entity == "VEHICLE":
                    mask = np.array([
                        config.ENTITY_TYPE_MAP.get(self._track_class.get(int(tid), "Automovil"), "VEHICLE") == "VEHICLE"
                        for tid in detections.tracker_id
                    ], dtype=bool)
                    line_dets = detections[mask]
                elif line_item.target_entity == "PEDESTRIAN":
                    mask = np.array([
                        config.ENTITY_TYPE_MAP.get(self._track_class.get(int(tid), "Automovil"), "VEHICLE") == "PEDESTRIAN"
                        for tid in detections.tracker_id
                    ], dtype=bool)
                    line_dets = detections[mask]
                else:
                    line_dets = detections
            else:
                line_dets = sv.Detections.empty()

            if len(line_dets) == 0 or line_dets.tracker_id is None:
                # Evict stale tracking state when no detections are active
                empty_d = sv.Detections.empty()
                empty_d.tracker_id = np.array([], dtype=int)
                line_item.line_zone.trigger(empty_d)
                continue

            crossed_in, crossed_out = line_item.line_zone.trigger(line_dets)
            for i, tid in enumerate(line_dets.tracker_id):
                tid = int(tid)
                v_type = self._track_class.get(tid, "Automovil")
                entity_type = config.ENTITY_TYPE_MAP.get(v_type, "VEHICLE")
                conf = float(line_dets.confidence[i]) if line_dets.confidence is not None else 0.85
                dwell = current_time - self._track_first_seen.get(tid, current_time)

                if crossed_in[i]:
                    if entity_type == "PEDESTRIAN":
                        line_item.pedestrians_in += 1
                        self._pedestrians_in_count += 1
                        self._crossed_pedestrians.add(tid)
                    else:
                        line_item.vehicles_in += 1
                        self._vehicles_in_count += 1
                        self._crossed_vehicles.add(tid)
                    try:
                        database.mark_sighting_crossed(self.cam_id, tid, line_item.id)
                        database.record_event(
                            camera_id=self.cam_id,
                            track_id=tid,
                            vehicle_type=v_type,
                            direction="IN",
                            confidence=conf,
                            dwell_time=dwell,
                            entity_type=entity_type,
                            line_id=line_item.id
                        )
                    except Exception as e:
                        logger.error(f"[{self.cam_id}] Error guardando evento IN ({entity_type}) en línea {line_item.id}: {e}")

                elif crossed_out[i]:
                    if entity_type == "PEDESTRIAN":
                        line_item.pedestrians_out += 1
                        self._pedestrians_out_count += 1
                        self._crossed_pedestrians.add(tid)
                    else:
                        line_item.vehicles_out += 1
                        self._vehicles_out_count += 1
                        self._crossed_vehicles.add(tid)
                    try:
                        database.mark_sighting_crossed(self.cam_id, tid, line_item.id)
                        database.record_event(
                            camera_id=self.cam_id,
                            track_id=tid,
                            vehicle_type=v_type,
                            direction="OUT",
                            confidence=conf,
                            dwell_time=dwell,
                            entity_type=entity_type,
                            line_id=line_item.id
                        )
                    except Exception as e:
                        logger.error(f"[{self.cam_id}] Error guardando evento OUT ({entity_type}) en línea {line_item.id}: {e}")

        # Totales acumulados: Métrica 1 (Vistos en Cámara) vs Métrica 2 (Cruces por Líneas)
        tot_seen_veh = self._initial_seen_veh + len(self._seen_vehicles)
        tot_seen_ped = self._initial_seen_ped + len(self._seen_pedestrians)
        tot_crossed_veh = self._initial_crossed_veh + len(self._crossed_vehicles)
        tot_crossed_ped = self._initial_crossed_ped + len(self._crossed_pedestrians)

        self.metrics.total_seen = tot_seen_veh + tot_seen_ped
        self.metrics.vehicles_seen = tot_seen_veh
        self.metrics.pedestrians_seen = tot_seen_ped
        self.metrics.vehicles_crossed = tot_crossed_veh
        self.metrics.pedestrians_crossed = tot_crossed_ped
        self.metrics.vehicles_crossing_rate = (tot_crossed_veh / max(tot_seen_veh, 1)) * 100.0
        self.metrics.pedestrians_crossing_rate = (tot_crossed_ped / max(tot_seen_ped, 1)) * 100.0

        self.metrics.total_in = self._vehicles_in_count + self._pedestrians_in_count
        self.metrics.total_out = self._vehicles_out_count + self._pedestrians_out_count
        self.metrics.vehicles_in = self._vehicles_in_count
        self.metrics.vehicles_out = self._vehicles_out_count
        self.metrics.pedestrians_in = self._pedestrians_in_count
        self.metrics.pedestrians_out = self._pedestrians_out_count
        self.metrics.lines = [l.to_dict() for l in self.counting_lines.values()]

        # ── 5. Cálculo dwell time y clasificación Lead/Bounced ───────────────
        active_ids = set(int(tid) for tid in detections.tracker_id) if detections.tracker_id is not None else set()
        self._update_dwell_metrics(active_ids, current_time)

        # ── 6. Conteo de clases activas y desglose por entidad ────────────────
        active_class_counts: dict[str, int] = defaultdict(int)
        vehicle_class_counts: dict[str, int] = defaultdict(int)
        pedestrian_dwells = []
        active_veh = 0
        active_ped = 0

        if detections.tracker_id is not None and detections.class_id is not None:
            for i in range(len(detections)):
                class_id = int(detections.class_id[i])
                label = config.CLASS_LABELS.get(class_id, "Desconocido")
                active_class_counts[label] += 1
                ent = config.ENTITY_TYPE_MAP.get(label, "VEHICLE")
                if ent == "PEDESTRIAN":
                    active_ped += 1
                    tid = int(detections.tracker_id[i])
                    pedestrian_dwells.append(current_time - self._track_first_seen.get(tid, current_time))
                else:
                    active_veh += 1
                    vehicle_class_counts[label] += 1

        self.metrics.class_counts = dict(active_class_counts)
        self.metrics.vehicle_class_counts = dict(vehicle_class_counts)
        self.metrics.vehicles_active = active_veh
        self.metrics.pedestrians_active = active_ped
        self.metrics.pedestrians_dwell_avg = (sum(pedestrian_dwells) / len(pedestrian_dwells)) if pedestrian_dwells else 0.0
        self.metrics.vehicles_leads = self._vehicles_leads
        self.metrics.vehicles_bounced = self._vehicles_bounced
        self.metrics.pedestrians_leads = self._pedestrians_leads
        self.metrics.pedestrians_bounced = self._pedestrians_bounced

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
        clasifica los tracks que ya desaparecieron como Lead o Bounced,
        desglosando entre vehículos y peatones.
        """
        known_ids = set(self._track_first_seen.keys())
        disappeared_ids = known_ids - active_ids - self._finished_tracks

        for tid in disappeared_ids:
            dwell = current_time - self._track_first_seen[tid]
            clase = self._track_class.get(tid, "Automovil")
            ent = config.ENTITY_TYPE_MAP.get(clase, "VEHICLE")

            if dwell >= config.DWELL_LEAD_SECONDS:
                self.metrics.leads += 1
                if ent == "PEDESTRIAN":
                    self._pedestrians_leads += 1
                else:
                    self._vehicles_leads += 1
            else:
                self.metrics.bounced += 1
                if ent == "PEDESTRIAN":
                    self._pedestrians_bounced += 1
                else:
                    self._vehicles_bounced += 1

            self._finished_tracks.add(tid)

    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Dibuja bounding boxes, etiquetas, trazas y línea de conteo en el frame.
        Diferencia visualmente peatones y vehículos.
        """
        labels = []
        if detections.tracker_id is not None:
            current_time = time.time()
            for i, track_id in enumerate(detections.tracker_id):
                track_id = int(track_id)
                clase = self._track_class.get(track_id, "?")
                ent = config.ENTITY_TYPE_MAP.get(clase, "VEHICLE")
                dwell = current_time - self._track_first_seen.get(track_id, current_time)
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                icon = "🚶" if ent == "PEDESTRIAN" else "🚗"
                is_crossed = (track_id in self._crossed_pedestrians) if ent == "PEDESTRIAN" else (track_id in self._crossed_vehicles)
                status_tag = " [CRUZÓ]" if is_crossed else " [VISTO]"
                etiqueta = f"{icon} #{track_id} {clase}{status_tag} {dwell:.1f}s [{conf:.0%}]"
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

        # Dibujar cada una de las líneas de conteo configuradas con su código de color y conteos
        for line in self.counting_lines.values():
            p1 = (int(line.coords[0][0]), int(line.coords[0][1]))
            p2 = (int(line.coords[1][0]), int(line.coords[1][1]))

            # Colores BGR según propósito:
            # Púrpura (234, 85, 168) para peatones, Naranja (0, 158, 245) para vehículos, Esmeralda (129, 185, 16) para mixto
            if line.target_entity == "PEDESTRIAN":
                line_color = (234, 85, 168)      # Púrpura/Magenta
                tag_icon = "PEATONES"
            elif line.target_entity == "VEHICLE":
                line_color = (0, 158, 245)       # Naranja/Ámbar
                tag_icon = "VEHICULOS"
            else:
                line_color = (129, 185, 16)      # Esmeralda
                tag_icon = "MIXTO"

            # Línea de sombra de contraste y línea principal
            cv2.line(frame, p1, p2, (15, 23, 42), 6, cv2.LINE_AA)
            cv2.line(frame, p1, p2, line_color, 3, cv2.LINE_AA)

            # Puntos extremos con halo (IN = Verde, OUT = Rojo)
            cv2.circle(frame, p1, 7, (16, 185, 129), -1)
            cv2.circle(frame, p1, 9, (255, 255, 255), 1)
            cv2.circle(frame, p2, 7, (59, 68, 239), -1)
            cv2.circle(frame, p2, 9, (255, 255, 255), 1)

            # Cartela flotante central con nombre y conteos
            mx = (p1[0] + p2[0]) // 2
            my = (p1[1] + p2[1]) // 2

            if line.target_entity == "PEDESTRIAN":
                in_c = line.pedestrians_in
                out_c = line.pedestrians_out
            elif line.target_entity == "VEHICLE":
                in_c = line.vehicles_in
                out_c = line.vehicles_out
            else:
                in_c = line.vehicles_in + line.pedestrians_in
                out_c = line.vehicles_out + line.pedestrians_out

            badge_text = f"[{tag_icon}] {line.name}: IN {in_c} | OUT {out_c}"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            bx1, by1 = mx - tw // 2 - 5, my - th - 6
            bx2, by2 = mx + tw // 2 + 5, my + 5
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (15, 23, 42), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), line_color, 1)
            cv2.putText(frame, badge_text, (bx1 + 5, my - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ── HUD de estado en la esquina superior izquierda (Doble Métrica) ───
        veh_crossed = self.metrics.vehicles_in + self.metrics.vehicles_out
        ped_crossed = self.metrics.pedestrians_in + self.metrics.pedestrians_out
        overlay_lines = [
            f"CANAL: {self.cam_id} | FPS: {self.metrics.fps_processing:.1f}",
            f"AUTOS/VEHICULOS -> [M1 Vistos: {self.metrics.vehicles_seen}] | [M2 Cruzaron: {veh_crossed}] ({self.metrics.vehicles_crossing_rate:.0f}%) [IN:{self.metrics.vehicles_in} OUT:{self.metrics.vehicles_out}]",
            f"PERSONAS/PEAT.  -> [M1 Vistos: {self.metrics.pedestrians_seen}] | [M2 Cruzaron: {ped_crossed}] ({self.metrics.pedestrians_crossing_rate:.0f}%) [IN:{self.metrics.pedestrians_in} OUT:{self.metrics.pedestrians_out}]",
            f"Delimitadores: {len(self.counting_lines)} lineas | En pantalla: {self.metrics.vehicles_active} veh, {self.metrics.pedestrians_active} peat",
        ]
        y_offset = 26
        for line_txt in overlay_lines:
            cv2.putText(
                frame, line_txt,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 0),    # Verde
                2,
                cv2.LINE_AA,
            )
            y_offset += 24

        return frame

    def update_lines(self, lines_config: list[dict]) -> None:
        """
        Actualiza todas las líneas de conteo de la cámara preservando conteos existentes.
        Soporta añadir, renombrar, cambiar coordenadas, cambiar entidad y eliminar líneas.
        """
        import supervision as sv
        norm_lines = config.normalize_counting_lines(self.cam_id, lines_config)
        new_lines_dict = {}

        for l_cfg in norm_lines:
            lid = l_cfg["id"]
            coords = l_cfg["coords"]
            lz = sv.LineZone(
                start=sv.Point(*coords[0]),
                end=sv.Point(*coords[1]),
                triggering_anchors=[sv.Position.CENTER]
            )
            if lid in self.counting_lines:
                prev = self.counting_lines[lid]
                new_line = CountingLine(
                    id=lid,
                    name=l_cfg["name"],
                    coords=coords,
                    target_entity=l_cfg["target_entity"],
                    line_zone=lz,
                    in_offset=prev.in_offset,
                    out_offset=prev.out_offset,
                    vehicles_in=prev.vehicles_in,
                    vehicles_out=prev.vehicles_out,
                    pedestrians_in=prev.pedestrians_in,
                    pedestrians_out=prev.pedestrians_out
                )
            else:
                new_line = CountingLine(
                    id=lid,
                    name=l_cfg["name"],
                    coords=coords,
                    target_entity=l_cfg["target_entity"],
                    line_zone=lz
                )
            new_lines_dict[lid] = new_line

        self.counting_lines = new_lines_dict
        first_line = next(iter(self.counting_lines.values()))
        self._line_zone = first_line.line_zone
        self.metrics.lines = [l.to_dict() for l in self.counting_lines.values()]
        logger.info(f"[{self.cam_id}] Actualizadas {len(self.counting_lines)} líneas de conteo.")

    def update_line(self, line_coords: tuple[tuple[int, int], tuple[int, int]]) -> None:
        """Compatibilidad hacia atrás: actualiza las coordenadas de la primera línea."""
        import supervision as sv
        if self.counting_lines:
            first_key = next(iter(self.counting_lines.keys()))
            first_line = self.counting_lines[first_key]
            first_line.coords = line_coords
            first_line.line_zone = sv.LineZone(
                start=sv.Point(*line_coords[0]),
                end=sv.Point(*line_coords[1]),
                triggering_anchors=[sv.Position.CENTER]
            )
            self._line_zone = first_line.line_zone
            self.metrics.lines = [l.to_dict() for l in self.counting_lines.values()]
            logger.info(f"[{self.cam_id}] Primera línea '{first_key}' actualizada a {line_coords}.")

    def reset_counts(self) -> None:
        """Reinicia todos los contadores de la cámara y de todas sus líneas configuradas."""
        import supervision as sv
        self.metrics = VehicleMetrics(cam_id=self.cam_id)
        self._track_first_seen.clear()
        self._track_class.clear()
        self._finished_tracks.clear()
        self._vehicles_finished_tracks.clear()
        self._pedestrians_finished_tracks.clear()
        self._class_counts.clear()
        self._vehicles_in_count = 0
        self._vehicles_out_count = 0
        self._pedestrians_in_count = 0
        self._pedestrians_out_count = 0
        self._vehicles_leads = 0
        self._vehicles_bounced = 0
        self._pedestrians_leads = 0
        self._pedestrians_bounced = 0

        self._seen_vehicles.clear()
        self._seen_pedestrians.clear()
        self._crossed_vehicles.clear()
        self._crossed_pedestrians.clear()
        self._initial_seen_veh = 0
        self._initial_seen_ped = 0
        self._initial_crossed_veh = 0
        self._initial_crossed_ped = 0

        for line in self.counting_lines.values():
            line.in_offset = 0
            line.out_offset = 0
            line.vehicles_in = 0
            line.vehicles_out = 0
            line.pedestrians_in = 0
            line.pedestrians_out = 0
            start = sv.Point(*line.coords[0])
            end = sv.Point(*line.coords[1])
            line.line_zone = sv.LineZone(
                start=start,
                end=end,
                triggering_anchors=[sv.Position.CENTER]
            )

        first_line = next(iter(self.counting_lines.values()))
        self._line_zone = first_line.line_zone
        self.metrics.lines = [l.to_dict() for l in self.counting_lines.values()]
        logger.info(f"[{self.cam_id}] Contadores y todas las líneas reiniciados.")
