"""Detección + tracking YOLO de manzanas, conteo por línea y CSV de métricas.

Cada manzana recibe un ID (tracker ByteTrack) y se cuenta UNA sola vez, cuando su
centro cruza la línea de conteo. El panel en vivo muestra el total y la categoría.

    python -m src.vision.detector --modelo models/best.pt --video ruta.mp4
    python -m src.vision.detector --modelo models/best.pt --video ruta.mp4 --guardar salida.mp4 --sin-ventana

Sin --modelo se usa el nombre por defecto (YOLO26 nano, preentrenado COCO).
"""
import argparse
import csv
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.config import RAIZ, Config, cargar_config
from src.vision.calibrador import px_a_mm
from src.vision.clasificador import COLORES_BGR, clasificar
from src.vision.panel import dibujar_caja, dibujar_linea_conteo, dibujar_panel

MODELO_DEFAULT = "yolo26n.pt"
CSV_SALIDA = RAIZ / "data" / "02_processed" / "metricas.csv"
COLUMNAS = ["id", "frame", "area_px", "diametro_mm_est", "hue", "saturacion",
            "valor", "aspecto", "clase_yolo"]


def hsv_promedio(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """HSV medio dentro de una elipse inscrita en el bbox (reduce fondo en las esquinas)."""
    recorte = frame[y1:y2, x1:x2]
    if recorte.size == 0:
        return None
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    mascara = np.zeros(hsv.shape[:2], dtype=np.uint8)
    h, w = mascara.shape
    cv2.ellipse(mascara, (w // 2, h // 2), (max(w // 2, 1), max(h // 2, 1)), 0, 0, 360, 255, -1)
    # El matiz es circular (rojo ~0/179): media circular para no sesgar
    ang = hsv[..., 0][mascara > 0].astype(np.float64) * 2 * np.pi / 180
    hue = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 180 / (2 * np.pi)) % 180
    sat, val = cv2.mean(hsv, mask=mascara)[1:3]
    return float(hue), float(sat), float(val)


def procesar(fuente, modelo: str, cfg: Config, mostrar: bool = True,
             csv_salida: Path = CSV_SALIDA, lote: str = "Lote 1",
             linea: float = 0.8, guardar: str = None, conf: float = 0.25) -> dict:
    yolo = YOLO(modelo)
    # Cámara (índice entero): en Windows DirectShow abre más rápido; si falla, backend por defecto
    cap = cv2.VideoCapture(fuente, cv2.CAP_DSHOW) if isinstance(fuente, int) else cv2.VideoCapture(fuente)
    if isinstance(fuente, int) and not cap.isOpened():
        cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 24.0
    x_linea = int(ancho * linea)

    escritor_video = None
    if guardar:
        os.makedirs(os.path.dirname(os.path.abspath(guardar)), exist_ok=True)
        escritor_video = cv2.VideoWriter(guardar, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps_video, (ancho, alto))

    os.makedirs(csv_salida.parent, exist_ok=True)
    historial = defaultdict(list)   # id -> categorías vistas (se usa la más frecuente)
    lado_previo = {}                # id -> signo de (centro_x - x_linea) en el frame anterior
    contadas = {}                   # id -> categoría con la que se contó al cruzar
    n_frame, fps_suave, t_prev = 0, 0.0, time.perf_counter()

    with open(csv_salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(COLUMNAS)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frame += 1

            res = yolo.track(frame, persist=True, imgsz=cfg.imgsz, conf=conf,
                             tracker="bytetrack.yaml", verbose=False)[0]
            dibujos = []
            if res.boxes is not None and res.boxes.id is not None:
                for caja, tid, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                          res.boxes.id.int().cpu().tolist(),
                                          res.boxes.cls.int().cpu().tolist()):
                    x1, y1, x2, y2 = [int(v) for v in caja]
                    x1, y1 = max(x1, 0), max(y1, 0)
                    w_caja, h_caja = x2 - x1, y2 - y1
                    if w_caja <= 0 or h_caja <= 0:
                        continue
                    hsv = hsv_promedio(frame, x1, y1, x2, y2)
                    if hsv is None:
                        continue
                    hue, sat, val = hsv

                    area_px = w_caja * h_caja  # área del bbox (modelo de detección, sin máscara)
                    diametro_mm = px_a_mm((w_caja + h_caja) / 2, cfg.px_por_mm)
                    aspecto = min(w_caja, h_caja) / max(w_caja, h_caja)
                    nombre = yolo.names[cls]

                    escritor.writerow([tid, n_frame, area_px, round(diametro_mm, 2),
                                       round(hue, 2), round(sat, 2), round(val, 2),
                                       round(aspecto, 4), nombre])

                    categoria = clasificar(diametro_mm, hue, sat, aspecto, nombre, cfg)
                    historial[tid].append(categoria)
                    categoria_estable = Counter(historial[tid]).most_common(1)[0][0]

                    # Conteo: se cuenta una vez, cuando el centro cambia de lado de la línea
                    lado = 1 if (x1 + x2) / 2 >= x_linea else -1
                    if tid in lado_previo and lado != lado_previo[tid] and tid not in contadas:
                        contadas[tid] = categoria_estable
                    lado_previo[tid] = lado

                    dibujos.append(((x1, y1, x2, y2), tid, COLORES_BGR[categoria_estable]))

            ahora = time.perf_counter()
            inst = 1.0 / max(ahora - t_prev, 1e-6)
            fps_suave = inst if fps_suave == 0 else 0.9 * fps_suave + 0.1 * inst
            t_prev = ahora

            dibujar_linea_conteo(frame, x_linea)
            for caja, tid, color in dibujos:
                dibujar_caja(frame, caja, tid, color)
            conteo = Counter(contadas.values())
            dibujar_panel(frame, conteo, cfg, lote, fps_suave)

            if escritor_video is not None:
                escritor_video.write(frame)
            if mostrar:
                cv2.imshow("AppleVision Chile (q para salir)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if escritor_video is not None:
        escritor_video.release()
    cv2.destroyAllWindows()
    resumen = dict(Counter(contadas.values()))
    print(f"CSV guardado en {csv_salida}")
    print(f"IDs vistos: {len(historial)} | contadas al cruzar la línea: {len(contadas)} | por categoría: {resumen}")
    return {"ids_vistos": len(historial), "contadas": len(contadas), "categorias": resumen}


def main() -> None:
    parser = argparse.ArgumentParser(description="Detección y conteo de manzanas con YOLO")
    parser.add_argument("--modelo", default=MODELO_DEFAULT)
    parser.add_argument("--video", default=None, help="Ruta de video; por defecto config.yaml")
    parser.add_argument("--sin-ventana", action="store_true")
    parser.add_argument("--lote", default="Lote 1", help="Texto de la tarjeta «Lote activo»")
    parser.add_argument("--linea", type=float, default=0.8,
                        help="Posición de la línea de conteo, como fracción del ancho (0-1)")
    parser.add_argument("--guardar", default=None, help="Guardar el video con el panel (.mp4)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confianza mínima de detección")
    args = parser.parse_args()

    cfg = cargar_config()
    fuente = args.video if args.video is not None else cfg.fuente_video
    procesar(fuente, args.modelo, cfg, mostrar=not args.sin_ventana, lote=args.lote,
             linea=args.linea, guardar=args.guardar, conf=args.conf)


if __name__ == "__main__":
    main()
