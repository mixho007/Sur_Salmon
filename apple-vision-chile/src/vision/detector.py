"""Detección + tracking YOLO de manzanas y generación del CSV de métricas.

Ejecutar primero este script para generar data/02_processed/detecciones.csv:
    python -m src.vision.detector --modelo models/best.pt --video ruta.mp4
Sin --modelo se usa el nombre por defecto (YOLO26 nano, preentrenado COCO).
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.config import RAIZ, Config, cargar_config
from src.vision.calibrador import px_a_mm
from src.vision.clasificador import COLORES_BGR, GOOD, DETALLE, NO_PASA, clasificar

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


def dibujar_panel(frame: np.ndarray, conteo: dict, cfg: Config) -> None:
    """Dibuja el panel semáforo (Buena/Con detalle/No pasa) en la esquina superior derecha."""
    filas = [
        ("Buena", GOOD, COLORES_BGR[GOOD]),
        ("Con detalle", DETALLE, COLORES_BGR[DETALLE]),
        ("No pasa", NO_PASA, COLORES_BGR[NO_PASA]),
    ]
    total = sum(conteo.get(clave, 0) for _, clave, _ in filas)

    ancho_panel, alto_fila, margen = 210, 26, 10
    alto_panel = alto_fila * (len(filas) + 1) + margen
    alto_frame, ancho_frame = frame.shape[:2]
    x1 = ancho_frame - ancho_panel - margen
    y1 = margen

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + ancho_panel, y1 + alto_panel), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x1 + ancho_panel, y1 + alto_panel), (255, 255, 255), 1)

    cv2.putText(frame, f"Total: {total}", (x1 + 10, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    for i, (etiqueta, clave, color) in enumerate(filas, start=1):
        y = y1 + alto_fila * i + 20
        cv2.rectangle(frame, (x1 + 10, y - 14), (x1 + 24, y), color, -1)
        cv2.putText(frame, f"{etiqueta}: {conteo.get(clave, 0)}", (x1 + 32, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def procesar(fuente, modelo: str, cfg: Config, mostrar: bool = True,
             csv_salida: Path = CSV_SALIDA) -> None:
    yolo = YOLO(modelo)
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")

    os.makedirs(csv_salida.parent, exist_ok=True)
    categoria_por_id = {}  # última categoría vista por ID -> evita recontar
    n_frame = 0

    with open(csv_salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(COLUMNAS)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frame += 1

            res = yolo.track(frame, persist=True, imgsz=cfg.imgsz, verbose=False)[0]
            if res.boxes is not None and res.boxes.id is not None:
                for caja, tid, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                          res.boxes.id.int().cpu().tolist(),
                                          res.boxes.cls.int().cpu().tolist()):
                    x1, y1, x2, y2 = [int(v) for v in caja]
                    x1, y1 = max(x1, 0), max(y1, 0)
                    ancho, alto = x2 - x1, y2 - y1
                    if ancho <= 0 or alto <= 0:
                        continue
                    hsv = hsv_promedio(frame, x1, y1, x2, y2)
                    if hsv is None:
                        continue
                    hue, sat, val = hsv

                    area_px = ancho * alto  # área del bbox (modelo de detección, sin máscara)
                    diametro_mm = px_a_mm((ancho + alto) / 2, cfg.px_por_mm)
                    aspecto = min(ancho, alto) / max(ancho, alto)
                    nombre = yolo.names[cls]

                    escritor.writerow([tid, n_frame, area_px, round(diametro_mm, 2),
                                       round(hue, 2), round(sat, 2), round(val, 2),
                                       round(aspecto, 4), nombre])

                    categoria = clasificar(diametro_mm, hue, sat, aspecto, nombre, cfg)
                    categoria_por_id[tid] = categoria
                    color = COLORES_BGR[categoria]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"#{tid} {diametro_mm:.0f}mm", (x1, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if mostrar:
                conteo = defaultdict(int)
                for c in categoria_por_id.values():
                    conteo[c] += 1
                dibujar_panel(frame, conteo, cfg)
                cv2.imshow("AppleVision Chile (q para salir)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"CSV guardado en {csv_salida} | manzanas únicas: {len(categoria_por_id)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detección y conteo de manzanas con YOLO")
    parser.add_argument("--modelo", default=MODELO_DEFAULT)
    parser.add_argument("--video", default=None, help="Ruta de video; por defecto config.yaml")
    parser.add_argument("--sin-ventana", action="store_true")
    args = parser.parse_args()

    cfg = cargar_config()
    fuente = args.video if args.video is not None else cfg.fuente_video
    procesar(fuente, args.modelo, cfg, mostrar=not args.sin_ventana)


if __name__ == "__main__":
    main()
