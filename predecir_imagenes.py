#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predice la clase de una pieza a partir de una imagen.
Uso:
  python predecir_imagenes.py ruta/a/imagen.jpg
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import cv2

# Importá el pipeline modular
from procesado_img import procesar_imagen_completa
# arriba
from momentos import features_from_mask, ensure_binary



# ===============================
# Parámetros / defaults
# ===============================
MODEL_PATH_DEFAULT = Path("models/kmeans_puro.npz")
EPS = 1e-12
DETECT_DOUBLE_LOG_DEFAULT = True
DOUBLE_LOG_Z_THRESHOLD = 8.0  # si |z_hu| > 8, sospechamos
# ===============================
# Utilidades numéricas
# ===============================
def hu_log(x: float | np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return -np.sign(x) * np.log10(np.abs(x) + EPS)

def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(std == 0.0, 1.0, std)
    return (x - mean) / std_safe

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

# ===============================
# Modelo
# ===============================

def cargar_modelo(model_path: Path) -> Dict[str, np.ndarray]:
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    data = np.load(model_path, allow_pickle=True)
    centroids = data["centroids"]           # (K,3) en z-score
    mean      = data["mean"].astype(float)  # (3,)
    std       = data["std"].astype(float)   # (3,)
    kept_cols = list(data["kept_cols"])
    label_ord = list(data["label_order"])
    exp = ['hu1','hu2','ar2']
    if kept_cols != exp:
        raise ValueError(f"kept_cols en npz = {kept_cols} (se esperaba {exp})")
    return dict(centroids=centroids, mean=mean, std=std, labels=label_ord)

# ===============================
# Features sobre máscara 0/255 (ALINEADA+RECORTADA)
# ===============================
# reemplazá tu extraer_features por este
def extraer_features(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    # 1) Binarización idéntica a momentos.py/CSV
    mask_bin = ensure_binary(mask)

    # 2) Features “CSV”: hu6 (con eps=1e-30) + ar2
    res = features_from_mask(mask_bin)
    if res is None:
        raise ValueError("No se detectó contorno principal en la máscara.")
    hu6, circ, roundness, ar, ar2, n_lados = res

    # 3) Transformación runtime: segundo log a Hu (eps=1e-12) tal como Mi_Kmeans.py
    feats = kmeans_runtime_transform(hu6[0], hu6[1], ar2)  # -> [hu1_log, hu2_log, ar2]

    # 4) Debug info
    H, W = mask_bin.shape[:2]
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_c = int(cv2.contourArea(max(cnts, key=cv2.contourArea))) if cnts else 0
    dbg = {
        'bbox_w': int(W),
        'bbox_h': int(H),
        'area':   area_c,
        'hu1_csv': float(hu6[0]),
        'hu2_csv': float(hu6[1]),
    }
    return feats, dbg

# cerca de utilidades numéricas
def kmeans_runtime_transform(hu1_csv: float, hu2_csv: float, ar2: float, eps: float = 1e-12) -> np.ndarray:
    """Replica el paso de Mi_Kmeans.py:
       hu_log = -sign(hu_csv) * log10(|hu_csv| + 1e-12)  (segundo log)
       Devuelve [hu1_log, hu2_log, ar2] para runtime.
    """
    def hu_log_eps(x, eps_):
        x = float(x)
        return -np.sign(x) * np.log10(abs(x) + eps_)
    h1 = hu_log_eps(hu1_csv, eps)
    h2 = hu_log_eps(hu2_csv, eps)
    return np.array([h1, h2, float(ar2)], dtype=float)





# ===============================
# Predicción
# ===============================
def predecir(
    feats: np.ndarray,
    model: dict,
    compat_double_log: bool = DETECT_DOUBLE_LOG_DEFAULT,
    z_threshold: float = DOUBLE_LOG_Z_THRESHOLD
) -> tuple[str, float, np.ndarray, str]:
    """
    Estima label por cercanía L2 al centroide en espacio z-score.
    Si z luce fuera de rango, prueba 're-log' de hu1/hu2 (compat).
    """
    C  = model["centroids"]
    mu = model["mean"]
    sd = model["std"]

    z = standardize(feats, mu, sd)
    dists = np.linalg.norm(C - z[None, :], axis=1)
    idx = int(np.argmin(dists))
    best = (idx, float(dists[idx]), z, "normal")

    if compat_double_log and (np.any(np.abs(z[:2]) > z_threshold)):
        feats2 = feats.copy()
        feats2[0] = float(hu_log(feats2[0]))
        feats2[1] = float(hu_log(feats2[1]))
        z2 = standardize(feats2, mu, sd)
        d2 = np.linalg.norm(C - z2[None, :], axis=1)
        idx2 = int(np.argmin(d2))
        if d2[idx2] + 1e-6 < best[1] * 0.6:  # mejora clara
            best = (idx2, float(d2[idx2]), z2, "double-log")

    label = model["labels"][best[0]]
    return label, best[1], best[2], best[3]

# ===============================
# Pipeline alto nivel
# ===============================
def predecir_desde_imagen(
    img_path: Path,
    model: dict,
    compat_double_log: bool = DETECT_DOUBLE_LOG_DEFAULT
) -> tuple[str, float, np.ndarray, np.ndarray, dict, str]:
    """
    - Ejecuta pipeline (procesado_core.procesar_imagen_completa)
    - Extrae features
    - Predice

    Devuelve: (label, dist, z, mask, dbg, mode)
    """
    mask = procesar_imagen_completa(img_path)  # 0/255, alineada+recortada
    feats, dbg = extraer_features(mask)
    label, dist, z, mode = predecir(feats, model, compat_double_log)
    return label, dist, z, mask, dbg, mode

# ===============================
# CLI
# ===============================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clasifica una pieza y (opcional) compara la máscara con una guardada."
    )
    p.add_argument("image", type=Path, help="Ruta a la imagen original (jpg/png).")
    p.add_argument("--model", type=Path, default=MODEL_PATH_DEFAULT, help="Ruta al .npz del K-Means.")
    p.add_argument("--no-compat", action="store_true", help="Deshabilita la compatibilidad de re-log (double-log).")
    return p

# ... (imports y funciones iguales a tu último código)

def main():
    args = build_parser().parse_args()

    if not args.image.exists():
        print(f"❌ Imagen no encontrada: {args.image}")
        raise SystemExit(1)

    # 1) Procesar imagen y extraer features RUNTIME (hu1_log, hu2_log, ar2)
    print(f"📸 Procesando imagen: {args.image.name}")
    mask = procesar_imagen_completa(args.image)
    feats, _ = extraer_features(mask)   # feats = [hu1_log(1e-12), hu2_log(1e-12), ar2]

    # 2) Cargar modelo (centroides en z-score + mean/std del espacio log-transformado)
    try:
        model = cargar_modelo(args.model)
        print(f"✅ Modelo cargado: {len(model['labels'])} clases → {model['labels']}")
        print(f"   mean={np.round(model['mean'],4).tolist()}  std={np.round(model['std'],4).tolist()}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        raise SystemExit(1)

    # 3) Predicción directa desde feats (sin reprocesar imagen)
    try:
        label, dist, z, mode = predecir(feats, model, compat_double_log=not args.no_compat)
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        raise SystemExit(1)

    # 4) Reporte
    print("\n🎯 Predicción:", label)
    print(f"   Distancia al centroide: {dist:.4f}")
    if mode == "double-log":
        print("   [COMPAT] Detectado probable doble-log; se aplicó re-log en hu1/hu2 (runtime).")
        print("            Sugerencia: regenerar el .npz con un solo esquema de log consistente.")

    print("\n   Features RUNTIME (hu1_log, hu2_log, ar2):")
    print(f"     hu1 = {feats[0]:.4f}")
    print(f"     hu2 = {feats[1]:.4f}")
    print(f"     ar2 = {feats[2]:.4f}")

if __name__ == "__main__":
    main()