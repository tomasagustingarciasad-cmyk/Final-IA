import cv2
import numpy as np
from pathlib import Path
import csv

# === Directorios con MÁSCARAS (salidas del pipeline) ===
BASE = Path("base_datos")
IN_DIRS = {
    "Arandela": BASE / "ARANDELAS",
    "Tuerca":   BASE / "TUERCAS",
    "Tornillo": BASE / "TORNILLOS",
    "Clavo":    BASE / "CLAVOS",
}

OUT_CSV = BASE / "features_imagenes.csv"

def list_images(indir: Path):
    # Soporta .png/.jpg/.jpeg sin duplicar por mayúsculas
    pats = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    files = []
    for p in pats:
        files.extend(indir.glob(p))
    # dedup por nombre base case-insensitive
    unique = {f.stem.lower(): f for f in files}.values()
    return sorted(unique)

def hu_log6_from_mask(mask_bin: np.ndarray):
    # Hu moments sobre máscara (binaria 0/255), log-estables (primeros 6)
    m = cv2.moments(mask_bin, binaryImage=True)
    hu = cv2.HuMoments(m).flatten()  # 7 valores
    eps = 1e-30
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + eps)
    return hu_log[:6]

def features_from_mask(mask_bin: np.ndarray):
    # Contorno principal
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)

    area = float(cv2.contourArea(c))
    per  = float(cv2.arcLength(c, True)) or 1.0
    circ = 4.0 * np.pi * area / (per * per)

    # Redondez por círculo mínimo envolvente
    (_, _), r = cv2.minEnclosingCircle(c)
    r = float(r) or 1.0
    roundness = area / (np.pi * r * r)

    # Aspect ratio del bounding box axis-aligned
    x, y, w, h = cv2.boundingRect(c)
    ar = min(w, h) / max(w, h)

    # Hu (log) desde la máscara completa
    hu6 = hu_log6_from_mask(mask_bin)

    return hu6.tolist(), circ, roundness, ar

def ensure_binary(img: np.ndarray):
    # Asegura binario 0/255 (por si alguna máscara quedó en escala de grises)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Si está invertida (fondo blanco, pieza negra), invierto:
    if cv2.countNonZero(mask) < (mask.size // 2):
        mask = cv2.bitwise_not(mask)
    return mask

# --- Recorrido y escritura CSV ---
rows = []
for label, indir in IN_DIRS.items():
    paths = list_images(indir)
    if not paths:
        print(f"[AVISO] No hay imágenes en {indir}")
        continue

    print(f"Procesando clase '{label}' ({len(paths)} imágenes) ...")
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [SKIP] No se pudo leer: {p}")
            continue
        mask = ensure_binary(img)
        feats = features_from_mask(mask)
        if feats is None:
            print(f"  [SKIP] Sin contornos: {p.name}")
            continue

        hu6, circ, roundness, ar = feats
        rows.append([p.name, label, *[f"{v:.6f}" for v in hu6], f"{circ:.6f}", f"{roundness:.6f}", f"{ar:.6f}"])

# Encabezado y guardado
header = ["file", "clase", "hu1", "hu2", "hu3", "hu4", "hu5", "hu6",
          "circularidad", "redondez", "aspect_ratio"]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\nListo. CSV guardado en: {OUT_CSV} ({len(rows)} filas)")
