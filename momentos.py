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

# Directorios con las imágenes ORIGINALES
ORIG_DIRS = {
    "Arandela": BASE / "Arandela",
    "Tuerca":   BASE / "Tuerca",
    "Tornillo": BASE / "Tornillo",
    "Clavo":    BASE / "Clavo",
}
OUT_CSV = BASE / "cualidades_imagenes.csv"

def list_images(indir: Path):
    pats = ["*.png","*.PNG","*.jpg","*.JPG","*.jpeg","*.JPEG"]
    files = []
    for p in pats: files.extend(indir.glob(p))
    unique = {f.stem.lower(): f for f in files}.values()
    return sorted(unique)

def find_original(label: str, stem: str):
    """Busca la imagen original que corresponde a una máscara por nombre (stem)."""
    odir = ORIG_DIRS[label]
    for ext in (".png",".jpg",".jpeg",".PNG",".JPG",".JPEG"):
        p = odir / f"{stem}{ext}"
        if p.exists(): return p
    # fallback: primer match por stem
    cands = list(odir.glob(stem + ".*"))
    return cands[0] if cands else None


def ensure_binary(img: np.ndarray):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Si el fondo quedó blanco (área blanca muy grande), invertimos
    if cv2.countNonZero(mask) > (mask.size // 2):
        mask = cv2.bitwise_not(mask)
    if cv2.countNonZero(mask) > int(0.90 * mask.size):
        mask = cv2.bitwise_not(mask)
    return mask

def hu_log6_from_mask(mask_bin: np.ndarray):
    m = cv2.moments(mask_bin, binaryImage=True)
    hu = cv2.HuMoments(m).flatten()
    eps = 1e-30
    return (-np.sign(hu) * np.log10(np.abs(hu) + eps))[:6]

# === NUEVO: contador de lados robusto sobre un contorno ===
def contar_lados_contorno(c: np.ndarray, eps_rel: float = 0.02) -> int:
    """
    Devuelve cantidad de lados de la envolvente convexa del contorno 'c'.
    0 => círculo/casi circular o indefinido.
    Hace 'snap' a 6 si el polígono cae en 5–7 (tuercas).
    """
    hull = cv2.convexHull(c)
    A = float(cv2.contourArea(hull))
    P = float(cv2.arcLength(hull, True))
    if P <= 1e-6 or A < 50:
        return 0

    circularidad = 4.0 * np.pi * A / (P * P)
    if circularidad > 0.92:
        return 0  # casi círculo → sin "lados" útiles

    approx = cv2.approxPolyDP(hull, eps_rel * P, True)

    # Fusionar vértices casi colineales para esquinas romas
    pts = [p[0] for p in approx]
    changed = True
    while changed and len(pts) >= 4:
        changed = False
        i = 0
        while i < len(pts):
            a = pts[(i - 1) % len(pts)]
            b = pts[i]
            d = pts[(i + 1) % len(pts)]
            v1 = (b - a).astype(np.float32)
            v2 = (d - b).astype(np.float32)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                pts.pop(i); changed = True; continue
            cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            if ang > 168.0:  # ~colineal
                pts.pop(i); changed = True
            else:
                i += 1

    n = len(pts)
    if 5 <= n <= 7:
        n = 6  # snap para tuercas
    return int(n)


def features_from_mask(mask_bin: np.ndarray):
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)

    area = float(cv2.contourArea(c))
    per  = float(cv2.arcLength(c, True)) or 1.0
    circ = 4.0 * np.pi * area / (per * per)

    (_, _), r = cv2.minEnclosingCircle(c)
    r = float(r) or 1.0
    roundness = area / (np.pi * r * r)

    x, y, w, h = cv2.boundingRect(c)
    ar = min(w, h) / max(w, h)

    # === NUEVO: dimensiones totales de la imagen ===
    h2, w2 = mask_bin.shape[:2]
    ar2 = min(w2, h2) / max(w2, h2)

     # === NUEVO: lados
    n_lados = contar_lados_contorno(c, eps_rel=0.02)

    hu6 = hu_log6_from_mask(mask_bin)
    return hu6.tolist(), circ, roundness, ar, ar2, n_lados

def texture_feats_from_original(orig_gray: np.ndarray, mask_bin: np.ndarray):
    """Textura interna: energía de gradiente y densidad de bordes dentro de la máscara."""
    if orig_gray.shape[:2] != mask_bin.shape[:2]:
        orig_gray = cv2.resize(orig_gray, (mask_bin.shape[1], mask_bin.shape[0]), interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(orig_gray, (3,3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # media del gradiente dentro de la pieza
    vals = mag[mask_bin > 0]
    grad_mean = float(vals.mean()) if vals.size else 0.0

    # densidad de bordes Canny dentro de la pieza
    canny = cv2.Canny(blur, 60, 180)
    edges_in = cv2.bitwise_and(canny, canny, mask=mask_bin)
    area = float(cv2.countNonZero(mask_bin)) + 1e-9
    edge_density = float(cv2.countNonZero(edges_in)) / area
    return grad_mean, edge_density

# --- Recorrido y escritura CSV ---
rows = []
for label, indir in IN_DIRS.items():
    paths = list_images(indir)
    if not paths:
        print(f"[AVISO] No hay imágenes en {indir}")
        continue

    print(f"Procesando clase '{label}' ({len(paths)} imágenes) ...")
    for p in paths:
        mask_img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            print(f"  [SKIP] No se pudo leer máscara: {p}")
            continue
        mask = ensure_binary(mask_img)

        feats = features_from_mask(mask)
        if feats is None:
            print(f"  [SKIP] Sin contornos: {p.name}")
            continue
        hu6, circ, roundness, ar, ar2, n_lados = feats

        # textura desde la imagen original
        orig_path = find_original(label, p.stem)
        if orig_path is not None:
            orig = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
            grad_mean, edge_density = texture_feats_from_original(orig, mask)
            edge_density = 0.0
        else:
            grad_mean, edge_density = (0.0, 0.0)
            print(f"  [WARN] No se encontró original para {p.name}")

        rows.append([p.name, label,
                     *[f"{v:.6f}" for v in hu6],
                     f"{circ:.6f}", f"{roundness:.6f}", f"{ar:.6f}", f"{ar2:.6f}",
                     str(n_lados), f"{grad_mean:.6f}", f"{edge_density:.6f}"])

# Encabezado y guardado
header = ["file","clase","hu1","hu2","hu3","hu4","hu5","hu6",
          "circularidad","redondez","aspect_ratio","ar2","n_lados",
          "grad_mean","edge_density"]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\nListo. CSV guardado en: {OUT_CSV} ({len(rows)} filas)")
