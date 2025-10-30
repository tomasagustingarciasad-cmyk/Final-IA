# procesado_img.py
import cv2
import numpy as np
from pathlib import Path

# === Parámetros base (igual que tu script) ===
RESIZE_W    = 640
TARGET_BG   = 215
R_IGNOREREL = 0.03
SIGMA_REL   = 0.15

# (input -> output)  NO CAMBIAR
pairs = [
    (Path("base_datos/Arandela"), Path("base_datos/ARANDELAS")),
    (Path("base_datos/Tuerca"),   Path("base_datos/TUERCAS")),
    (Path("base_datos/Tornillo"), Path("base_datos/TORNILLOS")),
    (Path("base_datos/Clavo"),    Path("base_datos/CLAVOS")),
]

# ---------------- utils ----------------
def list_images(indir: Path):
    files = list(indir.glob("*.[jJpP][pPnN][gG]"))  # .jpg .jpeg .png (cualquier combinación)
    unique = {f.stem.lower(): f for f in files}.values()
    return sorted(unique)

def _odd(n: int) -> int:
    """Devuelve n impar >= 3."""
    n = int(max(3, n))
    return n if (n % 2 == 1) else n + 1

def rel_kernel(h, w, rel: float) -> np.ndarray:
    ks = _odd(int(rel * min(h, w)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

def flatten_illum(gray: np.ndarray) -> np.ndarray:
    """Atenúa sombras sin halos (inpaint + blur grande) y normaliza a TARGET_BG."""
    h, w = gray.shape[:2]
    pre = cv2.GaussianBlur(gray, (3, 3), 0)

    _, rough = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(rough) > (rough.size // 2):
        rough = cv2.bitwise_not(rough)

    r_ignore = max(7, int(R_IGNOREREL * min(h, w)))
    K_ign = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ignore + 1, 2 * r_ignore + 1))
    ignore = cv2.dilate(rough, K_ign, iterations=1)

    inp = cv2.inpaint(gray, ignore, 7, cv2.INPAINT_TELEA)
    sigma = max(1.0, SIGMA_REL * min(h, w))
    bg = cv2.GaussianBlur(inp, (0, 0), sigma, sigma, borderType=cv2.BORDER_REFLECT)
    flat = np.clip(gray.astype(np.float32) - bg.astype(np.float32) + TARGET_BG, 0, 255).astype(np.uint8)
    return flat

def initial_mask(flat: np.ndarray) -> np.ndarray:
    """Binariza con umbral adaptativo y toma el componente principal."""
    h, w = flat.shape
    blur = cv2.GaussianBlur(flat, (5, 5), 0)
    block = max(35, ((min(h, w) // 18) | 1))
    bin_gray = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, 5)
    # fondo negro
    mask = bin_gray if cv2.countNonZero(bin_gray) < (bin_gray.size // 2) else cv2.bitwise_not(bin_gray)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == idx, 255, 0).astype(np.uint8)
    return mask

def quick_shape_metrics(mask: np.ndarray):
    """Devuelve (aspect_ratio, circularidad, convexidad) sobre el contorno principal."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)

    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True)) or 1.0
    circularidad = float(4.0 * np.pi * area / (per * per))  # ~1 círculo perfecto

    x, y, w, h = cv2.boundingRect(c)
    aspect = float(min(w, h) / max(w, h))  # 1 ~ cuadrado/círculo; 0 ~ muy alargado

    hull = cv2.convexHull(c)
    per_h = float(cv2.arcLength(hull, True)) or 1.0
    convexidad = float(per_h / per)  # ~1 borde liso; <1 borde “áspero” o con concavidades

    return aspect, circularidad, convexidad

def dynamic_border(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Aplica borde grueso/fino AUTOMÁTICO según aspect_ratio + circularidad + convexidad.
    No rellena agujeros (para respetar arandelas).
    """
    h, w = mask.shape
    aspect, circ, conv = quick_shape_metrics(mask)

    # perfiles por heurística
    info = {"aspect_ratio": aspect, "circularidad": circ, "convexidad": conv, "perfil": "fino"}

    # Redondo?
    es_redondo = (aspect > 0.75) and (circ > 0.60)
    if es_redondo:
        if conv > 0.97:
            # Arandela: borde bien grueso y liso
            K = rel_kernel(h, w, 0.020)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K, iterations=2)
            mask = cv2.dilate(mask, K, iterations=1)
            info["perfil"] = "arandela_grueso"
        else:
            # Tuerca: borde medio (mantener esquinas, sin “derretir” mucho)
            K = rel_kernel(h, w, 0.012)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K, iterations=1)
            info["perfil"] = "tuerca_medio"
    else:
        # Clavo/Tornillo: borde fino, limpiar ruido
        K = rel_kernel(h, w, 0.006)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K, iterations=1)
        info["perfil"] = "alargado_fino"

    return mask, info

def align_and_crop(mask: np.ndarray) -> np.ndarray:
    """Alinea por elipse y recorta bounding rect. Devuelve sólo la máscara final recortada."""
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask

    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        # sin suficientes puntos para elipse: solo recorte axis-aligned
        x, y, ww, hh = cv2.boundingRect(c)
        return mask[y:y+hh, x:x+ww]

    (cx, cy), (ax1, ax2), angle = cv2.fitEllipse(c)
    # normalizar para alinear eje mayor horizontal
    if ax2 > ax1:
        angle += 90.0
    if angle > 90: angle -= 180
    if angle <= -90: angle += 180

    center = (w // 2, h // 2)
    R = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot = cv2.warpAffine(mask, R, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    cnts_r, _ = cv2.findContours(rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_r:
        return rot
    c_r = max(cnts_r, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(c_r)
    crop = rot[y:y+hh, x:x+ww]
    return crop

# ------------- pipeline -------------
def process_all():
    for in_dir, out_dir in pairs:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = list_images(in_dir)
        if not paths:
            print(f"[AVISO] No hay imágenes en: {in_dir}")
            continue

        print(f"\n=== Procesando {in_dir} → {out_dir} ===")
        total, ok = len(paths), 0

        for path in paths:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"  [SKIP] No se pudo leer: {path.name}")
                continue

            # Resize manteniendo aspecto
            h0, w0 = gray.shape[:2]
            new_h = int(h0 * RESIZE_W / w0)
            gray = cv2.resize(gray, (RESIZE_W, new_h), interpolation=cv2.INTER_AREA)

            # Atenuar sombras y binarizar
            flat = flatten_illum(gray)
            mask0 = initial_mask(flat)

            # Heurística de borde grueso/fino (automática)
            mask_dyn, info = dynamic_border(mask0)

            # Si quedó demasiado chica/rota, fallback a cierre más fuerte
            if cv2.countNonZero(mask_dyn) < 0.005 * mask_dyn.size:
                Kfb = rel_kernel(*mask_dyn.shape, 0.012)
                mask_dyn = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, Kfb, iterations=2)

            # Alinear por elipse y recortar
            final_mask = align_and_crop(mask_dyn)

            # Guardar SOLO la binaria final (eliminando si existe)
            out_path = out_dir / (path.stem + ".png")
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception as e:
                    print(f"  [AVISO] No se pudo borrar previo: {e}")

            if cv2.imwrite(str(out_path), final_mask):
                ok += 1
                print(f"  ✓ {path.name} → {out_path.name} "
                      f"({info['perfil']}; asp={info['aspect_ratio']:.2f}, "
                      f"circ={info['circularidad']:.2f}, conv={info['convexidad']:.2f})")
            else:
                print(f"  [ERROR] No se pudo guardar: {out_path.name}")

        print(f"Listo {in_dir.name}: {ok}/{total} guardadas en {out_dir}")

if __name__ == "__main__":
    process_all()
