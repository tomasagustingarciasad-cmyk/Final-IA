# procesado_core.py
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

# =========================
# Parámetros por defecto
# =========================
RESIZE_W    = 640
TARGET_BG   = 215
R_IGNOREREL = 0.03
SIGMA_REL   = 0.15

# Pares por defecto (dataset completo)
PAIRS = [
    (Path("base_datos/Arandela"), Path("base_datos/ARANDELAS")),
    (Path("base_datos/Tuerca"),   Path("base_datos/TUERCAS")),
    (Path("base_datos/Tornillo"), Path("base_datos/TORNILLOS")),
    (Path("base_datos/Clavo"),    Path("base_datos/CLAVOS")),
]


# =========================
# Utilidades
# =========================
def list_images(indir: Path) -> List[Path]:
    """Lista imágenes únicas por nombre base (case-insensitive)."""
    pats = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    files = []
    for p in pats:
        files.extend(indir.glob(p))
    unique = {f.stem.lower(): f for f in files}.values()
    return sorted(unique)


# =========================
# 1) Pipeline hasta 'fused'
# =========================
def procesar_a_mascara(
    img_gray: np.ndarray,
    resize_w: int = RESIZE_W,
    target_bg: int = TARGET_BG,
    r_ign_rel: float = R_IGNOREREL,
    sigma_rel: float = SIGMA_REL,
) -> np.ndarray:
    """
    Aplica TODO el pipeline desde el resize hasta la fusión final (fused).
    Devuelve una máscara binaria uint8 (0/255) sin alinear/recortar.
    """
    if img_gray is None:
        raise ValueError("img_gray es None")

    # --- Resize ---
    h0, w0 = img_gray.shape[:2]
    new_h = int(h0 * resize_w / max(1, w0))
    img = cv2.resize(img_gray, (resize_w, new_h), interpolation=cv2.INTER_AREA)
    h, w = img.shape

    # --- Aplanado de sombras (inpaint + blur grande) ---
    pre = cv2.GaussianBlur(img, (3, 3), 0)
    _, rough = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(rough) > (rough.size // 2):
        rough = cv2.bitwise_not(rough)

    r_ignore = max(7, int(r_ign_rel * min(h, w)))
    K_ign = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_ignore+1, 2*r_ignore+1))
    ignore = cv2.dilate(rough, K_ign, iterations=1)

    inp   = cv2.inpaint(img, ignore, 7, cv2.INPAINT_TELEA)
    sigma = max(1.0, sigma_rel * min(h, w))
    bg    = cv2.GaussianBlur(inp, (0, 0), sigma, sigma, borderType=cv2.BORDER_REFLECT)
    flat  = np.clip(img.astype(np.float32) - bg.astype(np.float32) + target_bg, 0, 255).astype(np.uint8)

    blur = cv2.GaussianBlur(flat, (5, 5), 0)

    # --- Región por umbral adaptativo (mayor componente) ---
    block = max(35, ((min(h, w) // 18) | 1))
    binary_gray = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, block, 5)
    binary = binary_gray if cv2.countNonZero(binary_gray) < (binary_gray.size // 2) \
             else cv2.bitwise_not(binary_gray)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = np.where(labels == idx, 255, 0).astype(np.uint8)

    # --- Relleno de huecos (flood fill desde el borde) ---
    inv = 255 - binary
    ff  = inv.copy()
    mask_ff = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(ff, mask_ff, (0, 0), 0)
    holes  = ff
    filled = cv2.bitwise_or(binary, holes)

    # --- Borde por gradiente + Otsu + dilatar ---
    gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, edges = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # --- Refinamiento opcional con Canny ---
    try:
        v = np.median(blur)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges_canny = cv2.Canny(blur, lower, upper)
        edges_canny = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel, iterations=1)
        if cv2.countNonZero(edges_canny) > (0.01 * edges_canny.size):
            edges = cv2.bitwise_or(edges, edges_canny)
    except Exception:
        pass

    # --- Contorno principal -> sólido ---
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=-1)
        solid = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        solid = np.zeros_like(edges)

    # --- Fusión final ---
    fused = cv2.bitwise_or(solid, filled)
    return fused.astype(np.uint8)


# =========================
# 2) Alinear + recortar
# =========================
def alinear_recortar(mask_bin: np.ndarray) -> np.ndarray:
    """
    Devuelve la máscara alineada (minAreaRect) y recortada a su bounding box.
    Si no se puede, devuelve la original.
    """
    h, w = mask_bin.shape[:2]
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask_bin
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return mask_bin

    (cx, cy), (ax1, ax2), angle = cv2.minAreaRect(c)
    if ax2 > ax1:
        angle += 90.0
    if angle > 90:
        angle -= 180
    if angle <= -90:
        angle += 180

    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        mask_bin, rot_mat, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    cnts_r, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_r:
        return rotated
    c_r = max(cnts_r, key=cv2.contourArea)
    x, y, w_r, h_r = cv2.boundingRect(c_r)
    return rotated[y:y+h_r, x:x+w_r]


def alinear_recortar_y_guardar(
    fused: np.ndarray,
    out_path: Path,
    src_name: Optional[str] = None,
    verbose: bool = True
) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """
    Toma la binaria 'fused', alinea por minAreaRect, recorta bbox y guarda en 'out_path'.
    Devuelve (saved, shape_crop) donde shape_crop es (alto, ancho) si pudo guardar.
    """
    try:
        rotated_cropped = alinear_recortar(fused)
        if rotated_cropped is None or rotated_cropped.size == 0:
            return (False, None)

        # Overwrite seguro
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass

        ok = cv2.imwrite(str(out_path), rotated_cropped)
        if ok and verbose:
            base = src_name if src_name else out_path.name
            print(f"  ✓ {base} → {out_path.name} (alineada)")
        return (ok, rotated_cropped.shape if ok else None)

    except Exception as e:
        if verbose:
            print(f"  [AVISO] Alineación/recorte omitida ({e})")
        return (False, None)


# =========================
# 3) Helpers de alto nivel
# =========================
def procesar_archivo(path_img: Path, out_dir: Path, verbose: bool = True) -> bool:
    """
    Procesa UNA imagen:
      - lee en gris
      - pipeline hasta fused
      - intenta alinear/recortar/guardar
      - fallback: guarda fused tal cual
    Devuelve True si se guardó algún archivo, False si no.
    """
    img = cv2.imread(str(path_img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        if verbose:
            print(f"  [SKIP] No se pudo leer: {path_img.name}")
        return False

    fused = procesar_a_mascara(img)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path_img.stem + ".png")

    saved, _shape = alinear_recortar_y_guardar(
        fused=fused,
        out_path=out_path,
        src_name=path_img.name,
        verbose=verbose
    )
    if saved:
        return True

    try:
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        ok = cv2.imwrite(str(out_path), fused)
        if ok and verbose:
            print(f"  ✓ {path_img.name} → {out_path.name} (sin alinear)")
        return ok
    except Exception as e:
        if verbose:
            print(f"  [ERROR] No se pudo guardar: {out_path.name} ({e})")
        return False


def procesar_par(in_dir: Path, out_dir: Path) -> None:
    """Procesa todas las imágenes de un directorio a otro."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(in_dir)
    if not paths:
        print(f"[AVISO] No hay imágenes en: {in_dir}")
        return

    print(f"\n=== Procesando {in_dir} → guardando en {out_dir} ===")
    total, ok = len(paths), 0
    for path in paths:
        ok += 1 if procesar_archivo(path, out_dir, verbose=True) else 0
    print(f"Listo {in_dir.name}: {ok}/{total} guardadas en {out_dir}")

def procesar_pares(pairs: List[Tuple[Path, Path]]) -> None:
    """Procesa una lista de pares (input_dir, output_dir)."""
    for in_dir, out_dir in pairs:
        procesar_par(in_dir, out_dir)

def procesar_imagen_completa(img_path: Path) -> np.ndarray:
    """Procesa UNA imagen y devuelve la máscara final alineada/recortada."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {img_path}")
    fused = procesar_a_mascara(img)
    final = alinear_recortar((fused > 0).astype(np.uint8) * 255)
    return (final > 0).astype(np.uint8) * 255

if __name__ == "__main__":
    procesar_pares(PAIRS)
