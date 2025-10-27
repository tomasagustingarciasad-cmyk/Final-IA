import cv2
import numpy as np
from pathlib import Path

# === Parámetros ===
RESIZE_W    = 640
TARGET_BG   = 215
R_IGNOREREL = 0.03
SIGMA_REL   = 0.15

# (input -> output)
pairs = [
    (Path("base_datos/Arandela"), Path("base_datos/ARANDELAS")),
    (Path("base_datos/Tuerca"),   Path("base_datos/TUERCAS")),
    (Path("base_datos/Tornillo"), Path("base_datos/TORNILLOS")),
    (Path("base_datos/Clavo"),    Path("base_datos/CLAVOS")),
]

def list_images(indir: Path):
    files = list(indir.glob("*.[jJpP][pPnN][gG]"))  # busca .jpg, .jpeg, .png (todas combinaciones)
    # elimina duplicados por nombre
    unique = {f.stem.lower(): f for f in files}.values()
    return sorted(unique)


for in_dir, out_dir in pairs:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(in_dir)
    if not paths:
        print(f"[AVISO] No hay imágenes en: {in_dir}")
        continue

    print(f"\n=== Procesando {in_dir} → guardando en {out_dir} ===")
    total, ok = len(paths), 0

    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [SKIP] No se pudo leer: {path.name}")
            continue

        # --- Resize ---
        h0, w0 = img.shape[:2]
        new_h = int(h0 * RESIZE_W / w0)
        img = cv2.resize(img, (RESIZE_W, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape

        # --- Aplanado de sombras SIN halo (inpaint + blur grande) ---
        pre = cv2.GaussianBlur(img, (5, 5), 0)
        _, rough = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if cv2.countNonZero(rough) > (rough.size // 2):
            rough = cv2.bitwise_not(rough)

        r_ignore = max(7, int(R_IGNOREREL * min(h, w)))
        K_ign = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_ignore+1, 2*r_ignore+1))
        ignore = cv2.dilate(rough, K_ign, iterations=1)

        inp   = cv2.inpaint(img, ignore, 7, cv2.INPAINT_TELEA)
        sigma = max(1.0, SIGMA_REL * min(h, w))
        bg    = cv2.GaussianBlur(inp, (0, 0), sigma, sigma, borderType=cv2.BORDER_REFLECT)
        flat  = np.clip(img.astype(np.float32) - bg.astype(np.float32) + TARGET_BG, 0, 255).astype(np.uint8)

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

        # Rellenar huecos
        inv = 255 - binary
        ff = inv.copy()
        mask_ff = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(ff, mask_ff, (0, 0), 0)
        holes  = ff
        filled = cv2.bitwise_or(binary, holes)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # --- Borde por gradiente + Otsu + cerrar ---
        gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, edges = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(edges)
        if cnts:
            cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=-1)
            solid = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        else:
            solid = np.zeros_like(edges)

        # --- Fusión final (binarización que vamos a guardar) ---
        fused = cv2.bitwise_or(solid, filled)

        # --- Guardar como PNG binaria ---
        out_path = out_dir / (path.stem + ".png")
        if cv2.imwrite(str(out_path), fused):
            ok += 1
            print(f"  ✓ {path.name} → {out_path.name}")
        else:
            print(f"  [ERROR] No se pudo guardar: {out_path.name}")

    print(f"Listo {in_dir.name}: {ok}/{total} guardadas en {out_dir}")
