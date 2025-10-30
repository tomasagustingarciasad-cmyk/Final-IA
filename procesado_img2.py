import cv2
import numpy as np
from pathlib import Path

# === Parámetros ===
# Forzamos todas las piezas al mismo ancho para que las métricas geométricas sean comparables.
RESIZE_W    = 640
# Nivel de gris objetivo que usamos para "planchar" iluminación tras restar el fondo.
TARGET_BG   = 215
# Radio relativo usado para ignorar bordes gruesos al estimar el fondo.
R_IGNOREREL = 0.03
# Sigma relativo para el suavizado que modela el fondo.
SIGMA_REL   = 0.15

# (input -> output)
# Procesamos cada carpeta con originales (#1) y exportamos la máscara binaria en (#2).
pairs = [
    (Path("base_datos/Arandela"), Path("base_datos/ARANDELAS")),
    (Path("base_datos/Tuerca"),   Path("base_datos/TUERCAS")),
    (Path("base_datos/Tornillo"), Path("base_datos/TORNILLOS")),
    (Path("base_datos/Clavo"),    Path("base_datos/CLAVOS")),
]

def list_images(indir: Path):
# Buscamos archivos .jpg/.jpeg/.png sin importar mayúsculas y descartamos duplicados de nombre.
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
        # Escalamos manteniendo aspecto para que todas tengan el ancho RESIZE_W.
        h0, w0 = img.shape[:2]
        new_h = int(h0 * RESIZE_W / w0)
        img = cv2.resize(img, (RESIZE_W, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape

        # --- Aplanado de sombras SIN halo (inpaint + blur grande) ---
        # 1) Detectamos zonas oscuras/claras brutas con Otsu para saber dónde hay pieza.
        #pre = cv2.GaussianBlur(img, (5, 5), 0)
        pre = cv2.GaussianBlur(img, (3, 3), 0)

        _, rough = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if cv2.countNonZero(rough) > (rough.size // 2):
            rough = cv2.bitwise_not(rough)

        # 2) Expandimos la máscara para cubrir bordes gruesos y pedimos a inpaint que la rellene.
        r_ignore = max(7, int(R_IGNOREREL * min(h, w)))
        K_ign = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_ignore+1, 2*r_ignore+1))
        ignore = cv2.dilate(rough, K_ign, iterations=1)

        inp   = cv2.inpaint(img, ignore, 7, cv2.INPAINT_TELEA)
        sigma = max(1.0, SIGMA_REL * min(h, w))
        bg    = cv2.GaussianBlur(inp, (0, 0), sigma, sigma, borderType=cv2.BORDER_REFLECT)
        # 3) Restamos el fondo estimado y llevamos todo a TARGET_BG para homogenizar.
        flat  = np.clip(img.astype(np.float32) - bg.astype(np.float32) + TARGET_BG, 0, 255).astype(np.uint8)

        blur = cv2.GaussianBlur(flat, (5, 5), 0)
        
        # --- Región por umbral adaptativo (mayor componente) ---
        # Umbral local para separar pieza del fondo, escogiendo el componente más grande.
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
        # Eliminamos huecos internos con flood fill desde el borde exterior.
        inv = 255 - binary
        ff = inv.copy()
        mask_ff = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(ff, mask_ff, (0, 0), 0)
        holes  = ff
        filled = cv2.bitwise_or(binary, holes)
        
        # --- Borde por gradiente + Otsu + cerrar ---
        # Calculamos gradiente con Scharr y umbralizamos para quedarnos con bordes fuertes.
        gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, edges = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        
        
        
        edges = cv2.dilate(edges, kernel, iterations=1)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


        # --- Refinamiento opcional con Canny ---
        try:
            # Detecta bordes finos con umbrales adaptativos
            v = np.median(blur)
            lower = int(max(0, 0.66 * v))
            upper = int(min(255, 1.33 * v))
            edges_canny = cv2.Canny(blur, lower, upper)

            # Cierra pequeños cortes y unifica
            edges_canny = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Si Canny encontró un borde con área suficiente, lo fusionamos
            if cv2.countNonZero(edges_canny) > (0.01 * edges_canny.size):
                # “OR” lógico: mantiene el borde principal y agrega los más finos de Canny
                edges = cv2.bitwise_or(edges, edges_canny)
        except Exception as e:
            print(f"  [AVISO] Refinador Canny omitido ({e})")


        # Tomamos el contorno principal para obtener un borde compacto.
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(edges)
        if cnts:
            cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=-1)
            solid = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        else:
            solid = np.zeros_like(edges)

        # --- Fusión final (binarización que vamos a guardar) ---
        # Combinamos la región llenada y el sólido derivado del borde para robustez.
        fused = cv2.bitwise_or(solid, filled)


        # --- Guardado SOLO de binarización alineada/recortada ---
        out_path = out_dir / (path.stem + ".png")
        saved = False
        # --- Fusión final (binarización que vamos a guardar) ---
        try:
            cnts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if len(c) >= 5:
                    #(cx, cy), (ax1, ax2), angle = cv2.fitEllipse(c)
                    (cx, cy), (ax1, ax2), angle = cv2.minAreaRect(c)

                    # Alinear por eje mayor y normalizar ángulo a [-90, 90)
                    if ax2 > ax1:
                        angle += 90.0
                    if angle > 90:
                        angle -= 180
                    if angle <= -90:
                        angle += 180

                    # Rotar SOLO la máscara binaria (sin suavizar)
                    center = (w // 2, h // 2)
                    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_mask = cv2.warpAffine(
                        fused, rot_mat, (w, h),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0
                    )

                    # Recortar bounding box de la pieza ya rotada
                    cnts_r, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts_r:
                        c_r = max(cnts_r, key=cv2.contourArea)
                        x_r, y_r, w_r, h_r = cv2.boundingRect(c_r)
                        crop_mask = rotated_mask[y_r:y_r+h_r, x_r:x_r+w_r]

                        # Si existe un archivo previo con el mismo nombre, borrarlo
                        if out_path.exists():
                            try:
                                out_path.unlink()
                            except Exception as e:
                                print(f"  [AVISO] No se pudo borrar previo: {e}")

                        if cv2.imwrite(str(out_path), crop_mask):
                            ok += 1
                            saved = True
                            print(f"  ✓ {path.name} → {out_path.name} (alineada)")
        except Exception as e:
            print(f"  [AVISO] Alineación/recorte omitida ({e})")

        # Fallback: si no se pudo alinear/recortar, guardar la binaria tal cual
        if not saved:
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception as e:
                    print(f"  [AVISO] No se pudo borrar previo: {e}")
            if cv2.imwrite(str(out_path), fused):
                ok += 1
                print(f"  ✓ {path.name} → {out_path.name} (sin alinear)")
            else:
                print(f"  [ERROR] No se pudo guardar: {out_path.name}")

    print(f"Listo {in_dir.name}: {ok}/{total} guardadas en {out_dir}")
