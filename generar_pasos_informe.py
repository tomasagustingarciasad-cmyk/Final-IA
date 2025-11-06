import cv2
import numpy as np
from pathlib import Path
import sys

# Importamos las funciones y constantes de tu script original
try:
    from procesado_img import (
        alinear_recortar,
        RESIZE_W,
        TARGET_BG,
        R_IGNOREREL,
        SIGMA_REL
    )
except ImportError:
    print("Error: Asegúrate de que 'procesado_img.py' esté en la misma carpeta.")
    sys.exit(1)

def generar_y_mostrar_pasos(img_path: Path):
    """
    Carga una imagen y muestra las 6 etapas clave del procesamiento
    para que puedas tomar capturas de pantalla para el informe.
    
    Presiona cualquier tecla en la ventana para avanzar al siguiente paso.
    """
    
    # --- 1. IMAGEN ORIGINAL ---
    img_original = cv2.imread(str(img_path))
    if img_original is None:
        print(f"Error: No se pudo leer la imagen {img_path}")
        return
    
    # Redimensionamos la original para que quepa en pantalla
    h, w = img_original.shape[:2]
    new_h_orig = int(h * RESIZE_W / max(1, w))
    img_original_resized = cv2.resize(img_original, (RESIZE_W, new_h_orig))
    
    cv2.imshow("1. Imagen Original", img_original_resized)
    print("Mostrando 1. Original. Presiona una tecla para continuar...")
    cv2.waitKey(0)

    # --- INICIO DEL PIPELINE (copiado de 'procesar_a_mascara' en procesado_img.py) ---
    
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    # --- Resize ---
    h0, w0 = img_gray.shape[:2]
    new_h = int(h0 * RESIZE_W / max(1, w0))
    img = cv2.resize(img_gray, (RESIZE_W, new_h), interpolation=cv2.INTER_AREA)
    h, w = img.shape

    # --- Aplanado de sombras ---
    pre = cv2.GaussianBlur(img, (3, 3), 0)
    _, rough = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(rough) > (rough.size // 2):
        rough = cv2.bitwise_not(rough)
    
    r_ignore = max(7, int(R_IGNOREREL * min(h, w)))
    K_ign = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_ignore+1, 2*r_ignore+1))
    ignore = cv2.dilate(rough, K_ign, iterations=1)
    
    inp = cv2.inpaint(img, ignore, 7, cv2.INPAINT_TELEA)
    sigma = max(1.0, SIGMA_REL * min(h, w))
    bg = cv2.GaussianBlur(inp, (0, 0), sigma, sigma, borderType=cv2.BORDER_REFLECT)
    flat = np.clip(img.astype(np.float32) - bg.astype(np.float32) + TARGET_BG, 0, 255).astype(np.uint8)

    # --- 2. IMAGEN APLANADA ---
    cv2.imshow("2. Aplanada (Sin Sombras)", flat)
    print("Mostrando 2. Aplanada. Presiona una tecla para continuar...")
    cv2.waitKey(0)

    blur = cv2.GaussianBlur(flat, (5, 5), 0)

    # --- Región por umbral adaptativo ---
    block = max(35, ((min(h, w) // 18) | 1))
    binary_gray = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, block, 5)
    binary = binary_gray if cv2.countNonZero(binary_gray) < (binary_gray.size // 2) \
             else cv2.bitwise_not(binary_gray)
    
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = np.where(labels == idx, 255, 0).astype(np.uint8)

    # --- 3. MÁSCARA DE UMBRAL (Threshold) ---
    cv2.imshow("3. Mascara de Umbral", binary)
    print("Mostrando 3. Umbral Adaptativo. Presiona una tecla para continuar...")
    cv2.waitKey(0)

    # --- Relleno de huecos (solo para la fusión) ---
    inv = 255 - binary
    ff = inv.copy()
    mask_ff = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(ff, mask_ff, (0, 0), 0)
    holes = ff
    filled = cv2.bitwise_or(binary, holes)

    # --- Borde por gradiente + Otsu + Canny ---
    gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    _, edges = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
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

    # --- 4. MÁSCARA DE BORDES ---
    cv2.imshow("4. Mascara de Bordes", edges)
    print("Mostrando 4. Bordes. Presiona una tecla para continuar...")
    cv2.waitKey(0)

    # --- Contorno principal -> sólido (solo para la fusión) ---
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=-1)
        solid = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        solid = np.zeros_like(edges)

    # --- Fusión final ---
    fused = cv2.bitwise_or(solid, filled)

    # --- 5. MÁSCARA FUSIONADA ---
    cv2.imshow("5. Mascara Fusionada (Relleno + Bordes)", fused)
    print("Mostrando 5. Fusionada. Presiona una tecla para continuar...")
    cv2.waitKey(0)

    # --- Alinear y recortar (usando la función importada) ---
    final_mask = alinear_recortar(fused)
    
    # --- 6. MÁSCARA FINAL ---
    cv2.imshow("6. Mascara Final (Alineada y Recortada)", final_mask)
    print("Mostrando 6. Final. Presiona una tecla para cerrar todo.")
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("Proceso completado.")

# --- Bloque para ejecutar el script ---
if __name__ == "__main__":
    # Pide la ruta de la imagen en la consola
    ruta_imagen = input("Ingresa la ruta completa de la imagen que quieres procesar: ")
    p = Path(ruta_imagen)
    
    if not p.exists():
        print(f"Error: No se encuentra el archivo en: {p}")
    else:
        generar_y_mostrar_pasos(p)