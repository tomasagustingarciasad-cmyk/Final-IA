import cv2
import numpy as np
from procesado_img import (
    escalar, preprocesar_imagen, binarizar_robusto,
    morfologia_conservadora, componente_principal, rellenar_huecos,
    detectar_contornos, dibujar_contornos, deslumbrado_anillo
)

# Cambiá la ruta/clase a gusto
# img_path = "base_datos/Tornillo/Tornillo_6.JPG"
# img_path = "base_datos/Arandela/Arandela_1.JPG"
# img_path = "base_datos/Tuerca/Tuerca_9.JPG"
img_path = "base_datos/Clavo/Clavo_14.JPG"

img = cv2.imread(img_path)
if img is None:
    print("No se pudo leer:", img_path)
    raise SystemExit

# 1) Gris + sombra atenuada (parámetros que ya funcionaban)
gray, gray_atenuado = preprocesar_imagen(
    img, fuerza=1.5, radio_borde=0.01, sigma_ilum=0.20
)  # fuerza=1.2

# 2) Máscara rápida (para localizar pieza)
bin0 = binarizar_robusto(gray_atenuado)
bin0 = morfologia_conservadora(bin0)
mask0 = componente_principal(bin0)

gray_flat = deslumbrado_anillo(
    gray_atenuado, mask0,
    ring_px=9,        # antes 3
    k_rel=0.08,       # antes 0.03
    frac=0.88,        # antes 0.55
    trigger_sigma=1.3,  # antes 2.0
    min_ratio=0.008,  # un poco más sensible
    max_ratio=0.35    # margen por si el anillo es grande
)

# 3) Binarización estable
binary = binarizar_robusto(gray_flat)

# 4) Limpieza morfológica
binary = morfologia_conservadora(binary)

# 5) Conservar objeto y RELLENAR (clave para Hu)
binary = componente_principal(binary)
binary_filled = rellenar_huecos(binary)

# 6) Contornos para visualización
contornos = detectar_contornos(binary_filled)
vis = dibujar_contornos(img, contornos)

# --- Mostrar ---
cv2.imshow("Original", escalar(img))
cv2.imshow("Gris atenuado", escalar(cv2.cvtColor(gray_atenuado, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Máscara final (rellena)", escalar(cv2.cvtColor(binary_filled, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Contornos", escalar(vis))
cv2.waitKey(0)
cv2.destroyAllWindows()
