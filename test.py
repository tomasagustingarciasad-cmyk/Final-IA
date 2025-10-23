import cv2
import numpy as np
from procesado_imagen import (
    escalar, preprocesar_imagen, binarizar_robusto,
    morfologia_conservadora, componente_principal, rellenar_huecos,
    detectar_contornos, dibujar_contornos, edge_guided_fix
)

# Cambiá la ruta/clase a gusto
img_path = "base_datos/Tornillo/Tornillo_4.JPG"
#img_path = "base_datos/Arandela/Arandela_2.JPG"
img = cv2.imread(img_path)
if img is None:
    print("No se pudo leer:", img_path); raise SystemExit

# 1) Gris + sombra atenuada (parámetros que ya funcionaban)
gray, gray_atenuado = preprocesar_imagen(
    img, fuerza=1.2, radio_borde=0.025, sigma_ilum=0.20
)

# 2) Binarización estable
binary = binarizar_robusto(gray_atenuado)

# 3) Limpieza morfológica
binary = morfologia_conservadora(binary)

# 4) Conservar objeto y RELLENAR (clave para Hu)
binary = componente_principal(binary)

binary = edge_guided_fix(gray, binary, r=2, min_edge_ratio=0.04,
                           canny_low=20, canny_high=60)

binary_filled = rellenar_huecos(binary)

# 5) Contornos para visualización
contornos = detectar_contornos(binary_filled)
vis = dibujar_contornos(img, contornos)

# Debug corto (opcional)
print("mean(gray_atenuado):", np.mean(gray_atenuado))
print("pix binarios:", cv2.countNonZero(binary), "/", binary.size)

# --- Mostrar ---
cv2.imshow("Original", escalar(img))
cv2.imshow("Gris atenuado", escalar(cv2.cvtColor(gray_atenuado, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Máscara final (rellena)", escalar(cv2.cvtColor(binary_filled, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Contornos", escalar(vis))
cv2.waitKey(0)
cv2.destroyAllWindows()
