import cv2
import numpy as np
from procesado_img import (
    escalar, preprocesar_imagen, binarizar_robusto,
    morfologia_conservadora, componente_principal, rellenar_huecos,
    detectar_contornos, dibujar_contornos
)

# Cambiá la ruta/clase a gusto
img_path = "base_datos/Tornillo/Tornillo_2.JPG"
#img_path = "base_datos/Arandela/Arandela_2.JPG"
img = cv2.imread(img_path)
if img is None:
    print("No se pudo leer:", img_path); raise SystemExit

# 1) Gris + sombra atenuada (parámetros que ya funcionaban)
gray, gray_atenuado = preprocesar_imagen(
    img, fuerza=1.5, radio_borde=0.01, sigma_ilum=0.20
)#fuerza=1.2

# 2) Binarización estable
binary = binarizar_robusto(gray_atenuado)

# 3) Limpieza morfológica
binary = morfologia_conservadora(binary)

# 4) Conservar objeto y RELLENAR (clave para Hu)
binary = componente_principal(binary)

binary_filled = rellenar_huecos(binary)

# 5) Contornos para visualización
contornos = detectar_contornos(binary_filled)
vis = dibujar_contornos(img, contornos)


# --- Mostrar ---
cv2.imshow("Original", escalar(img))
cv2.imshow("Gris atenuado", escalar(cv2.cvtColor(gray_atenuado, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Máscara final (rellena)", escalar(cv2.cvtColor(binary_filled, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Contornos", escalar(vis))
cv2.waitKey(0)
cv2.destroyAllWindows()
