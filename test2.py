import cv2
import numpy as np
from procesado_img import (
    escalar, preprocesar_imagen, binarizar_robusto,
    morfologia_conservadora, componente_principal, rellenar_huecos,
    detectar_contornos, dibujar_contornos, deslumbrado_anillo, rellenar_mordidas, cerrar_muescas_local
)

# Cambiá la ruta/clase a gusto
img_path = "base_datos/Tornillo/Tornillo_4.JPG"
#img_path = "base_datos/Arandela/Arandela_1.JPG"
#img_path = "base_datos/Tuerca/Tuerca_10.JPG"
img = cv2.imread(img_path)
if img is None:
    print("No se pudo leer:", img_path); raise SystemExit

# 1) Gris + sombra atenuada (parámetros que ya funcionaban)
gray, gray_atenuado = preprocesar_imagen(
    img, fuerza=1.5, radio_borde=0.01, sigma_ilum=0.20
)#fuerza=1.2



# 2) Máscara rápida (para localizar pieza)
bin0 = binarizar_robusto(gray_atenuado)
bin0 = morfologia_conservadora(bin0)
mask0 = componente_principal(bin0)


gray_flat = deslumbrado_anillo(
    gray_atenuado, mask0,
    ring_px=10,          # banda interna un poco más ancha
    k_rel=0.07,          # 7% del lado menor
    frac=0.90,           # un toque más fuerte
    trigger_sigma=1.35,  # un pelín más sensible
    min_ratio=0.006, max_ratio=0.40,
    edge_clear_px=0,     # <- ahora sí permitimos tocar el px más externo
    delta_cap=26         # tope por píxel, sigue habiendo guardrail
)


# 2) Binarización estable
binary = binarizar_robusto(gray_flat)

# 3) Limpieza morfológica
binary = morfologia_conservadora(binary)

# 4) Conservar objeto y RELLENAR (clave para Hu)
binary = componente_principal(binary)
binary = rellenar_mordidas(binary, max_depth_px=20)  # 6–10 px; probá 8
binary = cerrar_muescas_local(binary, r_close=5, max_depth_px=15)
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
