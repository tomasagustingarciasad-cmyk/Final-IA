import cv2
import numpy as np
from procesado_img import (
    escalar, preprocesar_imagen, binarizar_robusto,
    morfologia_conservadora, componente_principal, rellenar_huecos,
    detectar_contornos, dibujar_contornos, deslumbrado_anillo, rellenar_mordidas, cerrar_muescas_local, medidas_forma, es_pieza_compacta
)

# Cambiá la ruta/clase a gusto
#img_path = "base_datos/Tornillo/Tornillo_6.JPG"
img_path = "base_datos/Arandela/Arandela_4.JPG"
#img_path = "base_datos/Tuerca/Tuerca_10.JPG"
#img_path = "base_datos/Clavo/Clavo_3.JPG"

img = cv2.imread(img_path)
if img is None:
    print("No se pudo leer:", img_path); raise SystemExit

# 1) Gris + sombra atenuada (parámetros que ya funcionaban)
gray, gray_atenuado = preprocesar_imagen(
    img, fuerza=1.5, radio_borde=0.01, sigma_ilum=0.20
)#fuerza=1.2



# 2) Máscara rápida (para localizar pieza) + medir forma
bin0 = binarizar_robusto(gray_atenuado)
bin0 = morfologia_conservadora(bin0)
mask0 = componente_principal(bin0)

m = medidas_forma(mask0)
compacta = es_pieza_compacta(m)

# 3) Preprocesado extra sólo si es compacta (anti-brillo en anillo interno)
if compacta:
    gray_in = deslumbrado_anillo(
        gray_atenuado, mask0,
        ring_px=9,           # 8–10
        k_rel=0.06,          # 0.06–0.08
        frac=0.85,           # 0.8–0.9
        trigger_sigma=1.4,   # 1.3–1.6
        min_ratio=0.006, max_ratio=0.40,
        edge_clear_px=1,     # no tocar el píxel más externo
        delta_cap=24
    )
else:
    gray_in = gray_atenuado  # tornillos/clavos: no aplicar anti-brillo fuerte

# 4) Binarización estable sobre la imagen seleccionada
binary = binarizar_robusto(gray_in)

# 5) Limpieza morfológica
binary = morfologia_conservadora(binary)

# 6) Conservar objeto principal
binary = componente_principal(binary)

# 7) Post-procesado SÓLO para compactas (cerrar muescas/concavidades finitas)
if compacta:
    binary = rellenar_mordidas(binary, max_depth_px=8)          # 6–10 px
    binary = cerrar_muescas_local(binary, r_close=3, max_depth_px=9)

# 8) Rellenar (sin agujeros internos por pedido tuyo)
binary_filled = rellenar_huecos(binary)

# 9) Contornos para visualización
contornos = detectar_contornos(binary_filled)
vis = dibujar_contornos(img, contornos)


# --- Mostrar ---
cv2.imshow("Original", escalar(img))
cv2.imshow("Gris atenuado", escalar(cv2.cvtColor(gray_atenuado, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Máscara final (rellena)", escalar(cv2.cvtColor(binary_filled, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Contornos", escalar(vis))
cv2.waitKey(0)
cv2.destroyAllWindows()
