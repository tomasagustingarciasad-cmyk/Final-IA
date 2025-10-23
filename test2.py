import cv2
from procesado_img import escalar, preprocesar_imagen, binarizar_robusto, morfologia_conservadora, detectar_contornos, dibujar_contornos

img_path = "base_datos/Tornillo/Tornillo_2.JPG"  # cambialo según quieras
img = cv2.imread(img_path)
if img is None:
    print("No se pudo leer:", img_path); raise SystemExit

gray, gray_atenuado = preprocesar_imagen(img, fuerza=1.2, radio_borde=0.015, sigma_ilum=0.12)
binary = binarizar_robusto(gray_atenuado)
binary_final = morfologia_conservadora(binary)
contornos = detectar_contornos(binary_final)
vis = dibujar_contornos(img, contornos)

cv2.imshow("Original", escalar(img))
#cv2.imshow("Gris", escalar(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Gris (sombra atenuada)", escalar(cv2.cvtColor(gray_atenuado, cv2.COLOR_GRAY2BGR)))
#cv2.imshow("Binarizada", escalar(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))
#cv2.imshow("Binarizada (final)", escalar(cv2.cvtColor(binary_final, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Contornos", escalar(vis))

cv2.waitKey(0); 
cv2.destroyAllWindows()
