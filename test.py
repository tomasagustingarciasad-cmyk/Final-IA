# test_tornillo.py
import cv2
from procesado_imagen import escalar, preprocesar_imagen, binarizar_robusto, morfologia_conservadora,detectar_contornos, dibujar_contornos

# Ruta de imagen
nombre_imagen = "base_datos/Tornillo/Tornillo_1.JPG"
# nombre_imagen = "base_datos/Tuerca/Tuerca_3.JPG"
# nombre_imagen = "base_datos/Tornillo/Tornillo_1.JPG"
# nombre_imagen = "base_datos/Clavo/Clavo_2.JPG"
# nombre_imagen = "base_datos/Arandela/Arandela_5.JPG"

# Cargar imagen
img = cv2.imread(nombre_imagen)
if img is None:
    print(f"Error: No se pudo cargar {nombre_imagen}")
    exit()

print("Procesando imagen...")

# Paso 1: preprocesar
gray = preprocesar_imagen(img)

# Paso 2: binarizar
binary = binarizar_robusto(gray)

# Paso 3: morfología
binary_final = morfologia_conservadora(binary)

contornos = detectar_contornos(binary)         # contornos
vis = dibujar_contornos(img, contornos)        # visualización

print("Binarización completada.")

# Mostrar resultados
cv2.imshow("Original", escalar(img))
cv2.imshow("Preprocesada", escalar(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Binarizada", escalar(cv2.cvtColor(binary_final, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Contornos", escalar(vis))


cv2.waitKey(0)
cv2.destroyAllWindows()
