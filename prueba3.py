import cv2
import numpy as np

def escalar(imagen, max_ancho=1280, max_alto=720):
    """Escala proporcionalmente sin agrandar"""
    alto, ancho = imagen.shape[:2]
    escala = min(max_ancho / ancho, max_alto / alto, 1.0)
    if escala < 1.0:
        nuevo_tamaño = (int(ancho * escala), int(alto * escala))
        imagen = cv2.resize(imagen, nuevo_tamaño, interpolation=cv2.INTER_AREA)
    return imagen

def preprocesar_imagen(img):
    """
    Preprocesamiento robusto para eliminar reflejos y mejorar contraste
    """
    # 1. Convertir a LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    
    # 2. Detección y corrección de reflejos más agresiva
    mask_reflex = cv2.inRange(L, 190, 255)
    kernel_reflex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_reflex = cv2.dilate(mask_reflex, kernel_reflex, iterations=1)
    L_sin_reflejos = cv2.inpaint(L, mask_reflex, 7, cv2.INPAINT_TELEA)
    
    # 3. Suavizado bilateral
    L_suavizado = cv2.bilateralFilter(L_sin_reflejos, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 4. Compresión logarítmica
    L_float = L_suavizado.astype(np.float32) / 255.0
    L_log = np.log1p(L_float * 10) / np.log(11)
    L_comprimido = cv2.normalize(L_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 5. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_final = clahe.apply(L_comprimido)
    
    # Recombinar y convertir a gris
    lab_final = cv2.merge((L_final, A, B))
    img_corregida = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(img_corregida, cv2.COLOR_BGR2GRAY)
    
    return gray

def binarizar_robusto(gray):
    """
    Binarización con múltiples estrategias combinadas
    """
    # Estrategia 1: Threshold adaptativo Gaussiano
    binary_gauss = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=8
    )
    
    # Estrategia 2: Threshold adaptativo con Media
    binary_mean = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=8
    )
    
    # Estrategia 3: Otsu con desenfoque previo
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combinación: Intersección de las tres (más conservadora)
    binary_combinada = cv2.bitwise_and(binary_gauss, binary_mean)
    binary_combinada = cv2.bitwise_and(binary_combinada, binary_otsu)
    
    # Limpieza de ruido
    binary_combinada = cv2.medianBlur(binary_combinada, 5)
    
    # Verificar polaridad (pieza blanca, fondo negro)
    if cv2.countNonZero(binary_combinada) > (binary_combinada.size // 2):
        binary_combinada = cv2.bitwise_not(binary_combinada)
    
    return binary_combinada

def morfologia_conservadora(binary):
    """
    Operaciones morfológicas conservadoras
    """
    h, w = binary.shape
    
    # Kernel pequeño y proporcional
    k = max(3, min(7, (min(h, w) // 400) | 1))
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    
    # Apertura para eliminar ruido
    result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, K, iterations=1)
    
    # Cierre suave para conectar partes cercanas
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, K, iterations=1)
    
    return result


# ========== PROGRAMA PRINCIPAL ==========
if __name__ == "__main__":
    # Seleccionar imagen
    # nombre_imagen = "base_datos_0/IMG_2765.JPG"  # Tuerca
    # nombre_imagen = "base_datos_0/IMG_2771.JPG"    # Tornillo
    nombre_imagen = "base_datos_0/IMG_2859.JPG"  # Arandela
    
    img = cv2.imread(nombre_imagen)
    if img is None:
        print(f"Error: No se pudo cargar {nombre_imagen}")
        exit()
    
    # Pipeline de binarización
    print("Procesando imagen...")
    
    # 1. Preprocesamiento
    gray = preprocesar_imagen(img)
    
    # 2. Binarización robusta
    binary = binarizar_robusto(gray)
    
    # 3. Limpieza morfológica
    binary_final = morfologia_conservadora(binary)
    
    print("Binarización completada.")
    
    # Visualización
    img_vista = escalar(img)
    gray_vista = escalar(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    binary_vista = escalar(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    final_vista = escalar(cv2.cvtColor(binary_final, cv2.COLOR_GRAY2BGR))
    
    cv2.imshow("1. Original", img_vista)
    cv2.imshow("2. Preprocesado", gray_vista)
    cv2.imshow("3. Binarización", binary_vista)
    cv2.imshow("4. Binarización Final", final_vista)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()