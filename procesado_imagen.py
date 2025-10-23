# procesado_imagen.py
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
    """Preprocesamiento robusto para eliminar reflejos y mejorar contraste"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    
     # —— Quitar sombras sobre L (fondo por mediana grande + división) ——
    h, w = L.shape
    #k_fondo = max(31, int(min(h, w) / 20))
    k_fondo = max(31, int(min(h, w) / 12)) 
    k_fondo = min(k_fondo, 101)

    if k_fondo % 2 == 0:
        k_fondo += 1

    fondo = cv2.medianBlur(L, k_fondo)

    L_float = L.astype(np.float32) + 1e-3
    fondo_float = fondo.astype(np.float32) + 1e-3
    gain = 128.0 / fondo_float
    gain = np.clip(gain, 0.85, 1.30)   # subir límite sup. si la sombra sigue fuerte (1.35–1.45)
    L_corr = (L_float * gain)
    L_corr = np.clip(L_corr, 0, 255).astype(np.uint8)

    # ---- PLAN B: corrección por sustracción (útil cuando la división deja halo) ----
    fondo_gauss = cv2.GaussianBlur(L, (0, 0), sigmaX=k_fondo/2, sigmaY=k_fondo/2)
    L_sub = cv2.subtract(L, fondo_gauss)
    L_sub = cv2.normalize(L_sub, None, 0, 255, cv2.NORM_MINMAX)

    # ---- Elegir automáticamente la mejor corrección (división vs sustracción) ----
    _, mask_div = cv2.threshold(L_corr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask_sub = cv2.threshold(L_sub, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ratio_div = cv2.countNonZero(mask_div) / mask_div.size
    ratio_sub = cv2.countNonZero(mask_sub) / mask_sub.size

    # rango razonable de cobertura de pieza: 0.1%–60% del frame
    ok_div = 0.001 < ratio_div < 0.60
    ok_sub = 0.001 < ratio_sub < 0.60

    base_L = L_corr
    if ok_sub and (ratio_sub > ratio_div or not ok_div):
        base_L = L_sub   # usa la alternativa que separa mejor pieza/fondo

    # Detección y corrección de reflejos
    mask_reflex = cv2.inRange(L_corr, 220, 255)
    kernel_reflex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_reflex = cv2.dilate(mask_reflex, kernel_reflex, iterations=1)
    L_sin_reflejos = cv2.inpaint(L_corr, mask_reflex, 7, cv2.INPAINT_TELEA)

    
    # Suavizado bilateral
    L_suavizado = cv2.bilateralFilter(L_sin_reflejos, d=9, sigmaColor=75, sigmaSpace=75)

    # Compresión logarítmica

    L_norm = L_suavizado.astype(np.float32) / 255.0
    L_log = np.log1p(L_norm * 10) / np.log(11)
    L_comprimido = cv2.normalize(L_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_final = clahe.apply(L_comprimido)
    
    lab_final = cv2.merge((L_final, A, B))
    img_corregida = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(img_corregida, cv2.COLOR_BGR2GRAY)
    
    return gray


def binarizar_robusto(gray):
    """Binarización combinando varios métodos"""
    binary_gauss = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=8
    )

    binary_mean = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=8
    )

    blur = cv2.GaussianBlur(gray, (5, 5), 0) 
    _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #_, binary_otsu = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    binary_combinada = cv2.bitwise_and(binary_gauss, binary_mean)
    binary_combinada = cv2.bitwise_and(binary_combinada, binary_otsu)
    binary_combinada = cv2.medianBlur(binary_combinada, 5)

    if cv2.countNonZero(binary_combinada) > (binary_combinada.size // 2):
        binary_combinada = cv2.bitwise_not(binary_combinada)
    
    return binary_combinada


def morfologia_conservadora(binary):
    """Operaciones morfológicas para limpieza"""
    h, w = binary.shape
    k = max(3, min(7, (min(h, w) // 400) | 1))
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, K, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, K, iterations=1)
    return result


# ====== NUEVO: helpers de contornos, usando tus nombres habituales ======

def detectar_contornos(binary, min_area_ratio=0.005):
    """Devuelve contornos externos relevantes (lista ordenada por área desc)"""
    h, w = binary.shape
    area_min = min_area_ratio * (h * w)
    contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [c for c in contornos if cv2.contourArea(c) >= area_min]
    contornos.sort(key=cv2.contourArea, reverse=True)
    return contornos

def dibujar_contornos(img, contornos, color=(0, 255, 0), grosor=2, bbox=True):
    """Dibuja contornos (y bbox opcional)"""
    img_dibujada = img.copy()
    cv2.drawContours(img_dibujada, contornos, -1, color, grosor)
    if bbox:
        for c in contornos:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_dibujada, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return img_dibujada