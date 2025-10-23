import cv2
import numpy as np

def escalar(imagen, max_ancho=1280, max_alto=720):
    h, w = imagen.shape[:2]
    s = min(max_ancho / w, max_alto / h, 1.0)
    if s < 1.0:
        imagen = cv2.resize(imagen, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return imagen

def preprocesar_imagen(img, fuerza=1.2, radio_borde=0.03, sigma_ilum=0.2):
    """
    Pasa a gris y atenúa la sombra del fondo SIN alterar el tono del elemento.
    Devuelve: (gray, gray_atenuado)
    - fuerza: 0.6–1.4 cuánta corrección aplicar en el fondo
    - radio_borde: ~0.01–0.02 del lado menor; zona donde NO corregimos (borde del objeto)
    - sigma_ilum: ~0.08–0.15 del lado menor; tamaño para estimar iluminación de fondo
    """
    # Gris original para comparar/salir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) LAB y canal L (luminancia) para no tocar cromas
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    h, w = L.shape

    # Tamaños relativos (robustos a distintos tamaños)
    k_dil = max(7,  (int(min(h, w) * 0.0125) | 1))  # ≈ lado*1.25%
    k_med = max(21, (int(min(h, w) * sigma_ilum) | 1))  # controla “lo grande” de la sombra
    r_borde = max(3, int(min(h, w) * radio_borde))

    # 2) Estimación de iluminación (método tipo snippet: dilate+median)
    dilated = cv2.dilate(L, np.ones((k_dil, k_dil), np.uint8))
    bg      = cv2.medianBlur(dilated, k_med)
    diff    = 255 - cv2.absdiff(L, bg)
    L_corr  = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3) Máscara de proximidad al borde: no corregir pegado al objeto
    edges = cv2.Canny(L, 50, 150)
    Kb    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_borde+1, 2*r_borde+1))
    prox  = cv2.dilate(edges, Kb, iterations=1)  # borde engrosado
    wmap  = 1.0 - cv2.GaussianBlur(prox/255.0, (0,0), r_borde/2)  # 1=fondo, 0=borde
    wmap  = np.clip(wmap, 0.0, 1.0).astype(np.float32)

    # 4) Fusión: aplicar corrección solo en el fondo, con “fuerza” ajustable
    Lf = L.astype(np.float32)
    Lc = L_corr.astype(np.float32)
    L_blend = Lf + wmap * fuerza * (Lc - Lf)          # preserva tono en el objeto
    L_blend = np.clip(L_blend, 0, 255).astype(np.uint8)

    # 5) Suavizado leve para estabilizar
    L_final = cv2.GaussianBlur(L_blend, (5, 5), 0)

    # 6) Reconstrucción y salida en gris atenuado
    lab_final     = cv2.merge((L_final, A, B))
    img_corregida = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    gray_atenuado = cv2.cvtColor(img_corregida, cv2.COLOR_BGR2GRAY)

    return gray, gray_atenuado

def binarizar_robusto(gray_atenuado):
    """Binarización combinando varios métodos"""
    binary_gauss = cv2.adaptiveThreshold(
        gray_atenuado, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=8
    )

    binary_mean = cv2.adaptiveThreshold(
        gray_atenuado, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=8
    )

    blur = cv2.GaussianBlur(gray_atenuado, (5, 5), 0) 
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