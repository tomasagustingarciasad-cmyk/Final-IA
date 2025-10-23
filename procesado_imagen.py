import cv2
import numpy as np

# ================= Utilidad =================

def escalar(imagen, max_ancho=1280, max_alto=720):
    h, w = imagen.shape[:2]
    s = min(max_ancho / w, max_alto / h, 1.0)
    if s < 1.0:
        imagen = cv2.resize(imagen, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return imagen


# ============== 1) Preprocesado: gris + atenuación de sombra ==============

def preprocesar_imagen(img, fuerza=1.2, radio_borde=0.025, sigma_ilum=0.20):
    """
    Devuelve (gray, gray_atenuado): gris original y gris con sombra atenuada,
    preservando el tono del metal.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    h, w = L.shape

    # Tamaños relativos
    k_dil = max(7,  (int(min(h, w) * 0.0125) | 1))
    k_med = max(21, (int(min(h, w) * sigma_ilum) | 1))
    r_borde = max(3, int(min(h, w) * radio_borde))

    # Iluminación de fondo (dilate + median)
    dilated = cv2.dilate(L, np.ones((k_dil, k_dil), np.uint8))
    bg      = cv2.medianBlur(dilated, k_med)
    diff    = 255 - cv2.absdiff(L, bg)
    L_corr  = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Máscara de sombra (bg > L)
    shadow_raw  = cv2.subtract(bg, L).astype(np.int16)
    shadow_raw[shadow_raw < 0] = 0
    tau = max(8, int(np.percentile(shadow_raw, 90) * 0.25))
    mask_shadow = (shadow_raw > tau).astype(np.float32)
    r_shadow = max(5, int(min(h, w) * 0.02))
    mask_shadow = cv2.GaussianBlur(mask_shadow, (0, 0), r_shadow)

    # Protección de borde
    edges = cv2.Canny(L, 50, 150)
    Kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_borde+1, 2*r_borde+1))
    prox = cv2.dilate(edges, Kb, iterations=1)
    wmap_borde = 1.0 - cv2.GaussianBlur(prox/255.0, (0, 0), r_borde/2)
    wmap_borde = np.clip(wmap_borde, 0.0, 1.0)

    # Corregir sólo donde hay sombra y lejos del borde
    wmap = np.maximum(wmap_borde, mask_shadow).astype(np.float32)

    # Fusión con tope por píxel
    Lf = L.astype(np.float32)
    Lc = L_corr.astype(np.float32)
    delta = np.clip(Lc - Lf, 0, 60)
    L_blend = Lf + wmap * fuerza * delta
    L_blend = np.clip(L_blend, 0, 255).astype(np.uint8)

    # Suavizado leve
    L_final = cv2.GaussianBlur(L_blend, (5, 5), 0)

    lab_final = cv2.merge((L_final, A, B))
    img_corregida = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    gray_atenuado = cv2.cvtColor(img_corregida, cv2.COLOR_BGR2GRAY)

    return gray, gray_atenuado


# ============== 2) Binarización robusta (selección automática) ==============

def binarizar_robusto(gray_atenuado):
    """
    Probamos varias máscaras (adaptativas y Otsu, con y sin inversión) y
    elegimos la que deja mayor componente externo. Fallback con Canny.
    """
    h, w = gray_atenuado.shape
    pre = cv2.GaussianBlur(gray_atenuado, (7, 7), 0)

    blockSize = max(61, (min(h, w) // 14) | 1)  # valor que te funcionaba bien
    C = 4

    bin_gauss_inv = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, blockSize, C)
    bin_mean_inv  = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV, blockSize, C)
    _, bin_otsu_inv = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bin_gauss = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, blockSize, C)
    bin_mean  = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, blockSize, C)
    _, bin_otsu = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidatas = [bin_gauss_inv, bin_mean_inv, bin_otsu_inv, bin_gauss, bin_mean, bin_otsu]

    def score(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        return max(cv2.contourArea(c) for c in cnts) / (h * w)

    scores = [score(m) for m in candidatas]
    idx = int(np.argmax(scores))
    binary = candidatas[idx]
    best = scores[idx]

    if best < 0.0008:
        edges = cv2.Canny(pre, 40, 120)
        K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, K, iterations=2)
        binary = np.zeros_like(edges)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(binary, cnts, -1, 255, thickness=cv2.FILLED)

    binary = cv2.medianBlur(binary, 5)
    if cv2.countNonZero(binary) > (binary.size // 2):
        binary = cv2.bitwise_not(binary)

    return binary


# ============== 3) Limpieza morfológica y refinado para Hu ==============

def morfologia_conservadora(binary):
    h, w = binary.shape
    area_ratio = cv2.countNonZero(binary) / float(h * w)

    # kernel un poco menor y siempre impar
    k = max(3, int(min(h, w) * 0.012))
    if k % 2 == 0: k += 1
    k = min(k, 21)
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Si el objeto es pequeño, no hagas OPEN (suele “morder” el borde)
    if area_ratio < 0.03:
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, K, iterations=2)
    else:
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  K, iterations=1)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, K, iterations=2)

    return result



def componente_principal(binary):
    """Conserva sólo el componente conexo de mayor área."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(binary)
    out[labels == largest_label] = 255
    return out


def edge_guided_fix(
    gray, binary,
    r=2,                         # grosor de la banda exterior (2–3 px)
    min_edge_ratio=0.06,         # % de borde en la banda para activar
    canny_low=40, canny_high=120,
    cierre_iter=1,
    max_added_ratio=0.08         # guardrail: expansión máxima permitida
):
    """
    Corrige 'mordidas' del contorno usando el borde del gris original.
    Devuelve una máscara binaria del mismo tamaño que 'binary'.

    - No toca nada si en la banda exterior no hay borde suficiente.
    - A lo sumo agrega 'r' píxeles hacia afuera.
    """
    # --- 1) Borde robusto (gradiente morfológico + Canny + closing) ---
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, K3)
    _, e_grad = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    e_canny = cv2.Canny(g, canny_low, canny_high)

    edges = cv2.bitwise_or(e_grad, e_canny)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, K3, iterations=cierre_iter)

    # --- 2) Banda exterior alrededor de tu máscara actual ---
    Kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    dil = cv2.dilate(binary, Kr, iterations=1)
    ring = cv2.subtract(dil, binary)               # solo la corona exterior
    ring_nz = cv2.countNonZero(ring)
    if ring_nz == 0:
        return binary.copy()

    # --- 3) ¿Hay borde real en la banda? si no, no tocamos nada ---
    ring_edges = cv2.bitwise_and(edges, edges, mask=ring)
    ratio = cv2.countNonZero(ring_edges) / ring_nz
    if ratio < min_edge_ratio:
        return binary.copy()

    # --- 4) Expandir solo donde hay borde en la banda ---
    mask_edge = ring_edges.copy()

    # Guardrail: no crecer demasiado por ruido
    added_ratio = cv2.countNonZero(mask_edge) / max(1, cv2.countNonZero(binary))
    if added_ratio > max_added_ratio:
        return binary.copy()

    # Unión y salida (sin rellenar: eso lo hacés después como siempre)
    binary_fix = cv2.bitwise_or(binary, mask_edge)
    return binary_fix



def rellenar_huecos(binary):
    """
    Rellena huecos internos del objeto (ideal para tornillos con brillo).
    Para arandelas, desactivar para conservar el agujero.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


# ============== 4) Contornos (visualización) ==============

def detectar_contornos(binary, min_area_ratio=0.0001):
    """Contornos externos relevantes (ordenados por área desc)."""
    h, w = binary.shape
    area_min = min_area_ratio * (h * w)
    contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [c for c in contornos if cv2.contourArea(c) >= area_min]
    contornos.sort(key=cv2.contourArea, reverse=True)
    return contornos


def dibujar_contornos(img, contornos, color=(0, 255, 0), grosor=2, bbox=True):
    """Dibuja contornos y (opcional) su bounding box."""
    vis = img.copy()
    cv2.drawContours(vis, contornos, -1, color, grosor)
    if bbox:
        for c in contornos:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return vis
