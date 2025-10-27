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

def preprocesar_imagen(img, fuerza=None, radio_borde=0.02, sigma_ilum=0.16,
                       target_bg=210, gamma_borde=1.7, delta_max_px=60):
    """
    Devuelve (gray, gray_atenuado). Corrige sombras del fondo preservando tono
    del metal y con seguridad anti-'engorde'.

    - fuerza=None  -> se autoajusta por imagen (recomendado)
      (si pasás un valor, se respeta pero se clampea internamente)
    - radio_borde: 0.02–0.03 para tornillos pequeños; 0.015–0.02 si ocupan gran parte
    - sigma_ilum: 0.12–0.20 (>=0.18 gatilla ruta 'downscale' de la mediana)
    """
    # 0) Grises y L* en LAB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    h, w = L.shape

    # --- tamaños
    k_dil = max(7, (int(min(h, w) * 0.0125) | 1))
    k_raw = int(min(h, w) * sigma_ilum) | 1

    # 1) Estimar fondo con mediana 'grande' sin romper AVX (k<16)
    dilated = cv2.dilate(L, np.ones((k_dil, k_dil), np.uint8))
    if k_raw <= 15:
        bg = cv2.medianBlur(dilated, max(3, k_raw))
    else:
        s = max(15.0 / k_raw, 0.1)  # factor de reducción
        small = cv2.resize(dilated, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv2.INTER_AREA)
        bg_sm = cv2.medianBlur(small, 15)
        bg = cv2.resize(bg_sm, (w, h), interpolation=cv2.INTER_LINEAR)

    diff   = 255 - cv2.absdiff(L, bg)
    L_corr = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2) Máscara de sombra real (sólo aclarar donde bg>L)
    shadow = cv2.subtract(bg, L).astype(np.int16)
    shadow[shadow < 0] = 0
    tau = max(8, int(np.percentile(shadow, 90) * 0.25))
    mask_shadow = (shadow > tau).astype(np.float32)
    r_shadow = max(5, int(min(h, w) * 0.02))
    mask_shadow = cv2.GaussianBlur(mask_shadow, (0, 0), r_shadow)

    # 3) Protección de borde (más dura, usando exponente gamma)
    r_borde_px = max(3, int(min(h, w) * radio_borde))
    edges = cv2.Canny(L, 50, 150)
    Kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_borde_px+1, 2*r_borde_px+1))
    prox = cv2.dilate(edges, Kb, iterations=1)  # 255 en contorno grueso
    w_borde = 1.0 - cv2.GaussianBlur(prox/255.0, (0, 0), r_borde_px/2.0)
    w_borde = np.clip(w_borde, 0.0, 1.0) ** gamma_borde  # cae más cerca del borde

    # 4) Peso final: sólo fondo y lejos del borde
    wmap = (mask_shadow * w_borde).astype(np.float32)

    # 5) Auto-fuerza (opcional) para unificar fondo
    Lf = L.astype(np.float32)
    Lc = L_corr.astype(np.float32)
    delta = np.clip(Lc - Lf, 0, float(delta_max_px))  # tope por píxel

    if fuerza is None:
        # fondo = lejos de bordes (w_borde alto) y con sombra (mask_shadow alto)
        bgmask = (wmap > 0.4).astype(np.uint8)
        n = cv2.countNonZero(bgmask)
        if n > 0:
            mean_bg  = float(np.mean(L[bgmask == 1]))
            mean_cor = float(np.mean((Lf + delta)[bgmask == 1]))
            disp     = max(1.0, mean_cor - mean_bg)  # cuánto puedo subir
            need     = float(target_bg - mean_bg)
            auto_f   = np.clip(need / disp, 0.9, 1.5)  # límites seguros
        else:
            auto_f = 1.2  # fallback
    else:
        auto_f = float(np.clip(fuerza, 0.9, 1.5))  # clamp


    # --- Extra: limpieza suave de fondo (anti manchas de sombra difusa) ---
    fondo_mask = (wmap > 0.3).astype(np.uint8)  # zonas seguras lejos del borde
    if np.count_nonzero(fondo_mask) > 0:
        # aplicar mediana grande sobre fondo, mantener bordes
        k_suave = max(9, int(min(h, w) * 0.025)) | 1
        fondo_mediana = cv2.medianBlur(L_corr, k_suave)
        # combinar: donde sea fondo, promediar suavemente con la mediana
        L_corr = cv2.addWeighted(L_corr, 0.7, fondo_mediana, 0.3, 0)
        # re-normalizar a 8-bit
        L_corr = cv2.normalize(L_corr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 6) Fusión
    L_blend = Lf + wmap * auto_f * delta
    L_blend = np.clip(L_blend, 0, 255).astype(np.uint8)

    # 7) Suavizado leve y salida
    L_final = cv2.GaussianBlur(L_blend, (5, 5), 0)
    lab_out = cv2.merge((L_final, A, B))
    img_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    gray_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
    return gray, gray_out


# ============== 2) Binarización robusta (selección automática) ==============

def binarizar_robusto(gray_atenuado):
    """
    Probamos varias máscaras (adaptativas y Otsu, con y sin inversión) y
    elegimos la que deja mayor componente externo. Fallback con Canny.
    """
    h, w = gray_atenuado.shape
    pre = cv2.GaussianBlur(gray_atenuado, (5, 5), 0)

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
    if k % 2 == 0:
        k += 1
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


def deslumbrado_anillo(
    gray, mask_obj,
    ring_px=3,          # grosor de la banda interna (2–4 px suelen andar)
    k_rel=0.03,         # tamaño relativo del kernel p/ top-hat (2–4% del lado menor)
    frac=0.55,          # cuánto “restar” del brillo detectado (0.4–0.7)
    trigger_sigma=2.0,  # umbral: píxeles > (media + sigma*std) en la banda
    min_ratio=0.025,    # activar solo si ≥ 2.5% de la banda está “quemada”
    max_ratio=0.20      # seguridad: si >20% está brillando, no tocar (podría ser error)
):
    """
    Suprime brillos especulares en el borde de piezas metálicas.
    Actúa SOLO dentro de una banda interna del borde de 'mask_obj' y SOLO si
    hay suficiente brillo. Devuelve un gris corregido (uint8) del mismo tamaño.

    - gray: imagen en gris (uint8) – usá gray_atenuado o gray original
    - mask_obj: binario del objeto (después de componente_principal)
    """
    h, w = gray.shape

    # Banda interna (ANILLO): máscara - erosión
    r = max(1, int(ring_px))
    Kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    inner = cv2.subtract(mask_obj, cv2.erode(mask_obj, Kr, iterations=1))
    area_inner = cv2.countNonZero(inner)
    if area_inner == 0:
        return gray

    # Medimos brillo fuerte en la banda (estadística local)
    vals = gray[inner.astype(bool)]
    if vals.size:
        mu, sigma = float(vals.mean()), float(vals.std())
    else:
        mu, sigma = 0.0, 0.0
    thr = mu + trigger_sigma * sigma

    hot = np.zeros_like(gray, np.uint8)
    hot[(gray > thr) & (inner > 0)] = 255

    ratio_hot = cv2.countNonZero(hot) / float(area_inner)
    if ratio_hot < min_ratio or ratio_hot > max_ratio:
        # Muy poco brillo ⇒ no hace falta; demasiado ⇒ podría ser falso positivo
        return gray

    # White Top-Hat para aislar puntas especulares finas
    k = max(3, int(min(h, w) * k_rel))
    if k % 2 == 0:
        k += 1
    k = min(k, 31)  # cap por seguridad
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, K)
    tophat = cv2.subtract(gray, opened)          # zonas “demasiado blancas”
    tophat = cv2.bitwise_and(tophat, inner)      # limitar al anillo interno

    # Corrección (resto una fracción del top-hat solo en el anillo)
    g32 = gray.astype(np.float32)
    th32 = tophat.astype(np.float32)
    g_fix = g32 - frac * th32
    g_fix = np.clip(g_fix, 0, 255).astype(np.uint8)

    return g_fix


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


def rellenar_mordidas(mask, max_depth_px=8):
    # 1) mayor contorno
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    c = max(cnts, key=cv2.contourArea)

    # 2) casco convexo (hull) rasterizado
    hull = cv2.convexHull(c)
    hull_mask = np.zeros_like(mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    # 3) “gap” = zonas dentro del hull pero fuera de la máscara (son las concavidades)
    gap = cv2.bitwise_and(hull_mask, cv2.bitwise_not(mask))

    # 4) profundidad de concavidad por distancia a la frontera de la pieza
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)

    # 5) rellenar solo concavidades superficiales (≤ max_depth_px)
    fill = (gap > 0) & (dist <= max_depth_px)

    out = mask.copy()
    out[fill] = 255
    return out
