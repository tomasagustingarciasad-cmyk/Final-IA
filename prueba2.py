import cv2
import numpy as np

# === Función para escalar proporcionalmente ===
def escalar(imagen, max_ancho=1280, max_alto=720):
    alto, ancho = imagen.shape[:2]
    escala = min(max_ancho / ancho, max_alto / alto, 1.0)  # nunca agrandar
    if escala < 1.0:
        nuevo_tamaño = (int(ancho * escala), int(alto * escala))
        imagen = cv2.resize(imagen, nuevo_tamaño, interpolation=cv2.INTER_AREA)
    return imagen

# Tuerca
#nombre_imagen = "base_datos_0/IMG_2765.JPG"
# Tornillo 
nombre_imagen = "base_datos_0/IMG_2771.JPG"
# Arandela
# nombre_imagen = "base_datos_0/IMG_2859.JPG"
img = cv2.imread(nombre_imagen)

# --- Paso 1: Convertir a LAB ---
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# --- Paso 2: Reducción de reflejos intensos ---
# Detectar píxeles muy brillantes (posibles reflejos)
mask_reflex = cv2.inRange(L, 200, 255)  # ajustá 200 según la luz de tus fotos
# Reemplazar con un promedio local para suavizarlos
L_reflex_reduced = cv2.inpaint(L, mask_reflex, 5, cv2.INPAINT_TELEA)

# --- Paso 3: Suavizado bilateral sólo sobre L ---
L_suavizado = cv2.bilateralFilter(L_reflex_reduced, d=15, sigmaColor=80, sigmaSpace=80)

# --- Paso 3.5 Compresion logaritmica ---
L_float = L_suavizado.astype(np.float32) / 255.0
L_log = np.log1p(L_float) / np.log(2.0)
L_suavizado = cv2.normalize(L_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# --- Paso 4: Ecualización adaptativa (CLAHE) ---
clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
L_equalized = clahe.apply(L_suavizado)

# Recombinar LAB y convertir a escala de grises
lab_corr = cv2.merge((L_equalized, A, B))
corr_gray = cv2.cvtColor(lab_corr, cv2.COLOR_LAB2BGR)
corr_gray = cv2.cvtColor(corr_gray, cv2.COLOR_BGR2GRAY)

# --- Paso 5: Binarización adaptativa robusta ---
binary = cv2.adaptiveThreshold(
    corr_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31, 5
)
# === tras crear 'binary' ===
binary = cv2.medianBlur(binary, 3)
# === tras crear 'binary' y aplicar medianBlur ===
# Asegurar polaridad (pieza blanca, fondo negro)
if cv2.countNonZero(binary) > (binary.size // 2):
    binary = cv2.bitwise_not(binary)


# kernel proporcional y cierre (tu código actual):
h, w = binary.shape
k = max(5, min(11, (min(h, w)//250) | 1))
K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
work = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, K, iterations=2)
work = cv2.dilate(work, K, iterations=1)

cv2.imshow("Post-cierre (work)", escalar(cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)))


# Escalar ambas imágenes solo para mostrar
img_vista = escalar(img)
gauss_vista = escalar(cv2.cvtColor(corr_gray, cv2.COLOR_GRAY2BGR))
binary_vista = escalar(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

# --- Mostrar resultados ---
cv2.imshow("Original", img_vista)
cv2.imshow("Canal L corregido", gauss_vista)
cv2.imshow("Binarización final", binary_vista)
cv2.imshow("Post-cierre (work)", escalar(cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)))


# --- contornos con jerarquía sobre 'work' (ya cerrado) ---
cnts, hier = cv2.findContours(work, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

mask_solid = np.zeros_like(work)   # sin huecos
mask_holes = np.zeros_like(work)   # conserva huecos

if hier is not None and len(cnts) > 0:
    # 1) buscar los contornos EXTERNOS
    externos = [i for i in range(len(cnts)) if hier[0][i][3] == -1]
    # 2) quedarnos con el EXTERNO de MAYOR ÁREA (la pieza)
    idx_max = max(externos, key=lambda i: cv2.contourArea(cnts[i]))

    # --- máscara SÓLIDA: sólo el externo mayor relleno ---
    cv2.drawContours(mask_solid, [cnts[idx_max]], -1, 255, cv2.FILLED)

    # --- máscara con HUECOS: vaciar hijos de ese externo ---
    mask_holes[:] = mask_solid
    for i in range(len(cnts)):
        if hier[0][i][3] == idx_max:  # hijo del externo mayor = hueco
            cv2.drawContours(mask_holes, [cnts[i]], -1, 0, cv2.FILLED)
else:
    # Fallback si no se detectó jerarquía (raro)
    mask_solid = work.copy()
    mask_holes = work.copy()

# Ajuste fino para desengrosar un poco el borde si hizo falta
mask_solid = cv2.erode(mask_solid, K, iterations=1)
mask_holes  = cv2.erode(mask_holes,  K, iterations=1)

# --- Selección automática según topología (tiene hueco o no) ---
area_solida = cv2.countNonZero(mask_solid)
area_huecos = cv2.countNonZero(mask_holes)
tiene_hueco = (area_solida - area_huecos) > (0.002 * area_solida)  # 0.2% de diferencia

mask_final = mask_holes if tiene_hueco else mask_solid

# Mostrar todo para depurar
cv2.imshow("Máscara Sólida", escalar(cv2.cvtColor(mask_solid, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Máscara con Huecos", escalar(cv2.cvtColor(mask_holes, cv2.COLOR_GRAY2BGR)))
cv2.imshow("Máscara FINAL", escalar(cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR)))
print("tiene_hueco:", tiene_hueco)



cv2.waitKey(0)
cv2.destroyAllWindows()
