import cv2
import numpy as np

# === Nombre del archivo (modificá según la imagen que quieras probar) ===
nombre_imagen = "base_datos_0/IMG_2794.JPG"

# === Leer la imagen ===
img = cv2.imread(nombre_imagen)

# === Conversión a escala de grises ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === Aplicar filtro Gaussiano ===
# kernel 5x5 → suavizado leve; aumentá a (7,7) o (9,9) si querés más suavizado
gray_gauss = cv2.GaussianBlur(gray, (9, 9), 0)

# === (nuevo) contorno exterior por bordes ===
# realza aristas de la silueta (hexágono)
edges = cv2.Canny(gray_gauss, 60, 150)  # puedes probar (50,120) o (70,170)
edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

# restringimos la búsqueda de contornos a la zona del objeto para evitar ruido
# (usa tu máscara de masa; si aún no existe, lo haremos más adelante)


# === 2.5) Ecualización del histograma (mejora de contraste) ===
gray_eq = cv2.equalizeHist(gray_gauss)

# gray_eq = cv2.medianBlur(gray_eq, 3) # <- agrega esta línea (anti-ruido fino)

# === 3) Binarización adaptativa (robusta frente a sombras) ===
binary = cv2.adaptiveThreshold(
    gray_eq, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,   # <<< ANTES: THRESH_BINARY
    31, 5
)


# (opcional, ayuda contra puntitos muy finos)
binary = cv2.medianBlur(binary, 3)

# === 4) Limpieza morfológica ===
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # <<< antes (5,5)
binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)



# === 5) Conservar exterior + agujeros SOLO del contorno principal ===
cnts, hier = cv2.findContours(binary_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
mask = np.ones_like(binary_clean) * 255  # fondo blanco

if hier is not None and len(cnts) > 0:
    hier = hier[0]  # (N,4): [Next, Prev, FirstChild, Parent]
    externos = [i for i, h in enumerate(hier) if h[3] == -1]
    if externos:
        i_max = max(externos, key=lambda i: cv2.contourArea(cnts[i]))
        # 1) exterior en NEGRO (relleno)
        cv2.drawContours(mask, cnts, i_max, 0, thickness=-1)
        # 2) agujeros (hijos directos) en BLANCO (relleno)
        for i, h in enumerate(hier):
            if h[3] == i_max:
                cv2.drawContours(mask, cnts, i, 255, thickness=-1)

binary_final = mask  # objeto negro, agujeros blancos, fondo blanco

# === (nuevo) extraer contorno exterior real usando edges, pero solo cerca del objeto ===
# dilatamos la masa (objeto blanco en mask_hu) para definir una ROI alrededor
mask_hu = 255 - binary_final          # objeto blanco, fondo negro (para Hu y ROI)
roi = cv2.dilate(mask_hu, np.ones((21,21), np.uint8), iterations=1)  # margen de seguridad
edges_roi = cv2.bitwise_and(edges, edges, mask=roi)

cnts_edge, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
outer_hex_mask = np.zeros_like(binary_final)
num_vertices = None
circularidad_exterior = None

if cnts_edge:
    c_out = max(cnts_edge, key=cv2.contourArea)
    # aproximación poligonal: epsilon = 2% del perímetro (ajustable 0.015–0.03)
    per = cv2.arcLength(c_out, True)
    approx = cv2.approxPolyDP(c_out, 0.02 * per, True)
    num_vertices = len(approx)

    # dibujar la silueta exterior como relleno (blanca en mask_hu_ext)
    mask_hu_ext = np.zeros_like(mask_hu)
    cv2.drawContours(mask_hu_ext, [c_out], -1, 255, thickness=-1)

    # circularidad del exterior (útil para círculo vs hexágono)
    area_out = cv2.contourArea(c_out)
    circularidad_exterior = (4*np.pi*area_out) / (per**2 + 1e-12)

    # si querés ver esta máscara exterior sola:
    outer_hex_mask = 255 - mask_hu_ext  # objeto negro, fondo blanco

# --- debug opcional en consola ---
print("Vértices exterior (aproxPolyDP):", num_vertices)
print("Circularidad exterior:", None if circularidad_exterior is None else f"{circularidad_exterior:.3f}")

M = cv2.moments(mask_hu, binaryImage=True)
hu = cv2.HuMoments(M).flatten()
hu_log = -np.sign(hu) * np.log10(np.abs(hu)+1e-12)
print("Hu (log):", hu_log)




# === Función para escalar proporcionalmente ===
def escalar(imagen, max_ancho=1280, max_alto=720):
    alto, ancho = imagen.shape[:2]
    escala = min(max_ancho / ancho, max_alto / alto, 1.0)  # nunca agrandar
    if escala < 1.0:
        nuevo_tamaño = (int(ancho * escala), int(alto * escala))
        imagen = cv2.resize(imagen, nuevo_tamaño, interpolation=cv2.INTER_AREA)
    return imagen

# Escalar ambas imágenes solo para mostrar
img_vista = escalar(img)
gray_vista = escalar(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
gauss_vista = escalar(cv2.cvtColor(gray_gauss, cv2.COLOR_GRAY2BGR))
binary_vista = escalar(cv2.cvtColor(binary_final, cv2.COLOR_GRAY2BGR))


# === Mostrar resultados ===
cv2.imshow("Original", img_vista)
#cv2.imshow("Escala de grises", gray_vista)
#cv2.imshow("Filtro Gaussiano", gauss_vista)
cv2.imshow("Binarización final", binary_vista)


contours, _ = cv2.findContours(255 - binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Contornos encontrados:", len(contours))
if contours:
    print("Área contorno mayor:", cv2.contourArea(max(contours, key=cv2.contourArea)))



# Esperar a que el usuario presione el cero para terminar el programa
cv2.waitKey(0)
cv2.destroyAllWindows()