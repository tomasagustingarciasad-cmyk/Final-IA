# calcular_momentos.py
import cv2
import numpy as np
from procesado_imagen import preprocesar_imagen, binarizar_robusto, morfologia_conservadora

def calcular_momentos_hu(img_path):
    """Calcula los momentos de Hu de una imagen binarizada"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")

    gray = preprocesar_imagen(img)
    binary = binarizar_robusto(gray)
    binary = morfologia_conservadora(binary)

    # Encontrar contornos principales
    contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        print("No se encontraron contornos.")
        return None

    c = max(contornos, key=cv2.contourArea)
    momentos = cv2.moments(c)
    hu = cv2.HuMoments(momentos).flatten()

    # Transformación logarítmica para estabilizar valores
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return hu_log

if __name__ == "__main__":
    ruta = "base_datos/Tornillo/Tornillo_1.JPG"
    hu = calcular_momentos_hu(ruta)
    if hu is not None:
        print("Momentos de Hu (log-transformados):")
        for i, val in enumerate(hu, start=1):
            print(f"Hu{i}: {val:.6f}")
