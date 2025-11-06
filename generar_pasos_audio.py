import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Importar toda la lógica de tus scripts ---
try:
    # Lógica de pre-procesamiento
    from procesar_audio import (
        smart_read_any, 
        rational_resample, 
        bandpass_filter,
        apply_notch, 
        trim_silence, 
        normalize_rms, 
        fix_duration,
        TARGET_SR, 
        NOTCH_Q
    )
    # Lógica de extracción de características
    from cualidades_audio import (
        find_voiced_region, 
        split_segments, 
        feats_segment, 
        N_SEG
    )
    # Lógica de clasificación
    from predecir_audio import (
        cargar_modelo as cargar_modelo_audio,
        predecir_knn
    )
except ImportError as e:
    print(f"Error: Asegúrate de que 'procesar_audio.py', 'cualidades_audio.py' y 'predecir_audio.py' estén en la misma carpeta.")
    print(f"Detalle: {e}")
    sys.exit(1)

def generar_y_mostrar_pasos_audio(audio_path: Path, modelo: dict):
    """
    Carga un audio y muestra las 4 etapas clave del procesamiento
    para tomar capturas de pantalla para el informe.
    """
    
    print("\n--- INICIO DEL PROCESO ---")

    # --- ETAPA 1: Cargar Audio Original ---
    print("\n[ETAPA 1: Carga de Audio Original]")
    try:
        # Usamos smart_read_any de tu script
        y, sr = smart_read_any(audio_path, ffmpeg_bin=None) 
        print(f"Audio cargado. Sample Rate: {sr} Hz, Muestras: {len(y)}")
        
        # Graficar Audio Original
        plt.figure(figsize=(10, 4))
        # Usamos librosa (si está disponible) o numpy para graficar
        try:
            import librosa.display
            librosa.display.waveshow(y, sr=sr, alpha=0.8)
        except ImportError:
            time = np.arange(len(y)) / float(sr)
            plt.plot(time, y)
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud")
            
        plt.title("1. Audio Original (Forma de Onda)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        print("Mostrando gráfico 1. Cierra la ventana del gráfico para continuar...")
        plt.show()

    except Exception as e:
        print(f"Error al cargar el audio: {e}")
        return

    # --- ETAPA 2: Pre-procesamiento ---
    print("\n[ETAPA 2: Pre-procesamiento del Audio]")
    
    # Este es el pipeline de procesar_audio.py, paso a paso
    y_proc = rational_resample(y, sr, TARGET_SR)
    y_proc = bandpass_filter(y_proc, TARGET_SR, hp_hz=100.0, lp_hz=5000.0)
    y_proc = apply_notch(y_proc, TARGET_SR, 50.0, NOTCH_Q)
    
    # 
    # === ESTA ES LA LÍNEA CORREGIDA ===
    #
    y_proc = trim_silence(y_proc, sr=TARGET_SR) # <-- Le faltaba el sr=TARGET_SR
    #
    # === FIN DE LA CORRECCIÓN ===
    #
    
    y_proc = normalize_rms(y_proc, target_dbfs=-20.0)
    y_final = fix_duration(y_proc, TARGET_SR, target_sec=1.20)
    
    print("Audio filtrado, recortado, normalizado y con duración fija.")

    # Graficar Audio Procesado
    plt.figure(figsize=(10, 4))
    try:
        import librosa.display
        librosa.display.waveshow(y_final, sr=TARGET_SR, alpha=0.8, color='green')
    except ImportError:
        time = np.arange(len(y_final)) / float(TARGET_SR)
        plt.plot(time, y_final, color='green')
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud Normalizada")

    plt.title("2. Audio Procesado (Limpio y Normalizado)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    print("Mostrando gráfico 2. Cierra la ventana del gráfico para continuar...")
    plt.show()

    # --- ETAPA 3: Extracción de Características ---
    print("\n[ETAPA 3: Extracción de Características]")
    
    # Esta es la lógica de cualidades_audio.py
    i0, i1 = find_voiced_region(y_final, TARGET_SR)
    y_voiced = y_final[i0:i1] if (i1 - i0) > TARGET_SR * 0.1 else y_final
    segs = split_segments(y_voiced, n_seg=N_SEG)
    
    features_vector = []
    for seg in segs:
        features_vector.extend(feats_segment(seg, sr=TARGET_SR))
    
    features_vector = np.array(features_vector, dtype=np.float32)
    
    print("Vector de características extraído.")
    print(f"  -> Dimensiones del vector: {features_vector.shape}")
    print(f"  -> Primeras 5 características (de 140): {features_vector[:5]}")
    
    input("\nPresiona Enter para continuar con la clasificación...")

    # --- ETAPA 4: Clasificación K-NN ---
    print("\n[ETAPA 4: Clasificación K-NN]")
    
    # Esta es la lógica de predecir_audio.py
    comando, dist_promedio = predecir_knn(features_vector, modelo)
    
    print("¡Clasificación completa!")
    print(f"  -> Comando detectado: {comando}")
    print(f"  -> Distancia promedio a vecinos: {dist_promedio:.4f}")

    print("\n--- FIN DEL PROCESO ---")

# --- Bloque para ejecutar el script ---
if __name__ == "__main__":
    # 1. Cargar el modelo K-NN
    model_path = Path("models/knn_audio_puro.npz")
    if not model_path.exists():
        print(f"Error: No se encuentra el modelo K-NN en: {model_path}")
        sys.exit(1)
        
    try:
        modelo_knn = cargar_modelo_audio(model_path)
        print("✅ Modelo K-NN cargado correctamente.")
    except Exception as e:
        print(f"Error fatal al cargar el modelo K-NN: {e}")
        sys.exit(1)

    # 2. Pedir la ruta del audio
    ruta_audio = input("Ingresa la ruta completa del archivo de audio que quieres procesar: ")
    p = Path(ruta_audio)
    
    if not p.exists():
        print(f"Error: No se encuentra el archivo en: {p}")
    else:
        generar_y_mostrar_pasos_audio(p, modelo_knn)