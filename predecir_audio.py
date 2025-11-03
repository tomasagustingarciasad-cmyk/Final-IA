"""
Predice un comando de voz desde un archivo de audio.
Aplica el MISMO pipeline de preprocesamiento y extracción de features
que se usó para el entrenamiento.

Uso: python predecir_audio.py ruta/a/mi_audio.wav
"""
import numpy as np
import soundfile as sf
from pathlib import Path
import sys
from collections import Counter, defaultdict

# Importamos las funciones de los otros scripts para reutilizar la lógica
# Asegúrate de que estos archivos estén en la misma carpeta
try:
    from procesar_audio import (
        to_mono,
        rational_resample,
        bandpass_filter,
        apply_notch,
        trim_silence,
        normalize_rms,
        fix_duration,
        NOTCH_Q,
        smart_read_any, 
    )
    from cualidades_audio import (
        find_voiced_region,
        split_segments,
        feats_segment,
        TARGET_SR,
        N_SEG,
    )
except ImportError:
    print("❌ Error: Asegúrate de que 'procesar_audio.py' y 'cualidades_audio.py' estén en la misma carpeta.")
    sys.exit(1)

# ===============================
# Configuración
# ===============================
MODEL_PATH = Path("models/knn_audio_puro.npz")

# ===============================
# Cargar Modelo
# ===============================
def cargar_modelo(model_path: Path): # ✅ AÑADIR el parámetro model_path
    """Carga el modelo K-NN entrenado y los metadatos de normalización."""
    if not model_path.exists(): # ✅ USAR el parámetro
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    data = np.load(model_path, allow_pickle=True) # ✅ USAR el parámetro
    # Validar que las claves esperadas existan
    expected_keys = {'X_train', 'y_train', 'mean', 'std', 'k', 'p', 'weighted'}
    if not expected_keys.issubset(data.files):
        raise KeyError(f"Faltan claves en el modelo. Encontradas: {list(data.files)}")

    return {
        'X_train': data['X_train'],
        'y_train': data['y_train'],
        'mean': data['mean'],
        'std': data['std'],
        'k': int(data['k']),
        'p': float(data['p']),
        'weighted': bool(data['weighted'])
    }

# ===============================
# Pipeline Completo (Procesamiento + Extracción)
# ===============================
def pipeline_completo_audio(audio_path: Path) -> np.ndarray:
    """
    Aplica el pipeline completo a UN archivo de audio.
    1. Lee y normaliza (procesar_audio.py)
    2. Extrae features (cualidades_audio.py)
    Retorna un vector de 140 features.
    """
    # --- 1. Leer y Normalizar (lógica de procesar_audio.py) ---
    # ✅ 2. USAR LA FUNCIÓN INTELIGENTE
    y, sr = smart_read_any(audio_path, ffmpeg_bin=None)

    # ✅ 3. SIMPLIFICAR: smart_read_any ya convierte a mono,
    #    así que la siguiente línea ya no es necesaria.
    # y = to_mono(y) 
    
    if sr != TARGET_SR:
        y = rational_resample(y, sr, TARGET_SR)
    
    y = bandpass_filter(y, TARGET_SR, hp_hz=100.0, lp_hz=5000.0)
    y = apply_notch(y, TARGET_SR, 50.0, NOTCH_Q)
    y = trim_silence(y, TARGET_SR)
    y = normalize_rms(y, target_dbfs=-20.0)
    y_norm = fix_duration(y, TARGET_SR, target_sec=1.20)

    # --- 2. Extraer Features (lógica de cualidades_audio.py) ---
    i0, i1 = find_voiced_region(y_norm, TARGET_SR)
    y_voiced = y_norm[i0:i1] if (i1 - i0) > TARGET_SR * 0.1 else y_norm
    
    segs = split_segments(y_voiced, n_seg=N_SEG)
    
    features_vector = []
    for seg in segs:
        features_vector.extend(feats_segment(seg, sr=TARGET_SR))
    
    # Verificación de consistencia
    expected_len = 14 * N_SEG
    if len(features_vector) != expected_len:
        raise ValueError(f"Se extrajeron {len(features_vector)} features, se esperaban {expected_len}.")
        
    return np.array(features_vector, dtype=np.float32)

# ===============================
# Predicción KNN
# ===============================
def predecir_knn(features: np.ndarray, modelo: dict) -> tuple[str, float]:
    """Predice la clase usando el modelo K-NN puro."""
    # Normalizar features con la media y std del entrenamiento
    features_norm = (features - modelo['mean']) / (modelo['std'] + 1e-8)
    
    # Calcular distancias (Minkowski)
    X_train = modelo['X_train']
    p = modelo['p']
    
    if p == 2:
        distancias = np.sqrt(((X_train - features_norm) ** 2).sum(axis=1))
    elif p == 1:
        distancias = np.abs(X_train - features_norm).sum(axis=1)
    else:
        distancias = (np.abs(X_train - features_norm) ** p).sum(axis=1) ** (1/p)
    
    # K vecinos más cercanos
    k = modelo['k']
    idx_vecinos = np.argsort(distancias)[:k]
    clases_vecinos = modelo['y_train'][idx_vecinos]
    distancias_vecinos = distancias[idx_vecinos]
    
    # Votación (ponderada si weighted=True)
    if modelo['weighted']:
        pesos = 1.0 / (distancias_vecinos + 1e-8)
        acumulado = defaultdict(float)
        for cls, w in zip(clases_vecinos, pesos):
            acumulado[cls] += w
        clase_pred = max(acumulado, key=acumulado.get)
    else:
        clase_pred = Counter(clases_vecinos).most_common(1)[0][0]
    
    return clase_pred, float(np.mean(distancias_vecinos))

# ===============================
# Main
# ===============================
def main():
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} ruta/a/audio.wav")
        sys.exit(1)
    
    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"❌ Audio no encontrado: {audio_path}")
        sys.exit(1)
    
    print(f"🎤 Procesando audio: {audio_path.name}")
    
    try:
        # 1. Cargar modelo
        modelo = cargar_modelo()
        print(f"✅ Modelo cargado: K={modelo['k']}, Clases={np.unique(modelo['y_train'])}")
        
        # 2. Aplicar pipeline completo para extraer features
        print("🔄 Aplicando pipeline de procesamiento y extracción de features...")
        features = pipeline_completo_audio(audio_path)
        print(f"✅ Vector de {len(features)} features extraído.")
        
        # 3. Predecir
        clase, dist_promedio = predecir_knn(features, modelo)
        
        # 4. Reportar
        print(f"\n🎯 Predicción: {clase}")
        print(f"   Distancia promedio a vecinos: {dist_promedio:.4f}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()