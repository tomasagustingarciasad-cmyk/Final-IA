# 📋 SEGUIMIENTO DEL TRABAJO FINAL - INTELIGENCIA ARTIFICIAL
## Sistema de Clasificación de Piezas con Visión Artificial y Comandos de Voz

---

## 📊 RESUMEN EJECUTIVO

**Fecha de análisis:** 5 de noviembre de 2025  
**Estado general:** � Avanzado - Sistema funcional completo (≈90% completado)

### Progreso por componentes:
- ✅ **Visión Artificial (K-Means Propio):** 100% completado - IMPLEMENTACIÓN PROPIA
- ✅ **Reconocimiento de Voz (K-NN Propio):** 100% completado - IMPLEMENTACIÓN PROPIA
- ✅ **Clasificador Bayesiano:** 100% completado - Con visualización en tiempo real
- ✅ **Integración del sistema:** 95% completado - App Flask funcional con ngrok
- � **Documentación y entregables:** 60% completado

---

## 🎯 ARQUITECTURA DEL SISTEMA IMPLEMENTADA

### Modelo Cliente-Servidor Completo

```
┌─────────────────────────────────────────────────────────┐
│  CELULAR (Frontend - HTML/JavaScript)                   │
│  ├─ Botón: "Sacar Foto" → Captura con cámara           │
│  ├─ Botón: "Grabar Comando" → Graba voz por 3 seg      │
│  └─ Muestra: Resultados, historial y conteos           │
└──────────────────┬──────────────────────────────────────┘
                   │ (HTTPS via ngrok)
                   ↓
┌─────────────────────────────────────────────────────────┐
│  SERVIDOR (Backend - Flask en PC)                       │
│  ├─ app.py: Orquestador principal                       │
│  ├─ /predict_image → Clasifica imagen con K-Means       │
│  ├─ /predict_command → Clasifica audio con K-NN         │
│  └─ Mantiene historial de 10 últimas piezas            │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│  MODELOS ENTRENADOS (Algoritmos Propios)                │
│  ├─ models/kmeans_puro2.npz (K-Means sin sklearn)      │
│  ├─ models/knn_audio_puro.npz (K-NN sin sklearn)       │
│  ├─ models/Mi_kmeans_mbkmeans2.joblib (metadata)       │
│  └─ models/Mi_scaler_kmeans2.joblib (escalado)         │
└─────────────────────────────────────────────────────────┘
```

### Flujo de Trabajo en Producción:
1. Usuario abre navegador en celular → https://[ngrok-url].ngrok-free.app
2. Saca foto de pieza → se clasifica con K-Means propio
3. Graba comando de voz → se clasifica con K-NN propio
4. Comando "Proporción" → activa Clasificador Bayesiano con gráfico 3D

---

## 🎯 REQUISITOS DEL TRABAJO PRÁCTICO (según PDF)

### Componentes principales requeridos:
1. ✅ **Agente inteligente** que procese imágenes y comandos de voz - COMPLETADO
2. ✅ **K-Means** para clasificación de piezas (Arandela, Tuerca, Tornillo, Clavo) - IMPLEMENTACIÓN PROPIA
3. ✅ **K-NN** para reconocimiento de comandos de voz (Contar, Proporción, Salir) - IMPLEMENTACIÓN PROPIA
4. ✅ **Clasificador Bayesiano** para estimación de proporciones en caja - COMPLETADO
5. ✅ **Sistema integrado** que combine todos los módulos - COMPLETADO
6. 🟡 **Documentación completa** y presentación - EN PROGRESO

---

## ✅ MÓDULO 1: ALGORITMO K-MEANS PROPIO (100% COMPLETADO)

### Estado: ✅ IMPLEMENTACIÓN DESDE CERO COMPLETADA

**Archivo principal:** `Mi_Kmeans.py`

#### ✅ Características implementadas (sin sklearn):
- [x] **Algoritmo MiniBatch K-Means** completo desde cero
- [x] Inicialización con **medoides** por clase (modo forzado)
- [x] Escalado interno **MinMax** integrado
- [x] Asignación 1-a-1 cluster→clase con algoritmo Hungarian/Greedy
- [x] Cálculo de inercia y distancias euclidianas manuales
- [x] Manejo de clusters vacíos con estrategia "farthest"
- [x] Múltiples corridas (n_init) para evitar mínimos locales
- [x] Convergencia por tolerancia de desplazamiento de centroides

#### 📊 Especificaciones técnicas:
```python
Características usadas: hu1_log, hu2_log, ar2 (3D)
K = 4 clusters (uno por clase)
Escalado: MinMax interno [(x - min) / (max - min)]
Inicialización: Medoides de cada clase
Batch size: 512 (adaptativo según dataset)
Max iteraciones: 30 (refinamiento corto)
Convergencia: tol = 1e-4
```

#### 🎯 Clases soportadas:
- **Arandela** (24 imágenes originales)
- **Tuerca** (39 imágenes originales)
- **Tornillo** (49 imágenes originales)
- **Clavo** (29 imágenes originales)
- **TOTAL: 141 imágenes en dataset**

#### 📁 Artefactos generados:
- `models/kmeans_puro2.npz` - Centroides en z-score + mean/std (runtime)
- `models/Mi_kmeans_mbkmeans2.joblib` - Objeto KMeans completo
- `models/Mi_scaler_kmeans2.joblib` - Metadata de escalado interno
- `Mi_features_clusterizados2.csv` - Features con asignación de clusters
- `semillas_usadas.csv` - Centroides iniciales (medoides)

#### 🔬 Validación:
- [x] Silhouette score implementado (sin sklearn)
- [x] Visualizaciones 2D (3 vistas) y 3D generadas
- [x] Crosstab cluster×clase para verificar asignación
- [x] Exportación compatible con predictor en runtime

#### 💡 Decisiones de diseño clave:
```
✓ Transformación logarítmica doble en Hu moments:
  hu_csv (de momentos.py, eps=1e-30)
  → hu_log = -sign(hu) * log10(|hu| + 1e-12) para K-Means
  
✓ ar2: Aspect ratio de la imagen completa (no del bounding box)
  Para capturar proporciones globales tras alineación

✓ Forzado K=4 cuando hay columna 'clase' en CSV
  Asegura un cluster por cada tipo de pieza
```

---

## ✅ MÓDULO 2: PREPROCESAMIENTO DE IMÁGENES (100% COMPLETADO)

### Estado: ✅ PIPELINE ROBUSTO IMPLEMENTADO

**Archivo principal:** `procesado_img.py` (refactorizado como módulo)

#### ✅ Pipeline completo de procesamiento:
1. **Resize proporcional** → 640px de ancho (mantiene aspect ratio)
2. **Eliminación de sombras** → inpaint sobre región dilatada
3. **Aplanado de fondo** → GaussianBlur grande + sustracción
4. **Umbralización adaptativa** → Block size variable según resolución
5. **Selección de componente principal** → connectedComponents
6. **Relleno de huecos** → floodFill desde bordes
7. **Detección de bordes** → Gradiente Scharr + Otsu + dilate
8. **Refinamiento Canny** → Con umbrales adaptativos
9. **Fusión de máscaras** → OR entre solid y filled
10. **Alineación** → minAreaRect para rotación óptima
11. **Recorte** → boundingRect sobre máscara alineada

#### 📊 Parámetros optimizados:
```python
RESIZE_W = 640         # Ancho estándar
TARGET_BG = 215        # Nivel de fondo objetivo
R_IGNOREREL = 0.03     # Radio relativo para ignore mask
SIGMA_REL = 0.15       # Sigma para blur de aplanado
```

#### 🔄 Funciones modulares:
- `procesar_a_mascara()` - Pipeline hasta máscara binaria fusionada
- `alinear_recortar()` - Rotación por minAreaRect + crop
- `procesar_imagen_completa()` - Conveniencia para inference (devuelve máscara final)
- `procesar_par()` - Procesamiento por lotes de carpetas

#### ✅ Salidas generadas:
```
base_datos/ARANDELAS/  - Máscaras procesadas de arandelas
base_datos/TUERCAS/    - Máscaras procesadas de tuercas
base_datos/TORNILLOS/  - Máscaras procesadas de tornillos
base_datos/CLAVOS/     - Máscaras procesadas de clavos
```

---

## ✅ MÓDULO 3: EXTRACCIÓN DE CARACTERÍSTICAS (100% COMPLETADO)

### Estado: ✅ FEATURES ROBUSTAS EXTRAÍDAS

**Archivo principal:** `momentos.py`

#### ✅ Características implementadas:
1. **Momentos de Hu** (6 invariantes) con eps=1e-30:
   - hu1, hu2, hu3, hu4, hu5, hu6
   - Transformación: `-sign(hu) * log10(|hu| + eps)`
   
2. **Características geométricas:**
   - Circularidad: `4π·Area / Perímetro²`
   - Redondez: `Area / (π·r²)` donde r = radio círculo mínimo
   - Aspect ratio (bounding box): `min(w,h) / max(w,h)`
   - **ar2** (imagen completa): `min(H,W) / max(H,W)`
   - **n_lados**: Conteo robusto de lados del polígono convexo

3. **Características de textura** (desde imagen original):
   - grad_mean: Energía promedio del gradiente dentro de la máscara
   - edge_density: Densidad de bordes Canny (actualmente en 0.0)

#### 📁 CSV generado:
```
base_datos/cualidades_imagenes.csv
  Columnas: file, clase, hu1, hu2, hu3, hu4, hu5, hu6,
            circularidad, redondez, aspect_ratio, ar2, 
            n_lados, grad_mean, edge_density
  Filas: 141 (todas las imágenes del dataset)
```

#### 🔬 Función de conteo de lados innovadora:
```python
def contar_lados_contorno(c, eps_rel=0.02):
    # Devuelve 0 para círculos (circularidad > 0.92)
    # Snap a 6 si detecta 5-7 lados (tuercas hexagonales)
    # Fusiona vértices colineales (ángulo > 168°)
```

---

## ✅ MÓDULO 4: ALGORITMO K-NN PROPIO (100% COMPLETADO)

### Estado: ✅ IMPLEMENTACIÓN DESDE CERO COMPLETADA

**Archivo principal:** `knn_audio.py`

#### ✅ Características implementadas (sin sklearn):
- [x] **Clase KNNPuro** completa desde cero
- [x] Distancia de Minkowski generalizada (p=1 Manhattan, p=2 Euclídea)
- [x] Votación ponderada por `1/(distancia + eps)`
- [x] Manejo de empates con desempate por vecino más cercano
- [x] Manejo de distancias cero (coincidencia perfecta)
- [x] Estrategia de votación para múltiples candidatos
- [x] Estandarización manual (media/varianza) sin sklearn

#### 📊 Especificaciones técnicas:
```python
K vecinos = 7
Distancia: Minkowski con p=2 (Euclídea)
Votación: Ponderada (weighted=True)
Features: 140 dimensiones (14 por segmento × 10 segmentos)
Split: 80/20 estratificado cuando posible
```

#### 🎤 Dataset de audio:
- **Contar:** 24 audios
- **Proporción:** 24 audios
- **Salir:** 24 audios
- **TOTAL: 72 audios en dataset**

#### 📁 Artefactos generados:
- `models/knn_audio_puro.npz` - X_train, y_train, mean, std, k, p, weighted
- `base_datos/features_audio.csv` - 140 features por audio
- `base_datos/X_audio.npy` - Matriz de features
- `base_datos/y_audio.npy` - Vector de etiquetas
- `base_datos/labels_audio.json` - Mapeo clase→índice
- `errores_test.csv` - Registro de clasificaciones incorrectas

#### 🔬 Validación implementada:
- [x] Matriz de confusión manual
- [x] Classification report (precision, recall, F1) manual
- [x] Detección y guardado de errores en CSV
- [x] Entrenamiento final con TODO el dataset (sin split)
- [x] Deduplicación por path_rel antes de entrenar

#### 💡 Optimizaciones:
```python
# Distancias optimizadas por caso
if p == 2:  # Euclídea
    return np.sqrt((diff * diff).sum(axis=1))
elif p == 1:  # Manhattan
    return np.abs(diff).sum(axis=1)
else:  # Minkowski general
    return np.power(np.abs(diff), p).sum(axis=1) ** (1.0/p)
```

---

## ✅ MÓDULO 5: PREPROCESAMIENTO DE AUDIO (100% COMPLETADO)

### Estado: ✅ PIPELINE MULTI-FORMATO ROBUSTO

**Archivo principal:** `procesar_audio.py`

#### ✅ Pipeline completo de normalización:
1. **Lectura inteligente** → `smart_read_any()` con fallback múltiple:
   - Intento 1: soundfile (WAV, OGG, FLAC)
   - Intento 2: librosa/audioread (MP3, M4A, etc.)
   - Intento 3: FFmpeg CLI (MP4, WEBM, AAC, WMA, etc.)
   
2. **Conversión a mono** → Promedio de canales
3. **Re-muestreo racional** → 16 kHz con resample_poly
4. **Filtro pasa-banda** → 100 Hz - 5000 Hz (Butterworth orden 4)
5. **Filtro Notch 50 Hz** → Elimina ruido eléctrico (Q=30)
6. **Recorte de silencios** → RMS deslizante con umbral relativo 10%
7. **Normalización RMS** → Target -20 dBFS con protección de clipping
8. **Duración fija** → 1.2 segundos (padding/crop central)
9. **Exportación PCM16** → WAV estándar

#### 📊 Parámetros optimizados:
```python
TARGET_SR = 16000      # Frecuencia de muestreo
HP_HZ = 100.0          # High-pass
LP_HZ = 5000.0         # Low-pass
USE_NOTCH_50HZ = True  # Filtro de red eléctrica
NOTCH_Q = 30.0         # Factor de calidad del notch
TARGET_DUR = 1.20      # Duración objetivo (segundos)
RMS_TARGET_DBFS = -20.0
TRIM_REL_THR = 0.10    # Umbral relativo de silencio
MIN_KEEP_MS = 250      # Mínimo a conservar tras trim
```

#### 🎵 Formatos soportados:
```
.wav, .ogg, .mp3, .m4a, .mp4, .webm, .flac, .aac, .wma
(y sus variantes en mayúsculas)
```

#### ✅ Salidas generadas:
```
base_datos/Audio_norm/Contar/     - 24 WAV normalizados
base_datos/Audio_norm/Proporcion/ - 24 WAV normalizados
base_datos/Audio_norm/Salir/      - 24 WAV normalizados
```

---

## ✅ MÓDULO 6: EXTRACCIÓN DE FEATURES DE AUDIO (100% COMPLETADO)

### Estado: ✅ FEATURES POR SEGMENTOS IMPLEMENTADAS

**Archivo principal:** `cualidades_audio.py`

#### ✅ Estrategia de segmentación:
- **10 segmentos** de igual duración por audio
- Detección de región voiced (RMS relativo > 10%)
- Padding de 50ms antes/después de región voiced

#### 📊 Features por segmento (14 features × 10 = 140 total):
1. **ZCR** (Zero Crossing Rate) - mean
2. **RMS** (Root Mean Square) - mean
3. **MFCC 1** - mean, max, std
4. **MFCC 2** - mean, max, std
5. **MFCC 4** - mean, max, std
6. **MFCC 5** - mean, max, std

#### 🔧 Parámetros técnicos:
```python
TARGET_SR = 16000
N_SEG = 10
N_MFCC = 13
MFCC_INDEXES = [0, 1, 3, 4]  # MFCC 1,2,4,5 en librosa
n_fft = 512
hop_length = 160
win_length = 400
```

#### ✅ Robustez implementada:
- Manejo de segmentos cortos o silenciosos → features = [0.0]*14
- Umbral de silencio: RMS < 1e-4
- Estabilización de MFCC via `power_to_db(ref=1.0)`
- Limpieza de NaN/Inf con `np.nan_to_num()`

#### 📁 CSV generado:
```
base_datos/features_audio.csv
  Columnas: id, clase, path_rel, s01_zcr_mean, s01_rms_mean,
            s01_mfcc1_mean, s01_mfcc1_max, s01_mfcc1_std, ...
            (hasta s10_mfcc5_std)
  Filas: 72 (todos los audios del dataset)
```

---

## ✅ MÓDULO 7: CLASIFICADOR BAYESIANO (100% COMPLETADO)

### Estado: ✅ ESTIMACIÓN PROBABILÍSTICA IMPLEMENTADA

**Archivo principal:** `estimador_bayesiano.py`

#### ✅ Funcionalidades implementadas:
- [x] Definición de 4 hipótesis de cajas con proporciones distintas
- [x] Estimación batch por máxima verosimilitud (log-probabilities)
- [x] Cálculo evolutivo pieza por pieza (actualización bayesiana)
- [x] Normalización con truco log-sum-exp
- [x] Generación de tabla de evolución en consola
- [x] **Visualización gráfica 3D en tiempo real** (matplotlib)

#### 📊 Hipótesis de cajas definidas:
```python
Caja A: 250 Tornillos, 250 Clavos, 250 Arandelas, 250 Tuercas (balanceada)
Caja B: 150 Tornillos, 300 Clavos, 300 Arandelas, 250 Tuercas
Caja C: 250 Tornillos, 350 Clavos, 250 Arandelas, 150 Tuercas
Caja D: 500 Tornillos, 500 Clavos, 0 Arandelas, 0 Tuercas (solo largos)
```

#### 🎯 Integración con app Flask:
- Comando **"Proporción"** activa:
  1. Estimación final de caja más probable
  2. Cálculo de evolución de probabilidades
  3. Impresión de tabla en consola del servidor
  4. **Lanzamiento de gráfico 3D en hilo separado** (no bloquea la app)

#### 📈 Visualización generada:
```
Gráfico: Evolución de P(Hipótesis | Evidencia)
Eje X: Cantidad de piezas observadas (0 a 10)
Eje Y: Probabilidad a posteriori (0 a 1)
Series: 4 líneas (una por cada hipótesis de caja)
Marcadores: Puntos en cada observación
```

#### 🔬 Algoritmo bayesiano:
```python
P(H|E) ∝ P(E|H) · P(H)

Donde:
- P(H) = Prior uniforme (1/4 para cada caja)

- P(E|H) = Verosimilitud (proporción de pieza en caja)
- Actualización secuencial en espacio log para estabilidad numérica
```

---

## ✅ MÓDULO 8: PREDICTORES EN RUNTIME (100% COMPLETADO)

### Estado: ✅ SCRIPTS DE INFERENCIA LISTOS

#### Archivo: `predecir_imagenes.py`

**Funcionalidades:**
- [x] Carga de modelo K-Means desde `kmeans_puro2.npz`
- [x] Pipeline completo de procesamiento (idéntico al entrenamiento)
- [x] Extracción de features con transformación log doble
- [x] Predicción por distancia L2 en espacio z-score
- [x] Modo de compatibilidad para detectar doble-log inconsistente
- [x] Reportes detallados con z-scores y distancias

**Features runtime:**
```python
procesar_imagen_completa(img_path) → mask (alineada+recortada)
extraer_features(mask) → [hu1_log, hu2_log, ar2]
predecir(features, model) → (clase, distancia, z_scores, mode)
```

#### Archivo: `predecir_audio.py`

**Funcionalidades:**
- [x] Carga de modelo K-NN desde `knn_audio_puro.npz`
- [x] Pipeline completo: `smart_read_any` + normalización + features
- [x] Extracción de 140 features (14×10 segmentos)
- [x] Predicción K-NN con votación ponderada manual
- [x] Normalización con mean/std del entrenamiento

**Features runtime:**
```python
pipeline_completo_audio(audio_path) → features[140]
predecir_knn(features, modelo) → (clase, dist_promedio)
```

#### 🔧 Compatibilidad asegurada:
- Idéntico preprocesamiento entre entrenamiento y predicción
- Mismos parámetros de normalización/escalado
- Mismos cálculos de distancia y votación
- Manejo de formatos diversos (MP4, WEBM, WAV, etc.)

---

## ✅ MÓDULO 9: APLICACIÓN WEB INTEGRADA (95% COMPLETADO)

### Estado: ✅ SISTEMA FLASK FUNCIONAL CON NGROK

**Archivo principal:** `app.py`

#### ✅ Endpoints implementados:

**1. GET /** - Página principal
- Renderiza `templates/index.html`
- Interfaz responsive para móvil
- Diseño dark mode moderno

**2. POST /predict_image** - Clasificación de imágenes
```python
Entrada: FormData con archivo de imagen
Proceso:
  1. Guarda archivo temporal en uploads/
  2. Copia permanente en base_datos/subida/
  3. Procesa con predecir_imagenes.procesar_imagen_completa()
  4. Extrae features y predice clase
  5. Agrega a deque ultimas_piezas (maxlen=10)
  6. Logging en consola: "📸 Imagen clasificada: [nombre] → [clase]"
  7. Limpia archivo temporal
Salida: JSON {prediccion, historial}
```

**3. POST /predict_command** - Clasificación de audio
```python
Entrada: FormData con archivo de audio
Proceso:
  1. Guarda archivo temporal en uploads/
  2. Procesa con predecir_audio.pipeline_completo_audio()
  3. Extrae 140 features y predice comando
  4. Logging en consola: "🎤 Comando reconocido: [nombre] → [comando]"
  5. Limpia archivo temporal
  6. Ejecuta acción según comando:
     - "Contar" → Devuelve Counter de ultimas_piezas
     - "Proporcion" → Llama estimador bayesiano + gráfico
     - "Salir" → Confirmación de dos pasos
Salida: JSON {comando_detectado, resultado_accion}
```

#### 🔒 Seguridad implementada:
- [x] Confirmación de salida en dos pasos (evita apagados accidentales)
- [x] Variable global `confirmacion_salir_pendiente`
- [x] Reset de confirmación al recibir otros comandos
- [x] Limpieza de archivos temporales tras procesamiento
- [x] secure_filename() para nombres de archivo

#### 🎨 Frontend (templates/index.html):

**Características UI:**
- Diseño responsive mobile-first
- Dark mode con paleta de colores moderna
- Botones grandes táctiles
- Animación de "pulso" durante grabación
- Historial visual con chips
- Log de estado en tiempo real

**Funcionalidades JavaScript:**
- Captura de foto con input file + capture="environment"
- Grabación de audio con MediaRecorder API
- Fetch async para comunicación con backend
- Actualización dinámica del historial
- Manejo de errores con feedback visual

#### 📡 Configuración de red:
```python
host='0.0.0.0'  # Permite conexiones externas
port=5000
debug=True

# Acceso via ngrok:
# ngrok http 5000
# URL: https://[random].ngrok-free.app
```

#### 🧠 Variables globales de estado:
```python
MODELO_IMAGEN: dict    # K-Means cargado al inicio
MODELO_AUDIO: dict     # K-NN cargado al inicio
ultimas_piezas: deque  # maxlen=10, FIFO
confirmacion_salir_pendiente: bool
DIR_UPLOADS_ORIGINAL: Path  # base_datos/subida/
```

---

## 📊 ANÁLISIS DEL DATASET ACTUAL

### Dataset de Imágenes:
```
Arandela:  24 imágenes originales
Tuerca:    39 imágenes originales
Tornillo:  49 imágenes originales
Clavo:     29 imágenes originales
───────────────────────────────────
TOTAL:    141 imágenes

Distribución:
- Tornillo: 34.8% (clase mayoritaria)
- Tuerca:   27.7%
- Clavo:    20.6%
- Arandela: 17.0% (clase minoritaria)

Estado: ✅ SUFICIENTE para K-Means con 4 clusters
Recomendación: Aceptable, balanceo razonable
```

### Dataset de Audios:
```
Contar:     24 audios (WAV normalizados)
Proporcion: 24 audios (WAV normalizados)
Salir:      24 audios (WAV normalizados)
───────────────────────────────────
TOTAL:      72 audios

Distribución: 100% balanceada (33.3% por clase)

Estado: ✅ EXCELENTE balance
Recomendación: Óptimo para clasificación
```

### Modelos entrenados:
```
models/
├── kmeans_puro2.npz          (528 bytes) - Centroides + metadata
├── knn_audio_puro.npz        (95 KB) - X_train + y_train + params
├── Mi_kmeans_mbkmeans2.joblib  - Objeto completo K-Means
├── Mi_scaler_kmeans2.joblib    - Metadata escalado interno
└── kmeans_puro.npz           (528 bytes) - Versión anterior (legacy)
```

---

## 🎯 DECISIONES TÉCNICAS CLAVE

### 1. Transformación logarítmica de Momentos de Hu:
```
Entrenamiento (Mi_Kmeans.py):
  hu_csv → hu_log = -sign(hu_csv) * log10(|hu_csv| + 1e-12)
  
Predicción (predecir_imagenes.py):
  Replica mismo transform con eps=1e-12
  
Razón: Estabiliza rangos extremos de momentos invariantes
```

### 2. Aspect Ratio ar2 vs ar:
```
ar:  Aspect ratio del bounding box del contorno
ar2: Aspect ratio de la imagen completa post-alineación

Usado en K-Means: ar2 (captura proporciones globales)
Ventaja: Más estable tras alineación por minAreaRect
```

### 3. Inicialización de K-Means:
```
Método: Medoides por clase (forzado)
Alternativas descartadas:
  - k-means++ aleatorio
  - Random
  - Centroides manuales

Razón: Mayor estabilidad con datos etiquetados
```

### 4. K-NN con K=7:
```
K elegido: 7 vecinos
P: 2 (distancia Euclídea)
Votación: Ponderada 1/(d + eps)

Razón empírica: Balance entre suavidad y sensibilidad
Dataset pequeño → K impar para romper empates
```

### 5. Segmentación de audio en 10 partes:
```
N_SEG = 10
Duración fija: 1.2 segundos
Features por segmento: 14
Total features: 140

Razón: Captura dinámica temporal sin RNN/LSTM
```

### 6. Formato de audio en app:
```
Browser capture: audio/webm (Chrome/Android)
Backend: smart_read_any() con triple fallback

Razón: Máxima compatibilidad cross-platform
```

---

## 🔬 MÉTRICAS Y VALIDACIÓN

### K-Means (Imágenes):
```
✅ Silhouette score: Implementado (función manual)
✅ Visualizaciones: 2D (3 vistas) + 3D interactivo
✅ Crosstab: Verificación cluster→clase
✅ Inercia: Calculada en cada iteración

Pendiente:
🟡 Davies-Bouldin index
🟡 Matriz de confusión formal (hay crosstab)
🟡 Cross-validation K-fold
```

### K-NN (Audio):
```
✅ Split 80/20 estratificado (cuando posible)
✅ Matriz de confusión manual implementada
✅ Classification report (precision, recall, F1)
✅ Detección de errores guardada en CSV
✅ Entrenamiento final con 100% del dataset

Último reporte (evaluación con split):
- 1 error detectado en test set
- Guardado en errores_test.csv

Pendiente:
🟡 Cross-validation K-fold formal
🟡 Curvas de aprendizaje
```

### Clasificador Bayesiano:
```
✅ Verosimilitud por conteo
✅ Actualización secuencial pieza a pieza
✅ Visualización de convergencia
✅ Tabla de evolución de probabilidades

Pendiente:
🟡 Comparación con frecuentista
🟡 Intervalos de credibilidad
```

---

## 🚀 TECNOLOGÍAS Y LIBRERÍAS UTILIZADAS

### Backend (Python):
```python
# Core ML/Procesamiento
numpy >= 1.24.0        # Operaciones matriciales
pandas >= 1.5.0        # Manejo de datos tabulares
opencv-python >= 4.7.0 # Procesamiento de imágenes
scipy >= 1.10.0        # Señales y optimización (Hungarian)

# Audio
librosa >= 0.10.0      # Análisis de audio
soundfile >= 0.12.0    # I/O de audio
resampy                # Re-muestreo de alta calidad

# Web
Flask >= 2.3.0         # Framework web
werkzeug >= 2.3.0      # Utilidades WSGI

# Visualización
matplotlib >= 3.7.0    # Gráficos científicos

# Persistencia
joblib >= 1.2.0        # Serialización de modelos

# Utilidades
pathlib (stdlib)       # Manejo de rutas
collections (stdlib)   # Counter, deque
threading (stdlib)     # Ejecución paralela de gráficos
```

### Frontend:
```javascript
// APIs nativas del navegador
MediaDevices.getUserMedia()  # Acceso a micrófono
MediaRecorder API            # Grabación de audio
File API                     # Captura de fotos
Fetch API                    # Comunicación asíncrona

// No requiere librerías externas (Vanilla JS)
```

### Infraestructura:
```bash
ngrok >= 3.0            # Túnel HTTPS para desarrollo
Windows PowerShell 5.1  # Shell del sistema
Python >= 3.10          # Intérprete

#### ✅ Tareas completadas:
- [x] Script de extracción de características (`momentos.py`)
- [x] Cálculo de momentos de Hu (6 invariantes)
- [x] Características geométricas:
  - [x] Circularidad
  - [x] Redondez
  - [x] Aspect ratio
- [x] Características de textura:
  - [x] Energía de gradiente
  - [x] Densidad de bordes
- [x] Generación de CSV con features (ver `base_datos/cualidades_imagenes.csv`)
- [x] Exportación posterior con clustering (`features_clusterizados.csv`)
- [x] Transformación logarítmica de momentos de Hu
- [x] Búsqueda automática de imágenes originales

#### 🟡 Tareas pendientes:
- [ ] Analizar correlación entre características
- [ ] Eliminar características redundantes si las hay
- [ ] Validar que los valores estén en rangos esperados
- [ ] Documentar significado de cada característica

#### 📊 Características extraídas actuales:
```
1. hu1, hu2, hu3, hu4, hu5, hu6 (Momentos de Hu)
2. circularidad
3. redondez
4. aspect_ratio
5. grad_mean (textura)
6. edge_density (textura)
```

---

## ✅ MÓDULO 3: CLUSTERING K-MEANS

### Estado: ✅ MAYORMENTE COMPLETADO (85%)

#### ✅ Tareas completadas:
- [x] Implementación de K-Means (`Kmeans.py`)
- [x] Uso de 4 centroides fijos (uno por clase)
- [x] Inicialización con medoides por clase
- [x] Normalización con StandardScaler
- [x] Asignación automática cluster → clase
- [x] Visualización 2D (3 vistas)
- [x] Visualización 3D (hu1, hu2, redondez)
- [x] Script de análisis de rangos (`Rangos.py`)
- [x] Exportación de resultados (`features_clusterizados.csv`)
- [x] Artefactos guardados: `scaler_kmeans.joblib`, `kmeans_mbkmeans.joblib`
- [x] Implementaciones/variantes en repo: `Kmeans.py`, `kmeans2.py`
- [ ] Gráficos adicionales (script dedicado 3D por crear si hace falta)

#### ✅ Avances nuevos (02/11):
- [x] K-Means propio sin sklearn (minibatch) en `Mi_Kmeans.py` con escalado interno min-max
- [x] Forzado K=4 cuando hay columna `clase` (paridad con script base)
- [x] Relleno de centroides faltantes vía k-means++ hasta completar K=4
- [x] Guardado de “scaler” compatible (objeto con `transform/inverse_transform`)
- [x] CSV de salida con encabezado estable: `file, clase, hu1_log, hu2_log, ar2, cluster`
- [x] Asignación 1-a-1 cluster→clase (Hungarian si hay SciPy; si no, greedy)

- #### 🟡 Tareas pendientes:
- [ ] Validar clustering con métricas:
  - [ ] Silhouette score
  - [ ] Davies-Bouldin index
  - [ ] Matriz de confusión
- [ ] Comparar K-Means custom vs sklearn
- [ ] Documentar por qué se eligieron hu1, hu2 y redondez

#### ⚠️ CRÍTICO:
```
SEGÚN TUS NOTAS: "Kmeans y Knn desarrollados por nosotros"
                  "En caso de usar librerías tenemos que saber bien que hace"

📌 ACCIÓN REQUERIDA:
   Debes implementar K-Means desde cero o demostrar
   conocimiento profundo del algoritmo de sklearn.
```

---

## � MÓDULO 4: CLASIFICADOR K-NN (RECONOCIMIENTO DE VOZ)

### Estado: � EN PROGRESO (≈55%)

#### ✅ Avances completados:
- [x] Pipeline de normalización de audio (`procesar_audio.py`):
  - Re-muestreo a 16 kHz, paso a mono, filtro pasa banda, notch 50 Hz opcional
  - Recorte de silencios (RMS deslizante), normalización a nivel RMS objetivo
  - Duración fija (padding/recorte central) y exportación a `base_datos/Audio_norm/`
- [x] Extracción de características por segmentos (`cualidades_audio.py`):
  - 10 segmentos por audio; por segmento: ZCR, RMS y MFCC 1,2,4,5 (mean, max, std) → 14×10=140 features
  - Artefactos generados: `base_datos/features_audio.csv`, `base_datos/X_audio.npy`, `base_datos/y_audio.npy`, `base_datos/labels_audio.json`
- [x] Clasificación base con K-NN (sklearn) (`knn_audio.py`):
  - Pipeline StandardScaler + KNN(k=5, weights="distance") con split 80/20 estratificado cuando es posible
  - Reporte en consola y exportación de errores a `errores_test.csv` (último run: 1 error listado)

#### 🟡 Próximos pasos (voz):
- [ ] Ampliar dataset de audio (≥ 30-40 muestras por clase, varias voces)
- [ ] Validación adecuada: k-fold, matriz de confusión estable y métricas macro por clase
- [ ] Selección de features/regularización: ajustar N_SEG, umbrales de silencios y MFCC
- [ ] Implementar versión PROPIA de K-NN (`knn_custom.py`) sin sklearn
  - [ ] Distancia euclidiana y normalización idéntica a la del pipeline base
  - [ ] Votación por vecinos con pesos 1/d² (equivalente a weights="distance")
  - [ ] Comparativa contra sklearn (misma partición)

#### 📌 Observaciones del último experimento:
- Existe `errores_test.csv` con misclasificaciones detectadas (ej.: Cont_Rob.wav → predicho "Salir").
- Conjunto actual es pequeño; resultados pueden variar entre splits.

#### 📦 Librerías de audio utilizadas/pendientes:
```python
# Ya en uso en el repo:
librosa, soundfile, numpy, scipy, scikit-learn, pandas

# Opcional si se grabará desde Python:
sounddevice
```

---

## 🔴 MÓDULO 5: CLASIFICADOR BAYESIANO

### Estado: 🔴 NO INICIADO (0%)

#### 🔴 Tareas por hacer:
- [ ] **Definir el problema bayesiano:**
  - [ ] ¿Estimar proporciones de piezas en una caja?
  - [ ] ¿Clasificar piezas con incertidumbre?
  - [ ] Clarificar según requisitos del PDF
  
- [ ] **Implementación:**
  - [ ] Crear `clasificador_bayesiano.py`
  - [ ] Calcular probabilidades a priori
  - [ ] Implementar teorema de Bayes
  - [ ] Calcular verosimilitudes
  - [ ] Calcular probabilidades a posteriori
  
- [ ] **Integración con el sistema:**
  - [ ] Decidir cuándo usar Bayesiano vs K-Means
  - [ ] Manejar casos ambiguos
  
- [ ] **Validación:**
  - [ ] Crear dataset de prueba
  - [ ] Comparar con K-Means

---

## 🔴 MÓDULO 6: INTEGRACIÓN DEL SISTEMA

### Estado: 🟡 INICIADO PARCIALMENTE (10%)

#### ✅ Tareas completadas:
- [x] Estructura básica de carpetas
- [x] Pipeline de procesamiento de imágenes → features → clustering

#### 🔴 Tareas por hacer:
- [ ] **Crear sistema integrado principal:**
  - [ ] `main.py` o `sistema_clasificacion.py`
  - [ ] Interfaz de usuario (CLI o GUI)
  - [ ] Flujo completo: captura → procesa → clasifica → comando voz
  
- [ ] **Captura en tiempo real:**
  - [ ] Integración con cámara web
  - [ ] Captura de audio en tiempo real
  
- [ ] **Lógica de decisión:**
  - [ ] Combinar resultados de K-Means + K-NN + Bayesiano
  - [ ] Manejo de errores y casos edge
  
- [ ] **Persistencia:**
  - [ ] Guardar modelos entrenados (joblib)
  - [ ] Cargar modelos al iniciar
  - [ ] Base de datos de resultados
  
- [ ] **API REST (opcional según notas):**
  - [ ] FastAPI para exponer el sistema
  - [ ] Endpoints para clasificación
  - [ ] Documentación automática con Swagger

---

## 🔴 MÓDULO 7: VALIDACIÓN Y TESTING

### Estado: 🔴 NO INICIADO (0%)

#### 🔴 Tareas por hacer:
- [ ] **Split de datos:**
  - [ ] 70-80% entrenamiento
  - [ ] 20-30% validación
  - [ ] Asegurar que no se use test para entrenar
  
- [ ] **Métricas de evaluación:**
  - [ ] Accuracy global
  - [ ] Precision, Recall, F1-score por clase
  - [ ] Matriz de confusión
  - [ ] Curvas ROC si aplica
  
- [ ] **Tests unitarios:**
  - [ ] Tests para preprocesamiento
  - [ ] Tests para extracción de features
  - [ ] Tests para clasificadores
  
- [ ] **Validación cruzada:**
  - [ ] K-fold cross-validation
  - [ ] Leave-one-out si dataset pequeño

---

## 📚 MÓDULO 8: DOCUMENTACIÓN

### Estado: 🟡 INICIADO (20%)

#### ✅ Tareas completadas:
- [x] Notas de clase (`Notas_tomadas.txt`)
- [x] Código comentado parcialmente

#### 🔴 Tareas por hacer:
- [ ] **README.md principal:**
  - [ ] Descripción del proyecto
  - [ ] Instalación y dependencias
  - [ ] Guía de uso
  - [ ] Ejemplos
  
- [ ] **Documentación técnica:**
  - [ ] Explicación de algoritmos implementados
  - [ ] Justificación de parámetros elegidos
  - [ ] Arquitectura del sistema (diagramas)
  
- [ ] **Informe del trabajo práctico:**
  - [ ] Introducción y objetivos
  - [ ] Marco teórico
  - [ ] Metodología
  - [ ] Resultados experimentales
  - [ ] Conclusiones
  - [ ] Bibliografía
  
- [ ] **Presentación:**
  - [ ] Slides (PowerPoint/Google Slides)
  - [ ] Demo en video
  - [ ] Preparar explicación oral

---

## 📦 MÓDULO 9: DEPENDENCIAS Y ENTORNO

### Estado: ✅ PARCIALMENTE COMPLETADO (60%)

#### ✅ Librerías ya utilizadas:
```python
- opencv-python (cv2)
- numpy
- pandas
- matplotlib
- scikit-learn (StandardScaler, KMeans)
- pathlib
- joblib
- librosa
- soundfile
- scipy
```

#### 🔴 Librerías necesarias por instalar:
```python
- sounddevice      # Para captura de audio (opcional)
- fastapi          # Si hacen API REST (opcional)
- uvicorn          # Para correr FastAPI
- joblib           # Para guardar modelos (puede que ya esté)
```

#### 🔴 Tareas por hacer:
- [ ] Crear `requirements.txt`
- [ ] Crear `environment.yml` (si usan conda)
- [ ] Documentar versiones específicas
- [ ] Instrucciones de instalación paso a paso

---

## 🎯 PRIORIDADES INMEDIATAS (próximos 7 días)

### 🔥 CRÍTICO - HACER YA:

1. **[ ] AMPLIAR DATASET DE IMÁGENES**
   - Capturar mínimo 10 imágenes más por clase
   - Variar ángulos, iluminación, posición
   - Objetivo: 20-25 imágenes por clase

2. **[ ] VALIDAR K-MEANS EQUIVALENTE**
  - Reportar métricas (silhouette, Davies-Bouldin, matriz de confusión)
  - Comparar resultados entre `Kmeans.py` (sklearn) y `Mi_Kmeans.py` (propio)
  - Dejar justificación y conclusiones

3. **[ ] IMPLEMENTAR K-NN DESDE CERO**
   - Crear `knn_custom.py`
   - Implementar distancia euclidiana manual
   - Implementar votación de vecinos

### 🟡 IMPORTANTE - Siguiente semana:

4. **[ ] AMPLIAR Y CURAR DATASET DE AUDIOS**
  - Definir lista final de comandos
  - Grabar ≥ 30-40 muestras por comando y por 2-3 personas
  - Reprocesar con `procesar_audio.py` y re-extraer con `cualidades_audio.py`

5. **[ ] IMPLEMENTAR CLASIFICADOR BAYESIANO**
   - Definir claramente el problema a resolver
   - Implementar teorema de Bayes

6. **[ ] CREAR SISTEMA INTEGRADO**
   - Script principal que una todo
   - Interfaz de usuario básica (CLI)

### 🟢 DESEABLE - Última semana:

7. **[ ] VALIDACIÓN EXHAUSTIVA**
   - Métricas de evaluación
   - Matrices de confusión
   - Comparación de modelos

8. **[ ] DOCUMENTACIÓN COMPLETA**
   - README detallado
   - Informe técnico
   - Preparar presentación

---

## 📊 CHECKLIST DE ENTREGABLES FINALES

### 📄 Código:
- [x] `Mi_Kmeans.py` - Implementación propia de K-Means (minibatch + escalado interno)
- [ ] `knn_custom.py` - Implementación propia de K-NN
- [ ] `clasificador_bayesiano.py` - Clasificador Bayesiano
- [ ] `sistema_integrado.py` - Sistema completo
- [ ] `procesado_img2.py` - ✅ Ya existe
- [ ] `momentos.py` - ✅ Ya existe
- [ ] `requirements.txt` - Dependencias
- [ ] `tests/` - Suite de tests

### 📊 Datos:
- [ ] Dataset de imágenes (mínimo 80-100)
- [ ] Dataset de audios (mínimo 120-150 muestras)
- [ ] CSV de características - ✅ Ya existe
- [ ] Modelos entrenados guardados (`.joblib` o `.pkl`)

### 📚 Documentación:
- [ ] `README.md` - Guía principal
- [ ] `INFORME_FINAL.pdf` - Informe académico
- [ ] `MANUAL_USO.md` - Instrucciones de uso
- [ ] Diagramas de arquitectura
- [ ] Presentación (slides)

### 🎥 Demo:
- [ ] Video demostrativo (5-10 minutos)
- [ ] Capturas de pantalla de funcionamiento
- [ ] Ejemplos de casos de uso

---

## 📝 NOTAS IMPORTANTES DE TUS APUNTES

### Conceptos clave a aplicar:
```
✅ Normalización implementada (StandardScaler)
✅ Centroides iniciales con medoides (implementado)
⚠️ PCA mencionado pero NO implementado (no es necesario según tu decisión)
✅ Distancia euclidiana en espacio normalizado
⚠️ Validación de media y desviación estándar por característica (pendiente)
🔴 K-NN para audio (NO implementado)
🔴 Clasificador Bayesiano (NO implementado)
```

### Decisiones de diseño tomadas:
- ✅ Usar 3 características: hu1_log, hu2_log, redondez
- ✅ 4 clusters (uno por clase)
- ✅ Inicialización con medoides por clase
- ✅ Preprocesamiento riguroso de imágenes

---

## 🚀 ROADMAP SUGERIDO

### Semana 1 (actual):
- Ampliar dataset de imágenes
- Implementar K-Means desde cero
- Implementar K-NN desde cero

### Semana 2:
- Capturar y procesar audios
- Integrar K-NN con reconocimiento de voz
- Implementar clasificador Bayesiano

### Semana 3:
- Sistema integrado completo
- Validación exhaustiva
- Corrección de bugs

### Semana 4 (entrega):
- Documentación final
- Informe técnico
- Presentación
- Video demo

---

## ⚠️ RIESGOS Y ALERTAS

### 🔴 Riesgos ALTOS:
1. **Dataset muy pequeño** (40 imágenes) → Overfitting probable
2. **Módulo de voz 0%** → Mayor esfuerzo requerido
3. **No hay implementación propia de K-Means/K-NN** → Requisito del TP

### 🟡 Riesgos MEDIOS:
1. Falta de validación cruzada
2. No hay sistema integrado
3. Documentación incompleta

### 🟢 Fortalezas:
1. ✅ Preprocesamiento de imágenes robusto
2. ✅ Extracción de características bien pensada
3. ✅ Código organizado y modular
4. ✅ Visualizaciones útiles

---

## 📞 CONSULTAS PENDIENTES CON EL DOCENTE

- [ ] Confirmar si K-Means/K-NN deben ser 100% propios o si sklearn está permitido
- [ ] Aclarar requisitos específicos del clasificador Bayesiano
- [ ] Confirmar formato de entrega (código + informe + presentación)
- [ ] Consultar sobre dataset mínimo aceptable

---

## 💾 BACKUP Y CONTROL DE VERSIONES

- [ ] Configurar `.gitignore` apropiado
- [ ] Hacer commits frecuentes con mensajes descriptivos
- [ ] Crear branches para features grandes
- [ ] Tag de versión final antes de entregar
- [ ] Backup en la nube (Google Drive, OneDrive, etc.)

---

**Última actualización:** 2 de noviembre de 2025  
**Próxima revisión:** Diaria hasta la entrega

---

## 🎓 CONCLUSIÓN DEL ANÁLISIS

### Estado actual: **55% completado**

**Puntos fuertes:**
- Excelente trabajo en preprocesamiento de imágenes
- Buena extracción de características
- Visualizaciones útiles y claras

**Áreas críticas que requieren atención inmediata:**
1. ⚠️ Implementar algoritmos desde cero (K-Means, K-NN)
2. ⚠️ Desarrollar completamente el módulo de reconocimiento de voz
3. ⚠️ Ampliar significativamente el dataset
4. ⚠️ Crear sistema integrado

**Recomendación:** 
Enfocarse en los próximos 7 días en:
1. Algoritmos propios (K-Means y K-NN)
2. Ampliación del dataset
3. Inicio del módulo de audio

Con trabajo constante y enfocado, el proyecto es **COMPLETAMENTE VIABLE** ✅
