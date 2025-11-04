# 📋 SEGUIMIENTO DEL TRABAJO FINAL - INTELIGENCIA ARTIFICIAL
## Sistema de Clasificación de Piezas con Visión Artificial y Comandos de Voz

---

## 📊 RESUMEN EJECUTIVO

**Fecha de análisis:** 2 de noviembre de 2025  
**Estado general:** 🟡 En progreso avanzado (≈62% completado)

### Progreso por componentes:
- ✅ **Visión Artificial (K-Means):** 85% completado
- � **Reconocimiento de Voz (K-NN):** 55% completado  
- 🔴 **Clasificador Bayesiano:** 0% completado
- 🟡 **Integración del sistema:** 12% completado
- 🔴 **Documentación y entregables:** 25% completado

---

## 🎯 REQUISITOS DEL TRABAJO PRÁCTICO (según PDF)

### Componentes principales requeridos:
1. **Agente inteligente** que procese imágenes y comandos de voz
2. **K-Means** para clasificación de piezas (Arandela, Tuerca, Tornillo, Clavo)
3. **K-NN** para reconocimiento de comandos de voz
4. **Clasificador Bayesiano** para estimación de proporciones en caja
5. **Sistema integrado** que combine todos los módulos
6. **Documentación completa** y presentación

---

## ✅ MÓDULO 1: PREPROCESAMIENTO DE IMÁGENES

### Estado: ✅ COMPLETADO (95%)

#### ✅ Tareas completadas:
- [x] Script de procesamiento de imágenes (`procesado_img2.py`)
- [x] Normalización de tamaño (640px de ancho)
- [x] Eliminación de sombras con inpaint
- [x] Umbralización adaptativa
- [x] Detección de bordes con gradiente Scharr
- [x] Rellenado de huecos
- [x] Generación de máscaras binarias
- [x] Procesamiento de 4 clases de piezas
- [x] Imágenes originales capturadas (carpeta `base_datos/`):
  - Arandelas: 11 imágenes (ver `base_datos/Arandela/`)
  - Tuercas: pendiente de conteo (ver `base_datos/Tuerca/`)
  - Tornillos: 14 imágenes (ver `base_datos/Tornillo/`)
  - Clavos: 14 imágenes (ver `base_datos/Clavo/`)
  - Procesadas/derivadas: `base_datos/ARANDELAS/`, `base_datos/CLAVOS/`, `base_datos/TORNILLOS/`, `base_datos/TUERCAS/`
  - **TOTAL conocido:** ≥39 (falta confirmar Tuercas)

#### 🟡 Tareas pendientes:
- [ ] **CRÍTICO:** Ampliar dataset a mínimo 20-30 imágenes por clase
- [ ] Validar calidad de máscaras generadas (revisar manualmente)
- [ ] Documentar parámetros de procesamiento elegidos
- [ ] Crear carpeta con ejemplos de "antes/después" del procesamiento

#### 💡 Recomendaciones:
```
⚠️ URGENTE: El dataset es MUY PEQUEÑO (40 imágenes total)
   - Mínimo recomendado: 80-120 imágenes (20-30 por clase)
   - Capturar más fotos con diferentes:
     * Ángulos de rotación
     * Iluminaciones
     * Posiciones
     * Fondos
```

---## 2. Arquitectura del Sistema

El proyecto se estructura en un modelo Cliente-Servidor, compuesto por tres capas principales:

```
┌─────────────────────────────────────────────────────────┐
│  CLIENTE (Frontend - Celular)                           │
│  ├─ Interfaz web (HTML, CSS, JavaScript)                 │
│  ├─ Botón para capturar foto (usa la cámara del móvil)   │
│  └─ Botón para grabar audio (usa el micrófono del móvil) │
└──────────────────┬──────────────────────────────────────┘
                   │ (Peticiones HTTP a través de la red local)
                   ↓
┌─────────────────────────────────────────────────────────┐
│  SERVIDOR (Backend - Python / Flask)                    │
│  ├─ Recibe archivos (imágenes, audio).                   │
│  ├─ Orquesta el procesamiento y la predicción.           │
│  ├─ Ejecuta los modelos de K-Means y K-NN.               │
│  └─ Gestiona el estado de la aplicación (historial).    │
└──────────────────┬──────────────────────────────────────┘






┌─────────────────────────────────────────────────────────┐
│  CELULAR (Frontend - HTML/JavaScript)                   │
│  ├─ Botón: "Capturar Foto" → Saca foto y la envía      │
│  ├─ Botón: "Grabar Comando" → Graba voz y la envía     │
│  └─ Muestra: Resultados y conteo actual                 │
└──────────────────┬──────────────────────────────────────┘
                   │ (HTTP Request)
                   ↓
┌─────────────────────────────────────────────────────────┐
│  COMPUTADORA (Backend - Python Flask/FastAPI)           │
│  ├─ Recibe foto → Clasifica con K-Means → Guarda       │
│  ├─ Recibe audio → Clasifica con K-NN → Ejecuta acción │
│  └─ Mantiene: CSV con conteo de piezas (10 últimas)    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│  MODELOS ENTRENADOS (archivos .joblib / .npz)          │
│  ├─ models/kmeans_puro.npz  (K-Means entrenado)        │
│  ├─ models/knn_audio_puro.npz (K-NN entrenado)         │
│  └─ resultados_sesion.csv (10 últimas clasificaciones) │
└─────────────────────────────────────────────────────────┘

## ✅ MÓDULO 2: EXTRACCIÓN DE CARACTERÍSTICAS

### Estado: ✅ COMPLETADO (90%)

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
