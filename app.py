import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from collections import deque, Counter
import numpy as np
from threading import Timer
# --- Importamos la lógica de nuestros scripts de predicción ---
# (Asegúrate de que los archivos .py estén en la misma carpeta)
try:
    # Lógica de predicción de imágenes
    from predecir_imagenes import (
        cargar_modelo as cargar_modelo_img,
        procesar_imagen_completa,
        extraer_features as extraer_features_img,
        predecir as predecir_img
    )
    # Lógica de predicción de audio
    from predecir_audio import (
        cargar_modelo as cargar_modelo_audio,
        pipeline_completo_audio,
        predecir_knn
    )
except ImportError as e:
    print(f"❌ Error al importar los módulos de predicción: {e}")
    print("Asegúrate de que 'predecir_imagenes.py' y 'predecir_audio.py' estén en la misma carpeta que 'app.py'")
    exit()

# ===============================
# Configuración de la App Flask
# ===============================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# --- Cargar modelos al iniciar la app ---
print("🧠 Cargando modelos al iniciar...")
try:
    # ✅ CORREGIDO: Especificamos las rutas a los modelos
    ruta_modelo_imagen = Path("models/kmeans_puro.npz")
    ruta_modelo_audio = Path("models/knn_audio_puro.npz")

    # Ahora pasamos las rutas a las funciones de carga
    MODELO_IMAGEN = cargar_modelo_img(ruta_modelo_imagen)
    MODELO_AUDIO = cargar_modelo_audio(ruta_modelo_audio)
    
    print("✅ Modelos cargados correctamente.")
except Exception as e:
    print(f"❌ ERROR FATAL: No se pudieron cargar los modelos: {e}")
    exit()

# --- Memoria de la aplicación (últimas 10 piezas) ---
ultimas_piezas = deque(maxlen=10)
confirmacion_salir_pendiente = False

# ===============================
# Endpoints (Rutas de la API)
# ===============================

# --- 1. Servir la página principal (el HTML) ---
@app.route('/')
def index():
    """Muestra la interfaz web principal."""
    return render_template('index.html')

# --- 2. Endpoint para predecir una imagen ---
@app.route('/predict_image', methods=['POST'])
def handle_predict_image():
    global confirmacion_salir_pendiente 
    confirmacion_salir_pendiente = False 
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    try:
        # Guardar el archivo temporalmente
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)

        # Predecir usando la lógica importada
        mask = procesar_imagen_completa(filepath)
        features, _ = extraer_features_img(mask)
        clase, _, _, _ = predecir_img(features, MODELO_IMAGEN)

        # Guardar en memoria
        ultimas_piezas.append(clase)

        # Limpiar archivo temporal
        os.remove(filepath)

        # Devolver resultado
        return jsonify({
            'prediccion': clase,
            'historial': list(ultimas_piezas)
        })

    except Exception as e:
        return jsonify({'error': f'Error procesando la imagen: {str(e)}'}), 500

# --- 3. Endpoint para ejecutar un comando de voz ---
@app.route('/predict_command', methods=['POST'])
def handle_predict_command():
    global confirmacion_salir_pendiente
    
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo de audio'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo de audio vacío'}), 400

    try:
        # Guardar el archivo temporalmente
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)

        # Predecir el comando usando la lógica importada
        features = pipeline_completo_audio(filepath)
        comando, _ = predecir_knn(features, MODELO_AUDIO)

        # Limpiar archivo temporal
        os.remove(filepath)

        # Ejecutar la acción correspondiente al comando
        resultado_comando = ""
        
        
        if comando.lower() == 'contar':
            confirmacion_salir_pendiente = False # Reseteamos la confirmación
            conteo = Counter(ultimas_piezas)
            resultado_comando = "Conteo actual: " + ", ".join([f"{k}: {v}" for k, v in conteo.items()])
            if not resultado_comando:
                resultado_comando = "Aún no hay piezas clasificadas."

        elif comando.lower() == 'proporcion':
            confirmacion_salir_pendiente = False
            total = len(ultimas_piezas)
            if total == 0:
                resultado_comando = "No hay piezas para calcular proporción."
            else:
                conteo = Counter(ultimas_piezas)
                resultado_comando = "Proporción: " + ", ".join([f"{k}: {v/total:.0%}" for k, v in conteo.items()])
        
        # elif comando.lower() == 'salir':
        #     if confirmacion_salir_pendiente:
        #         # Si ya estábamos esperando confirmación, ahora sí apagamos.
        #         resultado_comando = "Confirmado. Servidor apagándose..."
        #         shutdown_func = request.environ.get('werkzeug.server.shutdown')
        #         if shutdown_func is None:
        #             resultado_comando = "Error: No se puede apagar el servidor (no es Werkzeug)."
        #         else:
        #             shutdown_func() # Apagar
        #     else:
        #         # Es la primera vez que dice "Salir". Pedimos confirmación.
        #         confirmacion_salir_pendiente = True
        #         resultado_comando = "¿Estás seguro? Di 'Salir' de nuevo para confirmar."
        elif comando.lower() == 'salir':
            if confirmacion_salir_pendiente:
                resultado_comando = "Confirmado. Servidor apagándose..."
                shutdown_func = request.environ.get('werkzeug.server.shutdown')
                if shutdown_func:
                    shutdown_func()
                    Timer(1.0, shutdown_func).start()
                else:
                    Timer(1.0, lambda: os._exit(0)).start()
            else:
                # Es la primera vez que dice "Salir". Pedimos confirmación.
                 confirmacion_salir_pendiente = True
                 resultado_comando = "¿Estás seguro? Di 'Salir' de nuevo para confirmar."

        else:
            confirmacion_salir_pendiente = False
            resultado_comando = f"Comando '{comando}' no reconocido."

        return jsonify({
            'comando_detectado': comando,
            'resultado_accion': resultado_comando
        })

    except Exception as e:
        return jsonify({'error': f'Error procesando el audio: {str(e)}'}), 500

# ===============================
# Iniciar la aplicación
# ===============================
if __name__ == '__main__':
    # host='0.0.0.0' permite que otros dispositivos en tu red (como tu celular) se conecten.
    app.run(host='0.0.0.0', port=5000, debug=True)