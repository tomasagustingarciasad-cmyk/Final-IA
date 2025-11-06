# estimador_bayesiano.py
import math
from collections import Counter
import matplotlib.pyplot as plt

# --- 1. Definición del Problema (Sin cambios) ---
DEFINICIONES_CAJAS = {
    "Caja A": {"Tornillo": 250, "Clavo": 250, "Arandela": 250, "Tuerca": 250},
    "Caja B": {"Tornillo": 150, "Clavo": 300, "Arandela": 300, "Tuerca": 250},
    "Caja C": {"Tornillo": 250, "Clavo": 350, "Arandela": 250, "Tuerca": 150},
    "Caja D": {"Tornillo": 500, "Clavo": 500, "Arandela": 0, "Tuerca": 0}
}
NOMBRES_PIEZAS = ["Tornillo", "Clavo", "Arandela", "Tuerca"]
PROPORCIONES_CAJAS = {}
for nombre, conteos in DEFINICIONES_CAJAS.items():
    PROPORCIONES_CAJAS[nombre] = {
        pieza: conteos.get(pieza, 0) / 1000.0 
        for pieza in NOMBRES_PIEZAS
    }

# --- 2. Función de Estimación "Batch" (Corregida) ---

def estimar_caja(conteo_muestra: Counter) -> tuple[str, dict, float]:
    """
    Estima la caja más probable dado un CONTEO de muestra (batch).
    
    Devuelve: (nombre_mejor_caja, definicion_mejor_caja, probabilidad_normalizada)
    """
    
    # Almacenamos los log_probs de todas las hipótesis
    hipotesis = list(PROPORCIONES_CAJAS.keys())
    log_probs_cajas = []
    
    # Iteramos por cada una de las 4 cajas posibles
    for nombre_caja in hipotesis:
        
        log_prob_caja = 0.0
        
        # Iteramos por cada tipo de pieza para calcular el log-likelihood
        for pieza in NOMBRES_PIEZAS:
            n = conteo_muestra.get(pieza, 0)
            
            #
            # === ESTA ES LA LÍNEA CORREGIDA ===
            # Antes decía: p = proporciones.get(pieza, 0.0) (Error)
            p = PROPORCIONES_CAJAS[nombre_caja].get(pieza, 0.0)
            # === FIN DE LA CORRECCIÓN ===
            #
            
            # Manejo crucial de log(0)
            if p == 0.0:
                if n > 0:
                    log_prob_caja = -float('inf')
                    break 
                else:
                    pass
            else:
                log_prob_caja += n * math.log(p)
                
        # Añadimos el log-prob de esta caja a nuestra lista
        log_probs_cajas.append(log_prob_caja)
            
    if not log_probs_cajas:
        return "Desconocida", {}, 0.0
        
    # Normalizamos los log_probs usando la función que ya teníamos
    norm_log_probs = _normalizar_log_probs(log_probs_cajas)
    
    # Convertimos de log-probabilidad a probabilidad (0 a 1)
    probs = [math.exp(lp) for lp in norm_log_probs]
    
    # Encontramos la probabilidad más alta y su índice
    prob_ganadora = max(probs)
    idx_ganadora = probs.index(prob_ganadora)
    
    # Obtenemos el nombre de la caja ganadora
    mejor_caja = hipotesis[idx_ganadora]
        
    return mejor_caja, DEFINICIONES_CAJAS[mejor_caja], prob_ganadora

# --- 3. y 4. Funciones de Evolución y Gráficos (Sin cambios) ---

def _normalizar_log_probs(log_probs: list) -> list:
    """Normaliza un vector de log-probabilidades usando el truco log-sum-exp."""
    l_max = max(log_probs)
    if l_max == -float('inf'):
        return [math.log(1.0 / len(log_probs))] * len(log_probs)
    sum_exp = sum(math.exp(lp - l_max) for lp in log_probs)
    log_sum = l_max + math.log(sum_exp)
    return [lp - log_sum for lp in log_probs]

def calcular_evolucion_probabilidades(historial_piezas: list) -> dict:
    """
    Calcula la evolución de las probabilidades a posteriori pieza por pieza.
    """
    hipotesis = list(PROPORCIONES_CAJAS.keys()) 
    n_hipotesis = len(hipotesis)
    
    log_prob_actual = [math.log(1.0 / n_hipotesis)] * n_hipotesis
    evolucion = []
    
    evolucion.append({
        'n': 0,
        'pieza': 'Inicial',
        'probs': [math.exp(lp) for lp in log_prob_actual]
    })
    
    for i, pieza_observada in enumerate(historial_piezas):
        log_likelihoods = []
        for nombre_caja in hipotesis:
            p = PROPORCIONES_CAJAS[nombre_caja].get(pieza_observada, 0.0)
            if p == 0.0:
                log_likelihoods.append(-float('inf')) 
            else:
                log_likelihoods.append(math.log(p))
                
        log_prob_actual = [lp_viejo + ll_nuevo for lp_viejo, ll_nuevo in zip(log_prob_actual, log_likelihoods)]
        log_prob_actual = _normalizar_log_probs(log_prob_actual)
        
        evolucion.append({
            'n': i + 1,
            'pieza': pieza_observada,
            'probs': [math.exp(lp) for lp in log_prob_actual]
        })
        
    return {'hipotesis': hipotesis, 'evolucion': evolucion}

def generar_grafico_evolucion_local(data: dict):
    # ... (Esta función no cambia) ...
    try:
        hipotesis = data['hipotesis']
        evolucion = data['evolucion']
        n_vals = [row['n'] for row in evolucion]
        
        prob_series = [[] for _ in hipotesis]
        for row in evolucion:
            for i, p in enumerate(row['probs']):
                prob_series[i].append(p)
                
        plt.figure(figsize=(10, 6))
        for i, serie in enumerate(prob_series):
            plt.plot(n_vals, serie, label=hipotesis[i], marker='.', markersize=8)
            
        plt.title('Evolución de las Probabilidades a Posteriori')
        plt.xlabel('Cantidad de Piezas Observadas (n)')
        plt.ylabel('P(Hipótesis | Evidencia)')
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(-0.05, 1.05)
        plt.xticks(range(min(n_vals), max(n_vals) + 1))
        plt.tight_layout()
        
        plt.show() 
    
    except Exception as e:
        print(f"[Error al generar gráfico local]: {e}")

def imprimir_tabla_evolucion(data: dict):
    # ... (Esta función no cambia) ...
    try:
        hipotesis = data['hipotesis']
        evolucion = data['evolucion']
        
        headers_hipotesis = " | ".join([f"P({h.split()[-1]})" for h in hipotesis])
        header = f"{'n':>3} | {'Pieza':<10} | {headers_hipotesis}"
        
        print("\n" + "="*len(header))
        print("          Evolución de Probabilidades a Posteriori          ")
        print(header)
        print("="*len(header))
        
        for row in evolucion:
            n = row['n']
            pieza = row['pieza']
            probs = row['probs']
            max_p = max(probs)
            
            row_str = f"{n:>3} | {pieza:<10} | "
            prob_strs = []
            for p in probs:
                marker = "*" if p == max_p and p > (1.0 / len(hipotesis)) else " "
                prob_strs.append(f"{p:.4f}{marker}")
            
            row_str += " | ".join(prob_strs)
            print(row_str)
            
        print("="*len(header) + "\n")
    
    except Exception as e:
        print(f"[Error al imprimir tabla]: {e}")