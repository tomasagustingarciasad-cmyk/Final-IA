# rango_parametros.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("base_datos/features_imagenes.csv")
OUT_DIR  = Path("base_datos/plots_rangos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Orden deseado de clases
CLASSES = ["Arandela", "Tuerca", "Tornillo", "Clavo"]

# 9 parámetros solicitados
FEATURES = ["hu1","hu2","hu3","hu4","hu5","hu6",
            "circularidad","redondez","aspect_ratio"]

if not CSV_PATH.exists():
    raise SystemExit(f"No se encontró el CSV: {CSV_PATH}")

# Leer CSV
df = pd.read_csv(CSV_PATH)

# Asegurar columnas necesarias
expected = {"file","clase", *FEATURES}
missing = expected - set(df.columns)
if missing:
    raise SystemExit(f"Faltan columnas en el CSV: {missing}")

# Convertir a numérico por si vinieron como texto
for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Filtrar filas válidas (sin NaN en los features)
df = df.dropna(subset=FEATURES)

# Función para graficar rango por clase en un único parámetro
def plot_range_for_feature(feat: str):
    # Calcular min, mediana, max por clase (solo si hay datos de esa clase)
    rows = []
    for cls in CLASSES:
        s = df.loc[df["clase"] == cls, feat]
        if len(s) == 0:
            continue
        rows.append({
            "clase": cls,
            "min":   float(s.min()),
            "med":   float(s.median()),
            "max":   float(s.max()),
            "n":     int(s.count())
        })
    if not rows:
        print(f"[AVISO] No hay datos para '{feat}'.")
        return

    summary = pd.DataFrame(rows)
    # Mantener el orden de CLASSES
    summary["orden"] = summary["clase"].apply(lambda x: CLASSES.index(x))
    summary = summary.sort_values("orden")

    y = np.arange(len(summary))
    plt.figure(figsize=(8, 3.5))

    # Línea min–max y punto en la mediana
    for i, r in summary.iterrows():
        plt.plot([r["min"], r["max"]], [y[list(summary.index).index(i)] ]*2, "-")
        plt.plot(r["med"], y[list(summary.index).index(i)], "o")

    plt.yticks(y, summary["clase"])
    plt.xlabel(feat)
    plt.title(f"Rango por clase: {feat}")
    plt.grid(True, axis="x", linestyle=":")
    plt.tight_layout()

    out_path = OUT_DIR / f"rango_{feat}.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Guardado: {out_path}  (n por clase: {list(summary['n'])})")

# Generar una figura por parámetro
for feat in FEATURES:
    plot_range_for_feature(feat)

print("\nListo. Figuras en:", OUT_DIR)
