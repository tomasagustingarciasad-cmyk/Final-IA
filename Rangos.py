# grafico_rangos_todos.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("base_datos/features_imagenes.csv")
OUT_DIR  = Path("base_datos/plots_rangos")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMG  = OUT_DIR / "rangos_todos.png"

CLASSES  = ["Arandela", "Tuerca", "Tornillo", "Clavo"]
FEATURES = ["hu1","hu2","hu3","hu4","hu5","hu6","circularidad","redondez","aspect_ratio"]

if not CSV_PATH.exists():
    raise SystemExit(f"No se encontró el CSV: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Asegurar tipos numéricos
for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=FEATURES)

# Resumen min/med/max por clase y parámetro
summ = {}
for feat in FEATURES:
    rows = []
    for cls in CLASSES:
        s = df.loc[df["clase"] == cls, feat]
        if s.empty: 
            continue
        rows.append({"clase": cls, "min": s.min(), "med": s.median(), "max": s.max(), "n": int(s.count())})
    summ[feat] = pd.DataFrame(rows)

# Figura única con una columna por parámetro
n = len(FEATURES)
fig, axes = plt.subplots(1, n, figsize=(2.2*n, 4), sharey=True)

if n == 1:
    axes = [axes]

for ax, feat in zip(axes, FEATURES):
    if feat not in summ or summ[feat].empty:
        ax.set_xlabel(feat)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    s = summ[feat].copy()
    s["orden"] = s["clase"].apply(lambda x: CLASSES.index(x))
    s = s.sort_values("orden")
    y = np.arange(len(s))

    # Dibujar rangos y medianas
    for i, r in s.iterrows():
        yi = y[list(s.index).index(i)]
        ax.plot([r["min"], r["max"]], [yi, yi], "-")
        ax.plot(r["med"], yi, "o")

    ax.set_xlabel(feat)                    # ← etiqueta abajo
    ax.grid(True, axis="x", linestyle=":")
    if ax is axes[0]:
        ax.set_yticks(y)
        ax.set_yticklabels(s["clase"])
    else:
        ax.set_yticks(y)
        ax.set_yticklabels([])

# Dar más margen abajo para las etiquetas
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig(OUT_IMG, dpi=150)
plt.show()
print(f"Gráfico guardado en: {OUT_IMG}")
