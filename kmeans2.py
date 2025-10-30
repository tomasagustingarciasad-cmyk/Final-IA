# Kmeans.py — clustering rápido + 2D/3D con leyenda por elemento
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.utils import check_random_state
import joblib

# ----------------------------
# Config por defecto
# ----------------------------
CSV_DEFAULT_NAME = "cualidades_imagenes.csv"
CSV_OUT = "features_clusterizados.csv"
SCALER_OUT = "scaler_kmeans.joblib"
MODEL_OUT = "kmeans_mbkmeans.joblib"
RANDOM_STATE = 42
MAX_SILH_SAMPLES = 5000

# ----------------------------
# Utils
# ----------------------------
def log_transform_hu(x, eps=1e-12):
    x = np.asarray(x)
    return -np.sign(x) * np.log10(np.abs(x) + eps)

def maybe_downsample(X, y=None, max_n=MAX_SILH_SAMPLES, random_state=RANDOM_STATE):
    n = X.shape[0]
    if n <= max_n: return X, y
    rng = check_random_state(random_state)
    idx = rng.choice(n, size=max_n, replace=False)
    return (X[idx], None if y is None else y[idx])

def resolve_csv_path(cli_csv: str | None):
    if cli_csv:
        p = Path(cli_csv).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"No existe el CSV indicado: {p}")
        return p.resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        here / "base_datos" / CSV_DEFAULT_NAME,
        here / CSV_DEFAULT_NAME,
        here.parent / "base_datos" / CSV_DEFAULT_NAME,
        here.parent / CSV_DEFAULT_NAME,
    ]
    for c in candidates:
        if c.exists(): return c.resolve()
    hits = list(here.rglob(CSV_DEFAULT_NAME))
    if hits: return hits[0].resolve()
    raise FileNotFoundError(f"No se encontró {CSV_DEFAULT_NAME} cerca de {here}")

def assign_clusters_one_to_one(df: pd.DataFrame) -> dict[int, str]:
    """
    Devuelve un dict {cluster:int -> clase:str} con asignación 1-a-1.
    Usa Hungarian si SciPy está disponible; sino greedy.
    """
    if 'clase' not in df.columns:
        return {}
    # normalizar nombres (tuerca , TUERCA -> Tuerca)
    df = df.copy()
    df['clase_norm'] = df['clase'].astype(str).str.strip().str.lower()
    df['clase_norm'] = df['clase_norm'].map(
        {'arandela':'Arandela','clavo':'Clavo','tornillo':'Tornillo','tuerca':'Tuerca'}
    ).fillna(df['clase'].astype(str).str.strip().str.capitalize())

    ct = pd.crosstab(df['cluster'], df['clase_norm'])
    if ct.empty:
        return {}

    # intentar Hungarian
    mayoritaria = {}
    try:
        from scipy.optimize import linear_sum_assignment
        cost = ct.max().max() - ct.values  # maximizar conteos
        r_idx, c_idx = linear_sum_assignment(cost)
        for r, c in zip(r_idx, c_idx):
            mayoritaria[int(ct.index[r])] = str(ct.columns[c])
    except Exception:
        # greedy si no hay scipy
        temp = ct.values.copy()
        used_r, used_c = set(), set()
        while len(used_r) < ct.shape[0] and len(used_c) < ct.shape[1]:
            r, c = np.unravel_index(np.argmax(temp), temp.shape)
            if temp[r, c] < 0: break
            if r in used_r or c in used_c:
                temp[r, c] = -1
                continue
            mayoritaria[int(ct.index[r])] = str(ct.columns[c])
            used_r.add(r); used_c.add(c)
            temp[r, :] = -1; temp[:, c] = -1

    # completar clusters no asignados con mayoría simple
    for clu in sorted(df['cluster'].unique()):
        if clu not in mayoritaria:
            mayoritaria[clu] = (
                df.loc[df['cluster'] == clu, 'clase_norm'].value_counts().idxmax()
            )
    print("\nCrosstab cluster×clase:\n", ct)
    print("\nAsignación 1-a-1 cluster→clase:", mayoritaria)
    return mayoritaria

def plot_2d_and_3d(df, labels, kmeans, scaler, mayoritaria):
    cmap = plt.get_cmap('tab10')
    # ---- 2D (3 vistas)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    def scat(ax, x, y, title, xl, yl):
        sc = ax.scatter(x, y, c=labels, cmap=cmap, alpha=0.8, s=20)
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl)
        return sc

    scat(axes[0], df['hu1_log'], df['hu2_log'], 'hu1_log vs hu2_log', 'hu1_log', 'hu2_log')
    scat(axes[1], df['hu1_log'], df['ar2'], 'hu1_log vs ar2', 'hu1_log', 'ar2')
    scat(axes[2], df['hu2_log'], df['ar2'], 'hu2_log vs ar2', 'hu2_log', 'ar2')

    cent_scaled = kmeans.cluster_centers_
    cent_orig = scaler.inverse_transform(cent_scaled)  # [hu1_log, hu2_log, ar2]
    axes[0].scatter(cent_orig[:,0], cent_orig[:,1], marker='X', s=120, edgecolors='k')
    axes[1].scatter(cent_orig[:,0], cent_orig[:,2], marker='X', s=120, edgecolors='k')
    axes[2].scatter(cent_orig[:,1], cent_orig[:,2], marker='X', s=120, edgecolors='k')

    # leyenda consistente por color
    orden_clusters = sorted(np.unique(labels).tolist())
    handles = [Line2D([], [], marker='o', linestyle='None', markersize=8,
                      color=cmap(c), label=mayoritaria.get(c, f"Cluster {c}"))
               for c in orden_clusters]
    for ax in axes:
        leg = ax.legend(handles=handles, title="Elemento", loc='best', frameon=True)
        try: leg._legend_box.align = "left"
        except Exception: pass

    plt.tight_layout()

    # ---- 3D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig3 = plt.figure(figsize=(7.5, 6.5))
    ax3 = fig3.add_subplot(111, projection='3d')
    p3 = ax3.scatter(df['hu1_log'], df['hu2_log'], df['ar2'],
                     c=labels, cmap=cmap, alpha=0.85, s=20)
    ax3.set_xlabel('hu1_log'); ax3.set_ylabel('hu2_log'); ax3.set_zlabel('ar2')
    ax3.set_title('Espacio 3D: hu1_log, hu2_log, ar2')
    # centroides en 3D
    ax3.scatter(cent_orig[:,0], cent_orig[:,1], cent_orig[:,2],
                marker='X', s=160, edgecolors='k')
    leg3 = ax3.legend(handles=handles, title="Elemento", loc='upper right', frameon=True)
    try: leg3._legend_box.align = "left"
    except Exception: pass

    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Ruta al CSV de features.")
    ap.add_argument("--auto-k", type=str, default="true", help="true/false (usar silhouette).")
    ap.add_argument("--k", type=int, default=4, help="K manual si --auto-k=false.")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=7)
    args = ap.parse_args()

    csv_path = resolve_csv_path(args.csv)
    print(f"Usando CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # --- Normalizar nombres de clase ---
    if 'clase' in df.columns:
        df['clase'] = df['clase'].astype(str).str.strip().str.lower().str.capitalize()

    req_cols = {'hu1','hu2','ar2'}
    faltan = req_cols - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en {csv_path.name}: {faltan}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['hu1','hu2','ar2']).copy()
    if df.empty:
        raise ValueError("No hay filas válidas tras limpiar NaN/inf.")

    # features
    df['hu1_log'] = log_transform_hu(df['hu1'].values)
    df['hu2_log'] = log_transform_hu(df['hu2'].values)
    X = df[['hu1_log','hu2_log','ar2']].values

    # escalar
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # elegir K
    use_auto = str(args.auto_k).strip().lower() in {"1","true","yes","y","t"}
    if use_auto:
        best_k, best_score = None, -1.0
        silh_scores = {}
        for k in range(max(2, args.kmin), max(args.kmin, args.kmax)+1):
            mbk = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                                  batch_size=min(512, max(64, Xs.shape[0]//8)),
                                  n_init=10, max_no_improvement=20)
            lab_k = mbk.fit_predict(Xs)
            Xe, ye = maybe_downsample(Xs, lab_k)
            score = silhouette_score(Xe, ye)
            silh_scores[k] = round(float(score), 4)
            if score > best_score: best_k, best_score = k, score
        print("Silhouette por K:", silh_scores)
        K = best_k
        print(f"→ K seleccionado automáticamente: {K} (silhouette={best_score:.4f})")
    else:
        K = int(args.k)
        print(f"→ K fijado manualmente: {K}")

    # ==========================================================
    # Entrenar modelo final con centroides predefinidos por clase
    # ==========================================================
    batch_size = min(512, max(64, Xs.shape[0]//8))

    if 'clase' in df.columns:
        # Agrupar por clase y calcular centroides medios en el espacio escalado
        class_means = (
            df.groupby('clase')[['hu1_log','hu2_log','ar2']]
            .mean()
            .reindex(['Arandela','Tuerca','Tornillo','Clavo'])
            .dropna()
            .values
        )
        # Escalar esos centroides
        class_means_scaled = StandardScaler().fit(X).transform(class_means)

        # Forzar K=4 y usar esos centroides como inicio
        K = 4
        print("\nUsando centroides iniciales definidos por clase (K=4):")
        print(class_means)

        kmeans = MiniBatchKMeans(
            n_clusters=K,
            random_state=RANDOM_STATE,
            batch_size=batch_size,
            init=class_means_scaled,
            n_init=1,              # importante: no re-inicializar
            max_no_improvement=30
        )
    else:
        # fallback automático si no hay clases
        kmeans = MiniBatchKMeans(n_clusters=K, random_state=RANDOM_STATE,
                                batch_size=batch_size, n_init=10, max_no_improvement=30)

    labels = kmeans.fit_predict(Xs)

    df['cluster'] = labels

    # asignación 1-a-1 cluster→clase (si hay 'clase')
    mayoritaria = assign_clusters_one_to_one(df)

    # gráficos 2D + 3D
    plot_2d_and_3d(df, labels, kmeans, scaler, mayoritaria)

    # guardar artefactos
    df.to_csv(CSV_OUT, index=False)
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(kmeans, MODEL_OUT)
    print(f"\n✓ Guardado DataFrame con clusters en: {CSV_OUT}")
    print(f"✓ Guardado scaler en: {SCALER_OUT}")
    print(f"✓ Guardado modelo KMeans en: {MODEL_OUT}")

if __name__ == "__main__":
    main()
