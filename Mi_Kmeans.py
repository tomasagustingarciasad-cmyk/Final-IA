# Kmeans_puro.py — K-Means completo sin sklearn (MiniBatch + Silhouette + Escalado interno)
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import joblib

# =========================
# Config por defecto
# =========================
CSV_DEFAULT_NAME = "cualidades_imagenes.csv"
CSV_OUT = "Mi_features_clusterizados.csv"
SCALER_OUT = "Mi_scaler_kmeans.joblib"             # guardamos metadatos del escalado interno
MODEL_OUT = "Mi_kmeans_mbkmeans.joblib"            # guardamos el objeto KMeans puro
RANDOM_STATE = 42
MAX_SILH_SAMPLES = 5000
CLASS_ORDER = ['Arandela', 'Tuerca', 'Tornillo', 'Clavo']
RUNTIME_COLS = ['hu1', 'hu2', 'ar2']  # nombres que la app espera

# =========================
# Utils generales
# =========================
def log_transform_hu(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    return -np.sign(x) * np.log10(np.abs(x) + eps)

def _as_bool(s: str) -> bool:
    return str(s).strip().lower() in {"1","true","yes","y","t"}


def _maybe_downsample(X, y=None, max_n: int = MAX_SILH_SAMPLES, random_state: int = RANDOM_STATE):
    n = X.shape[0]
    if n <= max_n:
        return X, y
    rng = np.random.default_rng(random_state)
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
        if c.exists():
            return c.resolve()
    hits = list(here.rglob(CSV_DEFAULT_NAME))
    if hits:
        return hits[0].resolve()
    raise FileNotFoundError(f"No se encontró {CSV_DEFAULT_NAME} cerca de {here}")


def build_class_prototypes(
    df: pd.DataFrame,
    cols=('hu1_log','hu2_log','ar2'),
    class_order=CLASS_ORDER,
    mode: str = 'mean'  # 'mean' | 'medoid' | 'sample'
) -> tuple[np.ndarray, list[str]]:
    """
    Devuelve (seeds, used_classes) donde:
      - seeds: ndarray (K, 3) en ESCALA ORIGINAL (hu1_log, hu2_log, ar2)
      - used_classes: lista de nombres de clase en el mismo orden que seeds
    """
    seeds = []
    used = []
    for cls in class_order:
        g = df[df['clase'] == cls]
        if g.empty:
            continue
        Xg = g[list(cols)].to_numpy(dtype=float)
        if mode == 'mean':
            seeds.append(Xg.mean(axis=0))
            used.append(cls)
        elif mode == 'medoid':
            mu = Xg.mean(axis=0)
            idx = int(np.argmin(((Xg - mu)**2).sum(axis=1)))
            seeds.append(Xg[idx])
            used.append(cls)
        elif mode == 'sample':
            rng = np.random.default_rng(RANDOM_STATE)
            idx = int(rng.integers(0, Xg.shape[0]))
            seeds.append(Xg[idx])
            used.append(cls)
        else:
            raise ValueError("mode debe ser 'mean' | 'medoid' | 'sample'")
    if not seeds:
        return np.empty((0, len(cols))), []
    return np.vstack(seeds), used






def assign_clusters_one_to_one(df: pd.DataFrame) -> dict[int, str]:
    """
    Devuelve un dict {cluster:int -> clase:str} con asignación 1-a-1.
    Usa Hungarian si SciPy está disponible; sino greedy.
    """
    if 'clase' not in df.columns:
        return {}

    df = df.copy()

    # ✅ Todas las operaciones de string con .str
    df['clase_norm'] = df['clase'].astype(str).str.strip().str.lower()
    df['clase_norm'] = (
        df['clase_norm']
        .map({'arandela':'Arandela','clavo':'Clavo','tornillo':'Tornillo','tuerca':'Tuerca'})
        .fillna(df['clase'].astype(str).str.strip().str.capitalize())
    )

    ct = pd.crosstab(df['cluster'], df['clase_norm'])
    if ct.empty:
        return {}

    mayoritaria = {}
    try:
        # Intentar Hungarian si hay SciPy
        from scipy.optimize import linear_sum_assignment  # type: ignore
        cost = ct.max().max() - ct.values  # queremos maximizar conteos
        r_idx, c_idx = linear_sum_assignment(cost)
        for r, c in zip(r_idx, c_idx):
            mayoritaria[int(ct.index[r])] = str(ct.columns[c])
    except Exception:
        # Greedy si no hay SciPy
        temp = ct.values.copy()
        used_r, used_c = set(), set()
        while len(used_r) < ct.shape[0] and len(used_c) < ct.shape[1]:
            r, c = np.unravel_index(np.argmax(temp), temp.shape)
            if temp[r, c] < 0:
                break
            if r in used_r or c in used_c:
                temp[r, c] = -1
                continue
            mayoritaria[int(ct.index[r])] = str(ct.columns[c])
            used_r.add(r); used_c.add(c)
            temp[r, :] = -1; temp[:, c] = -1

    # completar clusters no asignados con mayoría simple
    for clu in sorted(df['cluster'].unique()):
        if clu not in mayoritaria:
            mayoritaria[clu] = df.loc[df['cluster'] == clu, 'clase_norm'].value_counts().idxmax()

    print("\nCrosstab cluster×clase:\n", ct)
    print("\nAsignación 1-a-1 cluster→clase:", mayoritaria)
    return mayoritaria


def plot_2d_and_3d(df, labels, kmeans, mayoritaria):
    COLOR_MAP = {
        'Arandela': '#1f77b4',  # azul
        'Tuerca': '#ff7f0e',    # naranja
        'Tornillo': '#2ca02c',  # verde
        'Clavo': '#d62728'      # rojo
    }
    
    # Mapear cada punto a su color según la clase asignada
    colores = [COLOR_MAP.get(mayoritaria.get(lab, ''), '#7f7f7f') for lab in labels]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def scat(ax, x, y, title, xl, yl):
        sc = ax.scatter(x, y, c=colores, alpha=0.8, s=20)
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl)
        return sc

    scat(axes[0], df['hu1_log'], df['hu2_log'], 'hu1_log vs hu2_log', 'hu1_log', 'hu2_log')
    scat(axes[1], df['hu1_log'], df['ar2'], 'hu1_log vs ar2', 'hu1_log', 'ar2')
    scat(axes[2], df['hu2_log'], df['ar2'], 'hu2_log vs ar2', 'hu2_log', 'ar2')

    cent_orig = kmeans.cluster_centers_  # ya en escala original
    axes[0].scatter(cent_orig[:, 0], cent_orig[:, 1], marker='X', s=120, edgecolors='k')
    axes[1].scatter(cent_orig[:, 0], cent_orig[:, 2], marker='X', s=120, edgecolors='k')
    axes[2].scatter(cent_orig[:, 1], cent_orig[:, 2], marker='X', s=120, edgecolors='k')

    orden_clusters = sorted(np.unique(labels).tolist())
    handles = [
        Line2D([], [], marker='o', linestyle='None', markersize=8,
               color=COLOR_MAP.get(mayoritaria.get(c, ''), '#7f7f7f'), 
               label=mayoritaria.get(c, f"Cluster {c}"))
        for c in orden_clusters
    ]
    for ax in axes:
        leg = ax.legend(handles=handles, title="Elemento", loc='best', frameon=True)
        try:
            leg._legend_box.align = "left"  # type: ignore
        except Exception:
            pass

    plt.tight_layout()

    # 3D
    from mpl_toolkits.mplot3d import Axes3D
    fig3 = plt.figure(figsize=(7.5, 6.5))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.scatter(df['hu1_log'], df['hu2_log'], df['ar2'], c=colores, alpha=0.85, s=20)
    ax3.set_xlabel('hu1_log'); ax3.set_ylabel('hu2_log'); ax3.set_zlabel('ar2')
    ax3.set_title('Espacio 3D: hu1_log, hu2_log, ar2')
    ax3.scatter(cent_orig[:, 0], cent_orig[:, 1], cent_orig[:, 2], marker='X', s=160, edgecolors='k', c='black')
    
    leg3 = ax3.legend(handles=handles, title="Elemento", loc='upper right', frameon=True)
    try:
        leg3._legend_box.align = "left"  # type: ignore
    except Exception:
        pass

    plt.show()

    # === Export runtime (.npz) para la app ===
    # NOTA: en runtime la app usa claves 'hu1','hu2','ar2'.
    # Aquí entrenamos con ('hu1_log','hu2_log','ar2'), pero semánticamente
    # 'hu1' en runtime = 'hu1_log' aquí (porque la app también hace log-transform).
    
    

    # medias/desvíos en el espacio de entrenamiento: (hu1_log, hu2_log, ar2)
    X_for_stats = df[['hu1_log','hu2_log','ar2']].to_numpy(dtype=float)
    mean = X_for_stats.mean(axis=0)
    std  = X_for_stats.std(axis=0, ddof=0)
    std[std == 0.0] = 1.0

    # centroides del modelo en el mismo espacio (hu1_log,hu2_log,ar2)
    cent_orig = kmeans.cluster_centers_          # (K,3)
    cent_z    = (cent_orig - mean) / std         # exportamos en z-score

    # orden de etiquetas para runtime
    K = cent_z.shape[0]
    try:
        # si mapeaste por índice (seeds), respeta 'used_classes'
        label_order = [mayoritaria[i] for i in range(K)]
    except Exception:
        from itertools import islice
        label_order = list(islice(CLASS_ORDER, K))

    out_dir = Path("models"); out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "kmeans_puro.npz",
             centroids=cent_z,       # (K,3) en z-score
             mean=mean,              # (3,)
             std=std,                # (3,)
             kept_cols=np.array(RUNTIME_COLS, dtype=object),  # nombres que usará la app
             label_order=np.array(label_order, dtype=object))

    print("\n✓ Export runtime -> models/kmeans_puro.npz")
    print("  kept_cols   :", RUNTIME_COLS)
    print("  label_order :", label_order)

# =========================
# KMeans puro (sin sklearn)
# =========================
Array = np.ndarray

def _check_random_state(seed: int | None) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)

@dataclass
class KMeans:
    """
    K-Means desde cero (sin sklearn), con:
      - init: 'k-means++' | 'random' | ndarray (centroides en escala original)
      - method: 'full' o 'minibatch'
      - n_init: corridas independientes, elegimos la de menor inercia
      - max_iter, tol
      - scale: None | 'standard' | 'minmax' (escalado interno)
      - empty_action: 'farthest' | 'random' | 'error'
    Atributos tras fit():
      - cluster_centers_: (k, d) en escala original
      - labels_: (n,)
      - inertia_: suma de distancias cuadradas (espacio escalado)
      - n_iter_: iteraciones de la mejor corrida
    """
    n_clusters: int
    init: Array | str = "k-means++"
    method: str = "minibatch"          # por defecto replicamos MiniBatch
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    scale: str | None = "minmax"       # por defecto como tu script
    empty_action: str = "farthest"
    batch_size: int = 256
    max_no_improvement: int = 30
    random_state: int | None = RANDOM_STATE
    verbose: bool = False

    # Seteados tras fit()
    cluster_centers_: Array | None = None
    labels_: Array | None = None
    inertia_: float | None = None
    n_iter_: int | None = None

    # Escalado interno
    _scale_kind: str | None = None
    _shift_: Array | None = None
    _scale_: Array | None = None

    # ===== API =====
    def fit(self, X: Array) -> "KMeans":
        X = self._as_2d_float(X)
        self._prepare_scaler(X)
        Xs = self._apply_scale(X)

        best = {"inertia": np.inf, "centers_s": None, "labels": None, "n_iter": None}
        rng_master = _check_random_state(self.random_state)

        for run in range(self.n_init):
            rng = _check_random_state(int(rng_master.integers(0, 2**31 - 1)))
            centers_s = self._init_centroids(Xs, rng)
            if self.method == "minibatch":
                labels, inertia, n_iter, centers_s = self._iterate_minibatch(Xs, centers_s, rng)
            elif self.method == "full":
                labels, inertia, n_iter, centers_s = self._iterate_full(Xs, centers_s, rng)
            else:
                raise ValueError("method debe ser 'full' o 'minibatch'.")

            if self.verbose:
                print(f"[KMeans] run {run+1}/{self.n_init} -> inertia={inertia:.6f}, iters={n_iter}")

            if inertia < best["inertia"]:
                best.update(inertia=inertia, centers_s=centers_s, labels=labels, n_iter=n_iter)

        self.labels_ = best["labels"]
        self.n_iter_ = int(best["n_iter"])
        self.inertia_ = float(best["inertia"])
        # Exponer centroides en escala original
        self.cluster_centers_ = self._inverse_scale(best["centers_s"])
        return self

    def predict(self, X: Array) -> Array:
        self._require_fitted()
        X = self._as_2d_float(X)
        Xs = self._apply_scale(X)
        centers_s = self._apply_scale(self.cluster_centers_)
        labels, _ = self._assign(Xs, centers_s)
        return labels

    def fit_predict(self, X: Array) -> Array:
        self.fit(X)
        return self.labels_.copy()

    def transform(self, X: Array) -> Array:
        self._require_fitted()
        X = self._as_2d_float(X)
        Xs = self._apply_scale(X)
        centers_s = self._apply_scale(self.cluster_centers_)
        return self._pairwise_sq_dists(Xs, centers_s)

    def score(self, X: Array) -> float:
        labels = self.predict(X)
        X = self._as_2d_float(X)
        Xs = self._apply_scale(X)
        centers_s = self._apply_scale(self.cluster_centers_)
        inertia = self._inertia_from_labels(Xs, labels, centers_s)
        return -float(inertia)

    # ===== Núcleo =====
    def _iterate_full(self, Xs: Array, centers_s: Array, rng: np.random.Generator):
        prev_centers = centers_s.copy()
        for it in range(1, self.max_iter + 1):
            labels, d2min = self._assign(Xs, centers_s)
            centers_s, labels = self._update_centers(Xs, labels, d2min, centers_s, rng)

            shift = np.linalg.norm(centers_s - prev_centers, axis=1).max()
            if self.verbose and it % 10 == 0:
                print(f"  iter {it:3d}: max_shift={shift:.6e}")
            if shift <= self.tol:
                break
            prev_centers = centers_s.copy()
        inertia = self._inertia_from_labels(Xs, labels, centers_s)
        return labels, float(inertia), it, centers_s

    def _iterate_minibatch(self, Xs: Array, centers: Array, rng: np.random.Generator):
        n, k = Xs.shape[0], centers.shape[0]
        counts = np.zeros(k, dtype=np.int64)
        best_batch_cost = np.inf
        no_improv = 0
        iters = 0

        for epoch in range(1, self.max_iter + 1):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                iters += 1
                end = min(start + self.batch_size, n)
                B = Xs[perm[start:end]]

                lab_b, d2min_b = self._assign(B, centers)
                # actualización incremental por cluster
                for j in range(k):
                    mask = (lab_b == j)
                    nj = int(mask.sum())
                    if nj == 0:
                        continue
                    mean_j = B[mask].mean(axis=0)
                    new_count = counts[j] + nj
                    eta = nj / float(new_count)
                    centers[j] = (1.0 - eta) * centers[j] + eta * mean_j
                    counts[j] = new_count

                batch_cost = float(d2min_b.mean())
                if batch_cost + 1e-12 < best_batch_cost - 1e-12:
                    best_batch_cost = batch_cost
                    no_improv = 0
                else:
                    no_improv += 1
                    if no_improv >= self.max_no_improvement:
                        break
            if no_improv >= self.max_no_improvement:
                break

        labels, d2min = self._assign(Xs, centers)
        inertia = float(d2min.sum())
        return labels, inertia, iters, centers

    def _assign(self, Xs: Array, centers_s: Array):
        d2 = self._pairwise_sq_dists(Xs, centers_s)
        labels = np.argmin(d2, axis=1)
        d2min = d2[np.arange(Xs.shape[0]), labels]
        return labels, d2min

    def _update_centers(self, Xs: Array, labels: Array, d2min: Array,
                        centers_s: Array, rng: np.random.Generator):
        k = self.n_clusters
        d = Xs.shape[1]
        new_centers = np.zeros((k, d), dtype=np.float64)

        for j in range(k):
            mask = (labels == j)
            if np.any(mask):
                new_centers[j] = Xs[mask].mean(axis=0)
            else:
                if self.empty_action == "error":
                    raise ValueError(f"Cluster {j} quedó vacío.")
                elif self.empty_action == "random":
                    idx = int(rng.integers(0, Xs.shape[0]))
                    new_centers[j] = Xs[idx]
                elif self.empty_action == "farthest":
                    idx = int(np.argmax(d2min))
                    labels[idx] = j
                    d2min[idx] = -1.0
                    new_centers[j] = Xs[idx]
                else:
                    raise ValueError("empty_action debe ser 'farthest', 'random' o 'error'.")

        if self.empty_action == "farthest":
            for j in range(k):
                mask = (labels == j)
                new_centers[j] = Xs[mask].mean(axis=0)

        return new_centers, labels

    # ===== Inicialización =====
    def _init_centroids(self, Xs: Array, rng: np.random.Generator) -> Array:
        n, d = Xs.shape
        k = self.n_clusters

        # Centroides provistos por el usuario (EN ESCALA ORIGINAL)
        if isinstance(self.init, np.ndarray):
            C = np.asarray(self.init, dtype=np.float64)
            if C.shape != (k, d):
                raise ValueError(f"init array debe tener shape ({k},{d}), recibido {C.shape}")
            if self._scale_kind is not None:
                C = (C - self._shift_) / self._scale_
            return C.copy()

        if self.init == "random":
            if k > n:
                raise ValueError(f"n_clusters={k} > n_samples={n}.")
            idx = rng.choice(n, size=k, replace=False)
            return Xs[idx].astype(np.float64, copy=True)

        if self.init == "k-means++":
            centers = np.empty((k, d), dtype=np.float64)
            idx0 = int(rng.integers(0, n))
            centers[0] = Xs[idx0]
            closest_d2 = self._pairwise_sq_dists(Xs, centers[0:1]).ravel()
            for j in range(1, k):
                probs = closest_d2 / closest_d2.sum()
                next_idx = int(rng.choice(n, p=probs))
                centers[j] = Xs[next_idx]
                d2_new = self._pairwise_sq_dists(Xs, centers[j:j+1]).ravel()
                closest_d2 = np.minimum(closest_d2, d2_new)
            return centers

        raise ValueError("init debe ser 'random' o 'k-means++' o ndarray.")

    # ===== Utilidades numéricas / escalado =====
    @staticmethod
    def _as_2d_float(X: Array) -> Array:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X debe ser 2D (n_samples, n_features).")
        return X

    @staticmethod
    def _pairwise_sq_dists(A: Array, B: Array) -> Array:
        A2 = np.sum(A * A, axis=1, keepdims=True)      # (n,1)
        B2 = np.sum(B * B, axis=1, keepdims=True).T    # (1,k)
        AB = A @ B.T                                   # (n,k)
        return A2 - 2.0 * AB + B2

    def _prepare_scaler(self, X: Array) -> None:
        self._scale_kind = (self.scale or None)
        if self._scale_kind is None:
            self._shift_ = None
            self._scale_ = None
            return
        if self._scale_kind == "standard":
            mu = X.mean(axis=0)
            sigma = X.std(axis=0, ddof=0)
            sigma[sigma == 0.0] = 1.0
            self._shift_ = mu
            self._scale_ = sigma
            return
        if self._scale_kind == "minmax":
            xmin = X.min(axis=0)
            xmax = X.max(axis=0)
            scale = xmax - xmin
            scale[scale == 0.0] = 1.0
            self._shift_ = xmin
            self._scale_ = scale
            return
        raise ValueError("scale debe ser None | 'standard' | 'minmax'.")

    def _apply_scale(self, X: Array) -> Array:
        if self._scale_kind is None:
            return X
        return (X - self._shift_) / self._scale_

    def _inverse_scale(self, Xs: Array) -> Array:
        if self._scale_kind is None:
            return Xs
        return Xs * self._scale_ + self._shift_

    @staticmethod
    def _inertia_from_labels(Xs: Array, labels: Array, centers_s: Array) -> float:
        d2 = KMeans._pairwise_sq_dists(Xs, centers_s)
        d2min = d2[np.arange(Xs.shape[0]), labels]
        return float(d2min.sum())

    def _require_fitted(self) -> None:
        if self.cluster_centers_ is None:
            raise RuntimeError("Debes llamar a fit() antes de predecir/transformar.")

# =========================
# Silhouette y Auto-K
# =========================
def silhouette_score_np(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, float)
    labels = np.asarray(labels, int)
    n = X.shape[0]
    if n < 3 or len(np.unique(labels)) < 2:
        return 0.0
    d2 = KMeans._pairwise_sq_dists(X, X)
    D = np.sqrt(np.maximum(d2, 0.0))

    uniq = np.unique(labels)
    a = np.zeros(n)
    b = np.full(n, np.inf)

    # a(i): distancia media intra-cluster
    for c in uniq:
        idx = np.where(labels == c)[0]
        if idx.size <= 1:
            a[idx] = 0.0
        else:
            Dij = D[np.ix_(idx, idx)]
            sums = Dij.sum(axis=1)  # incluye 0 en diagonal
            a[idx] = sums / (idx.size - 1)

    # b(i): mínima media a otros clusters
    for i in range(n):
        li = labels[i]
        best = np.inf
        for c in uniq:
            if c == li:
                continue
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                continue
            md = D[i, idx].mean()
            if md < best:
                best = md
        b[i] = best

    s = (b - a) / np.maximum(a, b)
    return float(np.nanmean(s))

def auto_k_silhouette(
    X: np.ndarray,
    kmin: int = 2,
    kmax: int = 7,
    max_samples: int = MAX_SILH_SAMPLES,
    random_state: int = RANDOM_STATE,
    **km_kwargs
):
    best_k, best_s = None, -1.0
    scores = {}
    # probamos cada K con los mismos kwargs (scale/method/etc.)
    for k in range(max(2, kmin), max(kmin, kmax) + 1):
        km = KMeans(n_clusters=k, random_state=random_state, **km_kwargs).fit(X)
        # silhouette en el espacio escalado del propio modelo
        Xs = km._apply_scale(np.asarray(X, float))
        Xe, ye = _maybe_downsample(Xs, km.labels_, max_n=max_samples, random_state=random_state)
        s = silhouette_score_np(Xe, ye)
        scores[k] = round(float(s), 4)
        if s > best_s:
            best_k, best_s = k, s
    return best_k, best_s, scores

# =========================
# Main (igual interfaz que tu script)
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Ruta al CSV de features.")
    ap.add_argument("--auto-k", type=str, default="true",
                    help="true/false: usar silhouette SOLO si NO hay etiquetas.")
    ap.add_argument("--k", type=int, default=4, help="K manual si --auto-k=false.")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=7)
    ap.add_argument("--seed-mode", type=str, default="medoid",
                    help="How to construir semillas: mean | medoid | sample")
    ap.add_argument("--map-by-index", type=str, default="true",
                    help="true/false: fijar mapeo cluster->clase por índice según CLASS_ORDER")
    args = ap.parse_args()


    csv_path = resolve_csv_path(args.csv)
    print(f"Usando CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalizar nombres de clase
    if 'clase' in df.columns:
        df['clase'] = df['clase'].astype(str).str.strip().str.lower().str.capitalize()

    req_cols = {'hu1', 'hu2', 'ar2'}
    faltan = req_cols - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en {csv_path.name}: {faltan}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['hu1', 'hu2', 'ar2']).copy()
    if df.empty:
        raise ValueError("No hay filas válidas tras limpiar NaN/inf.")

    # Features (igual que tu script)
    df['hu1_log'] = log_transform_hu(df['hu1'].values)
    df['hu2_log'] = log_transform_hu(df['hu2'].values)
    X = df[['hu1_log', 'hu2_log', 'ar2']].values

    # Elegir K
    use_auto = _as_bool(args.auto_k)
    map_by_index = _as_bool(args.map_by_index)
    batch_size = min(512, max(64, X.shape[0] // 8))

    have_labels = ('clase' in df.columns) and (df['clase'].notna().any())

    if have_labels:
        # 1) Construir semillas con datos etiquetados (ESCALA ORIGINAL)
        seeds, used_classes = build_class_prototypes(
            df, cols=('hu1_log','hu2_log','ar2'),
            class_order=CLASS_ORDER,
            mode=args.seed_mode.strip().lower()
        )
        if seeds.shape[0] == 0:
            raise ValueError("No hay clases con datos suficientes para construir semillas.")
        K = seeds.shape[0]
        print(f"\nSemillas construidas a partir de etiquetas (mode={args.seed_mode}):")
        for i, (cls, v) in enumerate(zip(used_classes, seeds)):
            print(f"  {i:>2} -> {cls:<9}  [{v[0]: .3f}, {v[1]: .3f}, {v[2]: .3f}]")

        # 2) Entrenar con esas semillas y pocas iteraciones (refinado leve)
        kmeans = KMeans(
            n_clusters=K,
            init=seeds,              # ← semillas etiquetadas
            n_init=1,                # no re-inicializar
            scale="minmax",
            method="minibatch",
            batch_size=batch_size,
            max_no_improvement=30,
            max_iter=30,             # refinamiento corto
            empty_action="farthest",
            random_state=RANDOM_STATE,
            verbose=False
        )
        labels = kmeans.fit_predict(X)

        # 3) Mapeo cluster→clase
        if map_by_index:
            # Índices en el MISMO orden en que construimos 'seeds'
            mayoritaria = {i: cls for i, cls in enumerate(used_classes)}
        else:
            mayoritaria = assign_clusters_one_to_one(pd.DataFrame({
                'cluster': labels,
                'clase': df['clase']  # usa tus etiquetas reales
            }))
    else:
        # No hay etiquetas -> flujo original con auto-K/silhouette
        if use_auto:
            K, best_s, silh_scores = auto_k_silhouette(
                X,
                kmin=args.kmin,
                kmax=args.kmax,
                random_state=RANDOM_STATE,
                max_samples=MAX_SILH_SAMPLES,
                init="k-means++",
                n_init=10,
                scale="minmax",
                method="minibatch",
                batch_size=batch_size,
                max_no_improvement=20,
                empty_action="farthest",
                verbose=False
            )
            print("Silhouette por K:", silh_scores)
            print(f"→ K seleccionado automáticamente: {K} (silhouette={best_s:.4f})")
        else:
            K = int(args.k)
            print(f"→ K fijado manualmente: {K}")

        kmeans = KMeans(
            n_clusters=K,
            init="k-means++",
            n_init=10,
            scale="minmax",
            method="minibatch",
            batch_size=batch_size,
            max_no_improvement=30,
            empty_action="farthest",
            random_state=RANDOM_STATE,
            verbose=False
        )
        labels = kmeans.fit_predict(X)
        mayoritaria = assign_clusters_one_to_one(pd.DataFrame({
            'cluster': labels,
            'clase': df['clase'] if 'clase' in df.columns else None
        }))

    df['cluster'] = labels
# Solo recalcular si NO estamos mapeando por índice
    if not (have_labels and map_by_index):
        mayoritaria = assign_clusters_one_to_one(df)

    # Asignación 1-a-1 cluster→clase (si hay 'clase')
    mayoritaria = assign_clusters_one_to_one(df)

    # Gráficos 2D + 3D
    plot_2d_and_3d(df, labels, kmeans, mayoritaria)

    # Guardados (igual interfaz que antes)
    columnas_deseadas = ['file', 'clase', 'hu1_log', 'hu2_log', 'ar2', 'cluster']
    columnas_presentes = [c for c in columnas_deseadas if c in df.columns]
    df_salida = df[columnas_presentes].copy()

    # Redondeo numérico
    for col in ['ar2', 'hu1_log', 'hu2_log']:
        if col in df_salida.columns:
            df_salida[col] = df_salida[col].astype(float).round(3)

    df_salida.to_csv(CSV_OUT, index=False)
    if have_labels:
        np.savetxt("semillas_usadas.csv", seeds, delimiter=",",
                header="hu1_log,hu2_log,ar2", comments="")
        print(f"\n✓ Guardado semillas usadas en: semillas_usadas.csv")


    # Guardar “scaler” (metadatos internos) y modelo
    meta_scaler = {"kind": kmeans._scale_kind, "shift": kmeans._shift_, "scale": kmeans._scale_}
    joblib.dump(meta_scaler, SCALER_OUT)
    joblib.dump(kmeans, MODEL_OUT)

    print(f"\n✓ Guardado DataFrame con clusters en: {CSV_OUT}")
    print(f"✓ Guardado 'scaler' interno en: {SCALER_OUT}")
    print(f"✓ Guardado modelo KMeans en: {MODEL_OUT}")

if __name__ == "__main__":
    main()
