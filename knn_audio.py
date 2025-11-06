
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter, defaultdict

CSV_PATH = Path("base_datos/features_audio.csv")
TARGET = "clase"
FEATURE_REGEX = r"^s\d+_"  
VAR_THR = 1e-6              
TEST_SIZE = 0.20
RANDOM_STATE = 42
K_NEIGHBORS = 7
MINKOWSKI_P = 2            # p=2 Euclídea, p=1 Manhattan
WEIGHTED = True   


rng = np.random.default_rng(RANDOM_STATE)

def variance_threshold(X_df: pd.DataFrame, thr: float):
    """Filtra columnas con varianza < thr. Devuelve (X_filtrado, kept_cols)."""
    if X_df.shape[1] == 0:
        return X_df, []
    var = X_df.var(axis=0, ddof=0)  
    kept_cols = var[var >= thr].index.tolist()
    if len(kept_cols) == 0:
        return X_df, X_df.columns.tolist() 
    return X_df[kept_cols].copy(), kept_cols

class Standardizer:
    """Estandarizador simple (media/varianza) entrenado en TRAIN y aplicado a TRAIN/TEST."""
    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        # evitar división por 0
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray):
        return (X - self.mean_) / self.std_

def stratified_train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Split estratificado si todas las clases tienen al menos 2 muestras; si no, split simple."""
    y = np.asarray(y)
    n = len(y)
    idx_all = np.arange(n)

    uniques, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        perm = rng.permutation(n)
        n_test = max(1, int(round(n * test_size)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        return train_idx, test_idx, False

    per_class_indices = {c: idx_all[y == c] for c in uniques}
    test_idx_list = []
    train_idx_list = []
    for c, ids in per_class_indices.items():
        ids = np.array(ids)
        ids = ids[rng.permutation(len(ids))]
        n_test_c = max(1, int(round(len(ids) * test_size)))
        test_idx_list.append(ids[:n_test_c])
        train_idx_list.append(ids[n_test_c:])
    test_idx = np.concatenate(test_idx_list)
    train_idx = np.concatenate(train_idx_list)

    test_idx = test_idx[rng.permutation(len(test_idx))]
    train_idx = train_idx[rng.permutation(len(train_idx))]
    return train_idx, test_idx, True

def confusion_matrix(y_true, y_pred, labels):
    """Matriz de confusión con orden de etiquetas = labels."""
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt in lab_to_idx and yp in lab_to_idx:
            cm[lab_to_idx[yt], lab_to_idx[yp]] += 1
    return cm

def classification_report(y_true, y_pred, labels, zero_division=0):
    """Reporte estilo sklearn (resumen). Devuelve dict y string."""
    cm = confusion_matrix(y_true, y_pred, labels)
    support = cm.sum(axis=1)
    precisions = []
    recalls = []
    f1s = []
    lines = []
    header = f"{'clase':<20} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else (0 if zero_division == 0 else 1)
        rec = tp / (tp + fn) if (tp + fn) > 0 else (0 if zero_division == 0 else 1)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        lines.append(f"{str(lab):<20} {prec:10.3f} {rec:10.3f} {f1:10.3f} {support[i]:8d}")

    support_total = support.sum()
    macro_p = np.mean(precisions) if len(precisions) else 0
    macro_r = np.mean(recalls) if len(recalls) else 0
    macro_f1 = np.mean(f1s) if len(f1s) else 0

    weighted_p = np.average(precisions, weights=support) if support_total > 0 else 0
    weighted_r = np.average(recalls, weights=support) if support_total > 0 else 0
    weighted_f1 = np.average(f1s, weights=support) if support_total > 0 else 0

    lines.append("-" * len(header))
    lines.append(f"{'macro avg':<20} {macro_p:10.3f} {macro_r:10.3f} {macro_f1:10.3f} {support_total:8d}")
    lines.append(f"{'weighted avg':<20} {weighted_p:10.3f} {weighted_r:10.3f} {weighted_f1:10.3f} {support_total:8d}")
    return {
        "per_class": dict(zip(labels, zip(precisions, recalls, f1s, support))),
        "macro_avg": (macro_p, macro_r, macro_f1),
        "weighted_avg": (weighted_p, weighted_r, weighted_f1),
        "support_total": support_total,
    }, "\n".join(lines)


class KNNPuro:
    def __init__(self, n_neighbors=5, p=2, weighted=True, eps=1e-8):
        self.k = int(n_neighbors)
        self.p = float(p)
        self.weighted = bool(weighted)
        self.eps = eps

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        if self.X_train.ndim != 2:
            raise ValueError("X_train debe ser 2D")
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_train y y_train deben tener igual cantidad de filas")
        return self

    def _distances(self, x):
        diff = self.X_train - x
        if self.p == 2:
            return np.sqrt((diff * diff).sum(axis=1))
        elif self.p == 1:
            return np.abs(diff).sum(axis=1)
        else:
            return np.power(np.abs(diff), self.p).sum(axis=1) ** (1.0 / self.p)

    def _vote(self, idx, dist):
        vecinos = self.y_train[idx]
        d = dist[idx]
        zeros = np.where(d <= self.eps)[0]
        if len(zeros) > 0:
            return Counter(vecinos[zeros]).most_common(1)[0][0]

        if not self.weighted:
            return Counter(vecinos).most_common(1)[0][0]

        pesos = 1.0 / (d + self.eps)
        acumulado = defaultdict(float)
        for cls, w in zip(vecinos, pesos):
            acumulado[cls] += w
        # romper empates favoreciendo el vecino más cercano
        max_w = max(acumulado.values())
        candidatos = [c for c, w in acumulado.items() if abs(w - max_w) <= 1e-12]
        if len(candidatos) == 1:
            return candidatos[0]
        else:
            # de los candidatos, elegir el que aparezca primero entre los más cercanos
            for c in vecinos:
                if c in candidatos:
                    return c

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = []
        for x in X:
            dist = self._distances(x)
            idx = np.argsort(dist)[: self.k]
            preds.append(self._vote(idx, dist))
        return np.array(preds)



def main():
    print("🚀 Iniciando entrenamiento de K-NN para audio...\n")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el CSV en: {CSV_PATH.resolve()}")
    df_raw = pd.read_csv(CSV_PATH)
    print(f"✔ CSV cargado: {CSV_PATH}  |  filas={len(df_raw)}, columnas={len(df_raw.columns)}")

    if TARGET not in df_raw.columns:
        raise ValueError(f"No se encontró la columna objetivo '{TARGET}' en el CSV.")

    feat_cols = [c for c in df_raw.columns if re.match(FEATURE_REGEX, c)]
    if not feat_cols:
        aux = {"id", "path_rel", TARGET}
        feat_cols = [c for c in df_raw.columns if c not in aux and pd.api.types.is_numeric_dtype(df_raw[c])]
        print("⚠ No se hallaron columnas que cumplan el patrón "
              f"{FEATURE_REGEX}. Uso fallback: {len(feat_cols)} columnas numéricas.")
    print(f"✔ Features seleccionados: {len(feat_cols)} columnas")


    df = df_raw.copy()
    if "path_rel" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["path_rel"], keep="first")
        print(f"✓ Deduplicado por path_rel: {before} -> {len(df)}")
    elif "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["id"], keep="first")
        print(f"✓ Deduplicado por id: {before} -> {len(df)}")


    df_with_meta = df.copy()

    df = df.drop(columns=[c for c in ["id", "path_rel"] if c in df.columns], errors="ignore")

    
    
    
    
    
    
    
    
    
    # === 4) X, y + limpieza de NaN/Inf ===
    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y_ser = df[TARGET].astype("category")

    # Inf -> NaN
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    # Dropeo de filas con demasiados NaN (p. ej. >10% de las columnas)
    max_nan = int(0.10 * X_df.shape[1])
    keep_mask = X_df.isna().sum(axis=1) <= max_nan
    dropped = (~keep_mask).sum()
    if dropped > 0:
        print(f"✓ Filas removidas por exceso de NaN: {dropped}")
    X_df = X_df.loc[keep_mask].copy()
    y_ser = y_ser.loc[keep_mask].copy()
    df_with_meta = df_with_meta.loc[keep_mask].copy()

    # Imputación simple por mediana
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    # Re-chequeo
    if X_df.shape[0] == 0:
        raise ValueError("Después de la limpieza no quedaron filas. Revisá el CSV / reglas de NaN.")
    if X_df.shape[1] == 0:
        raise ValueError("No hay columnas de features seleccionadas. Revisá el patrón o el CSV.")

    # === 5) Variance Threshold (elimina columnas casi constantes) ===
    X_vt_df, kept_cols = variance_threshold(X_df, VAR_THR)
    if len(kept_cols) < X_df.shape[1]:
        print(f"✓ VarianceThreshold removió {X_df.shape[1] - len(kept_cols)} columnas casi constantes.")
    if X_vt_df.shape[1] == 0:
        print("⚠ Todas las columnas fueron filtradas por varianza. Uso features originales sin VT.")
        X_vt_df = X_df.copy()
        kept_cols = X_df.columns.tolist()

    # Convertir a numpy
    X_all = X_vt_df.to_numpy(dtype=float)
    y_all = y_ser.to_numpy()

    print(f"✔ Dataset listo: n_muestras={X_all.shape[0]}, n_features={X_all.shape[1]}")
    print("Clases y conteos:")
    print(pd.Series(y_all).value_counts())

    # === 6) Split 80/20 (estratificado si es posible) ===
    train_idx, test_idx, strat_ok = stratified_train_test_split(X_all, y_all, TEST_SIZE, RANDOM_STATE)
    if not strat_ok:
        print("⚠ No se puede estratificar (hay clases con 1 muestra). Split simple sin stratify.")
    print(f"✔ Split: train={len(train_idx)} | test={len(test_idx)}")

    X_tr, X_te = X_all[train_idx], X_all[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]

    # === 7) Estandarización (fit en train, aplicar en train y test)
    scaler_eval = Standardizer().fit(X_tr)
    X_tr = scaler_eval.transform(X_tr)
    X_te = scaler_eval.transform(X_te)

    # === 8) KNN puro
    knn = KNNPuro(n_neighbors=K_NEIGHBORS, p=MINKOWSKI_P, weighted=WEIGHTED)
    knn.fit(X_tr, y_tr)
    y_pred = knn.predict(X_te)

    # === 9) Identificar audios mal clasificados ===
    print("\n=== ERRORES DETECTADOS ===")
    cols_meta = [c for c in ["path_rel", "id"] if c in df_with_meta.columns]
    if not cols_meta:
        print("⚠ No se encontró ninguna columna 'path_rel' o 'id' en el CSV, no se puede mostrar el nombre de archivo.")
    else:
        meta_te = df_with_meta.iloc[test_idx][cols_meta].reset_index(drop=True)
        res = pd.DataFrame({
            "y_true": pd.Series(y_te),
            "y_pred": pd.Series(y_pred)
        })
        res = pd.concat([meta_te, res], axis=1)
        errores = res[res.y_true != res.y_pred]
        if errores.empty:
            print("✅ No hay errores en este split.")
        else:
            print("❌ Audios mal clasificados:")
            print(errores)
            errores.to_csv("errores_test.csv", index=False)
            print("\n📁 Se guardó 'errores_test.csv' con los detalles.")

    # === 10) Resultados ===
    print("\n========== RESULTADOS ==========")
    print("Features finales (primeras 20):", kept_cols[:20], "...")

    # Etiquetas ordenadas (como categorías originales si estaban)
    labels = list(pd.Categorical(y_all).categories)

    cm = confusion_matrix(y_te, y_pred, labels)
    cm_df = pd.DataFrame(cm,
                         index=[f"real_{c}" for c in labels],
                         columns=[f"pred_{c}" for c in labels])

    metrics_dict, report_str = classification_report(y_te, y_pred, labels, zero_division=0)

    print("\nReporte de clasificación:")
    print(report_str)
    print("\nMatriz de confusión:")
    print(cm_df)

    # 7) Entrenar CON TODO EL DATASET (sin split)
    print("\n" + "="*50)
    print("🔧 ENTRENAMIENTO FINAL (usando TODO el dataset)")
    print("   (El modelo anterior con split era solo para evaluar)")
    print("="*50 + "\n")
    scaler_final = Standardizer().fit(X_all)
    X_scaled = scaler_final.transform(X_all)
    
    # --- Llamada a la función de ploteo ---
    #plot_3d_scatter(X_scaled, y_all, kept_cols) # <-- AÑADIR ESTA LÍNEA

    knn = KNNPuro(n_neighbors=K_NEIGHBORS, p=MINKOWSKI_P, weighted=WEIGHTED)
    knn.fit(X_scaled, y_all)

    # 8) Guardar modelo completo
    Path("models").mkdir(exist_ok=True)
    np.savez(
        "models/knn_audio_puro.npz",
        X_train=knn.X_train,
        y_train=knn.y_train,
        mean=scaler_final.mean_,
        std=scaler_final.std_,
        kept_cols=np.array(kept_cols, dtype=object),
        k=K_NEIGHBORS,
        p=MINKOWSKI_P,
        weighted=WEIGHTED
    )
    print("\n✅ Modelo K-NN guardado en: models/knn_audio_puro.npz")
    print(f"   - X_train shape: {knn.X_train.shape}")
    print(f"   - Clases: {np.unique(y_all)}")
    print(f"   - K vecinos: {K_NEIGHBORS}")

if __name__ == "__main__":
    main()
