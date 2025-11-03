# -*- coding: utf-8 -*-
import argparse, json, csv
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

# ----------------- Config por defecto -----------------
IN_DIR_DEFAULT  = r"base_datos\Audio_norm"
OUT_CSV_DEFAULT = r"base_datos\features_audio.csv"
OUT_X_DEFAULT   = r"base_datos\X_audio.npy"
OUT_Y_DEFAULT   = r"base_datos\y_audio.npy"
OUT_LBL_DEFAULT = r"base_datos\labels_audio.json"

TARGET_SR = 16000
N_SEG = 10
# MFCC a usar según diapo: 1,2,4,5  (en librosa son índices 0,1,3,4)
MFCC_INDEXES = [0, 1, 3, 4]
N_MFCC = 13

# Clases esperadas (carpetas). Podés cambiar el orden si querés otro mapeo.
CLASSES = ["Contar", "Proporcion", "Salir"]
LABELS = {c: i for i, c in enumerate(CLASSES)}

# ----------------- Utils -----------------
def load_wav_mono(path: Path, sr=TARGET_SR):
    """Lee WAV mono a sr fijo. (Los normalizados ya están a 16k mono, pero validamos.)"""
    y, sr_in = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr_in != sr:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=sr)
        sr_in = sr
    return y.astype(np.float32), sr_in

def split_segments(y: np.ndarray, n_seg=N_SEG):
    """Divide en n_seg segmentos iguales (último absorbe cualquier resto)."""
    N = len(y)
    idx = np.linspace(0, N, n_seg + 1, dtype=int)
    segs = []
    for i in range(n_seg):
        a, b = idx[i], idx[i+1]
        seg = y[a:b]
        if len(seg) < 1:
            # seguridad: si algún segmento queda vacío, mete un cero
            seg = np.zeros(1, dtype=np.float32)
        segs.append(seg)
    return segs

def find_voiced_region(y: np.ndarray, sr=TARGET_SR, thr_rel=0.1, pad_ms=50):
    """Devuelve (i0, i1) de la zona con voz, usando RMS relativo."""
    n_fft = 512; hop = 160; win = 400
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    if rms.size == 0 or np.max(rms) <= 1e-8:
        return 0, len(y)
    thr = thr_rel * float(np.max(rms))
    idx = np.where(rms > thr)[0]
    if idx.size == 0:
        return 0, len(y)
    pad = int(pad_ms/1000.0 * sr)
    i0 = max(0, idx[0]*hop - pad)
    i1 = min(len(y), (idx[-1]*hop + win) + pad)
    return i0, i1


def feats_segment(yseg: np.ndarray, sr=TARGET_SR):
    """14 features por segmento, robusto a silencio."""
    # 1) si el segmento es demasiado chico o casi silencioso → todo 0
    if len(yseg) < 32:
        return [0.0]*14
    seg_rms = float(np.sqrt(np.mean(yseg.astype(np.float64)**2) + 1e-12))
    if seg_rms < 1e-4:   # umbral de 'silencio'
        return [0.0]*14

    # 2) params
    n_fft = 512
    hop = 160
    win = 400

    # 3) ZCR y RMS
    zcr = librosa.feature.zero_crossing_rate(y=yseg, frame_length=win, hop_length=hop, center=True)
    rms = librosa.feature.rms(y=yseg, frame_length=win, hop_length=hop, center=True)
    zcr_mean = float(np.mean(zcr))
    rms_mean = float(np.mean(rms))

    # 4) MFCC estable (vía mel + power_to_db con ref=1.0)
    S = librosa.feature.melspectrogram(
        y=yseg, sr=sr, n_fft=n_fft, hop_length=hop, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=1.0)  # evita -inf cuando todo es pequeño
    mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=N_MFCC, dct_type=2, norm='ortho')

    # Limpieza por si acaso
    mfcc = np.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)

    stats = []
    for k in MFCC_INDEXES:
        c = mfcc[k, :] if mfcc.shape[1] > 0 else np.array([0.0], dtype=np.float32)
        stats.extend([float(np.mean(c)), float(np.max(c)), float(np.std(c))])

    return [zcr_mean, rms_mean] + stats


def header_names(n_seg=N_SEG):
    names = []
    for s in range(1, n_seg+1):
        names += [
            f"s{s:02d}_zcr_mean",
            f"s{s:02d}_rms_mean",
            f"s{s:02d}_mfcc1_mean", f"s{s:02d}_mfcc1_max", f"s{s:02d}_mfcc1_std",
            f"s{s:02d}_mfcc2_mean", f"s{s:02d}_mfcc2_max", f"s{s:02d}_mfcc2_std",
            f"s{s:02d}_mfcc4_mean", f"s{s:02d}_mfcc4_max", f"s{s:02d}_mfcc4_std",
            f"s{s:02d}_mfcc5_mean", f"s{s:02d}_mfcc5_max", f"s{s:02d}_mfcc5_std",
        ]
    return names

# ----------------- Pipeline -----------------
def process_all(in_dir: Path, out_csv: Path, out_X: Path, out_y: Path, out_lbl: Path):
    rows = []
    X_list, y_list = [], []

    for cls in CLASSES:
        sub = in_dir / cls
        if not sub.exists():
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() != ".wav" or not f.is_file():
                continue
            try:
                y, sr = load_wav_mono(f, sr=TARGET_SR)
                i0, i1 = find_voiced_region(y, sr)
                y_voiced = y[i0:i1] if (i1 - i0) > sr*0.1 else y  # si quedó muy corto, usa entero
                segs = split_segments(y_voiced, n_seg=N_SEG)

                feats = []
                for seg in segs:
                    feats.extend(feats_segment(seg, sr=sr))
                # sanity check: 14 * N_SEG
                if len(feats) != 14 * N_SEG:
                    raise RuntimeError(f"Dimensión inesperada de features para {f.name}: {len(feats)}")

                # fila CSV y matrices
                rel_path = f.relative_to(in_dir).as_posix()
                file_id = f.stem
                rows.append([file_id, cls, rel_path] + feats)
                X_list.append(feats)
                y_list.append(LABELS[cls])

                print(f"✓ {rel_path}")
            except Exception as e:
                print(f"✗ ERROR {f}: {e}")

    # Guardar CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "clase", "path_rel"] + header_names(N_SEG))
        for r in rows:
            w.writerow(r)

    # Guardar arrays
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    np.save(out_X, X)
    np.save(out_y, y)

    # Guardar labels
    with out_lbl.open("w", encoding="utf-8") as fh:
        json.dump(LABELS, fh, ensure_ascii=False, indent=2)

    print(f"\nListo. {X.shape[0]} audios procesados. "
          f"Dimensión de features: {X.shape[1]} (esperado {14*N_SEG}).")
    print(f"CSV: {out_csv}")
    print(f"X:   {out_X}")
    print(f"y:   {out_y}")
    print(f"labels: {out_lbl}")

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="Extracción de features por segmentos para comandos de voz.")
    parser.add_argument("--in_dir",  default=IN_DIR_DEFAULT, help="Carpeta con WAV normalizados (subcarpetas por clase).")
    parser.add_argument("--out_csv", default=OUT_CSV_DEFAULT)
    parser.add_argument("--out_X",   default=OUT_X_DEFAULT)
    parser.add_argument("--out_y",   default=OUT_Y_DEFAULT)
    parser.add_argument("--out_labels", default=OUT_LBL_DEFAULT)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    in_dir = (base / args.in_dir).resolve()
    out_csv = (base / args.out_csv).resolve()
    out_X   = (base / args.out_X).resolve()
    out_y   = (base / args.out_y).resolve()
    out_lbl = (base / args.out_labels).resolve()

    process_all(in_dir, out_csv, out_X, out_y, out_lbl)

if __name__ == "__main__":
    main()
