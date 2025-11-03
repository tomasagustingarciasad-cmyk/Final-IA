# -*- coding: utf-8 -*-
import argparse, subprocess, shutil, os
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, resample_poly
from fractions import Fraction
import librosa
HAVE_LIBROSA = True
# ---------- Paths y parámetros ----------
BASE_DIR = Path(__file__).resolve().parent

TARGET_SR = 16000
HP_HZ = 100.0
LP_HZ = 5000.0
USE_NOTCH_50HZ = True
NOTCH_Q = 30.0
TARGET_DUR = 1.20
RMS_TARGET_DBFS = -20.0
SIL_MARGIN = 0.05
RMS_WIN_MS = 20
RMS_HOP_MS = 10
TRIM_REL_THR = 0.10
MIN_KEEP_MS = 250

# ---------- DSP utils ----------
def to_mono(x): return x if x.ndim == 1 else x.mean(axis=1)

def rational_resample(x, sr_in, sr_out):
    if sr_in == sr_out: return x
    frac = Fraction(sr_out, sr_in).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator)

def bandpass_filter(x, sr, hp_hz, lp_hz):
    nyq = 0.5*sr
    low = max(hp_hz/nyq, 1e-6); high = min(lp_hz/nyq, 0.999)
    if high <= low: return x
    sos = butter(4, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, x)

def apply_notch(x, sr, f0, q):
    b, a = iirnotch(w0=f0, Q=q, fs=sr)
    return filtfilt(b, a, x)

def moving_rms(x, sr, win_ms=20, hop_ms=10):
    win = max(1, int(sr*win_ms/1000.0))
    hop = max(1, int(sr*hop_ms/1000.0))
    kernel = np.ones(win, dtype=np.float64)/float(win)
    pow_env = np.convolve(x.astype(np.float64)**2, kernel, mode='same')
    rms = np.sqrt(pow_env + 1e-12)
    idx = np.arange(0, len(rms), hop)
    return rms[idx], idx, hop, win

def trim_silence(x, sr, rel_thr=TRIM_REL_THR, margin_s=SIL_MARGIN):
    rms, idx, hop, win = moving_rms(x, sr, RMS_WIN_MS, RMS_HOP_MS)
    rmax = float(np.max(rms)) if rms.size else 0.0
    thr = max(0.02, rel_thr*rmax)
    voiced = np.where(rms > thr)[0]
    if voiced.size == 0: return x
    start = int(max(0, idx[voiced[0]] - int(margin_s*sr)))
    end   = int(min(len(x), idx[voiced[-1]] + win + int(margin_s*sr)))
    if (end - start) < int(MIN_KEEP_MS/1000.0 * sr): return x
    return x[start:end]

def rms_level(x): return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))

def normalize_rms(x, target_dbfs=-20.0):
    target_lin = 10.0 ** (target_dbfs/20.0)
    cur = rms_level(x)
    if cur <= 1e-8: return x
    g = target_lin/cur
    peak = float(np.max(np.abs(x))*g)
    if peak > 0.99:
        g = 0.99/(float(np.max(np.abs(x))) + 1e-12)
    return (x*g).astype(np.float32)

def fix_duration(x, sr, target_sec=1.20):
    target_n = int(round(sr*target_sec)); n = len(x)
    if n == target_n: return x
    if n > target_n:
        start = (n - target_n)//2
        return x[start:start+target_n]
    pad = target_n - n; left = pad//2; right = pad - left
    return np.pad(x, (left, right), mode='constant')

def write_wav_pcm16(path, x, sr):
    x = np.clip(x, -1.0, 1.0)
    sf.write(file=path, data=x, samplerate=sr, subtype='PCM_16', format='WAV')

# ---------- FFmpeg helpers ----------
def resolve_ffmpeg(explicit_path: str|None):
    candidates = []
    if explicit_path: candidates.append(explicit_path)
    env = os.environ.get("FFMPEG_BIN")
    if env: candidates.append(env)
    local = BASE_DIR / "tools" / "ffmpeg" / "ffmpeg.exe"
    if local.exists(): candidates.append(str(local))
    which = shutil.which("ffmpeg")
    if which: candidates.append(which)
    for c in candidates:
        p = Path(c)
        if p.exists(): return str(p)
    return None

def smart_read_any(path: Path, ffmpeg_bin: str|None):
    # 1) soundfile
    try:
        y, sr = sf.read(str(path), dtype='float32', always_2d=False)
        return to_mono(np.asarray(y, dtype=np.float32)), int(sr)
    except Exception:
        pass
    # 2) librosa/audioread
    if HAVE_LIBROSA:
        try:
            y, sr = librosa.load(str(path), sr=None, mono=False)
            y = to_mono(np.asarray(y, dtype=np.float32))
            return y, int(sr)
        except Exception:
            pass
    # 3) ffmpeg CLI
    ff = resolve_ffmpeg(ffmpeg_bin)
    if ff is None:
        raise RuntimeError("No se pudo leer el audio. FFmpeg no está disponible. "
                           "Instalalo o pasá --ffmpeg <ruta-a-ffmpeg.exe>.")
    try:
        cmd = [ff, "-v", "error", "-i", str(path),
               "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
               "-ar", str(TARGET_SR), "pipe:1"]
        raw = subprocess.check_output(cmd)
        y = np.frombuffer(raw, dtype=np.float32)
        return y, TARGET_SR
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg no pudo decodificar {path}: {e}")

# ---------- Proceso por archivo ----------
def process_one(path_in: Path, path_out: Path, args):
    x, sr = smart_read_any(path_in, args.ffmpeg)

    if sr != TARGET_SR:
        x = rational_resample(x, sr, TARGET_SR); sr = TARGET_SR

    x = bandpass_filter(x, sr, args.hp_hz, args.lp_hz)
    if args.notch_50: x = apply_notch(x, sr, 50.0, NOTCH_Q)
    x = trim_silence(x, sr, rel_thr=args.trim_rel_thr, margin_s=args.trim_margin)
    x = normalize_rms(x, target_dbfs=args.rms_dbfs)
    x = fix_duration(x, sr, target_sec=args.target_dur)

    out = path_out.with_suffix(".wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav_pcm16(str(out), x, sr)

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Preprocesado/normalización multi-formato.")
    parser.add_argument("--in_dir",  default=r"base_datos\Audio")
    parser.add_argument("--out_dir", default=r"base_datos\Audio_norm")
    parser.add_argument("--ffmpeg",  default=None, help="Ruta a ffmpeg.exe (opcional)")
    parser.add_argument("--hp_hz", type=float, default=HP_HZ)
    parser.add_argument("--lp_hz", type=float, default=LP_HZ)
    parser.add_argument("--notch_50", action="store_true", default=USE_NOTCH_50HZ)
    parser.add_argument("--trim_rel_thr", type=float, default=TRIM_REL_THR)
    parser.add_argument("--trim_margin", type=float, default=SIL_MARGIN)
    parser.add_argument("--rms_dbfs", type=float, default=RMS_TARGET_DBFS)
    parser.add_argument("--target_dur", type=float, default=TARGET_DUR)
    parser.add_argument("--exts", nargs="+", default=[
        ".wav",".WAV",".ogg",".OGG",".mp3",".MP3",".m4a",".M4A",
        ".mp4",".MP4",".webm",".WEBM",".flac",".FLAC",".aac",".AAC",".wma",".WMA"
    ])
    args = parser.parse_args()

    in_dir = (BASE_DIR / args.in_dir).resolve()
    out_dir = (BASE_DIR / args.out_dir).resolve()
    classes = ["Contar","Proporcion","Salir"]

    total_in = total_out = 0
    for sub in classes:
        src = in_dir / sub
        if not src.exists(): continue
        dst = out_dir / sub
        for f in sorted(src.iterdir()):
            if not f.is_file() or f.suffix not in args.exts: continue
            total_in += 1
            out_path = dst / f.stem
            rel_out = (dst / (f.stem + ".wav")).relative_to(out_dir).as_posix()
            try:
                process_one(f, out_path, args)
                total_out += 1
                print(f"✓ {sub}\\{f.name} -> {rel_out}")
            except Exception as e:
                print(f"✗ ERROR en {f}: {e}")

    print(f"\nListo. Procesados {total_out}/{total_in}. Salida: {out_dir}")

if __name__ == "__main__":
    main()
