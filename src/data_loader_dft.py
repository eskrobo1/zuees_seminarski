import os
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def parse_metadata_from_filename(filename: str):
    """
    Parsira Z_fault i labelu iz imena fajla.
    Primjer: Z_fault_0,5_line_69_50%.txt -> zfault_val=0.5, label='line_69_sec2'
    """
    base = os.path.basename(filename).replace(".txt", "")
    parts = [p.strip() for p in base.split("_")]

    zfault_str = parts[2].replace(",", ".")
    zfault_val = float(zfault_str)

    line = parts[4]
    pct = float(parts[5].replace("%", "")) / 100.0
    sections = 3
    for i in range(sections):
        if pct <= (i + 1) / sections:
            sec = i + 1
            break
    else:
        sec = sections

    label = f"line_{line}_sec{sec}"
    return zfault_val, label


def clean_file_content(filepath: str) -> str:
    """Otvori fajl, ukloni header/footer linije i vrati čist tekst."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        if "SOLUTION_DATA" in line:
            continue
        if "---" in line:
            continue
        if "All OK" in line:
            continue
        if not line.strip():
            continue
        clean_lines.append(line)

    return "".join(clean_lines)


def compute_dft_features(signal: np.ndarray, fs: float = 1000.0):
    """
    Računa DFT magnitudu signala.
    signal: (T,) ili (T, F) -> ovdje uzimamo prosjek preko feature-a.
    Uklanja se DC komponenta (srednja vrijednost).
    """
    if signal.ndim == 2:  
        sig = signal.mean(axis=1)  # prosjek svih feature-a
    else:
        sig = signal

    # ukloni DC komponentu
    sig = sig - np.mean(sig)

    N = len(sig)
    freqs = np.fft.rfftfreq(N, 1/fs)
    spectrum = np.abs(np.fft.rfft(sig)) / N
    return spectrum, freqs


def load_dft_data(folder="simulation_results", timesteps=651, fs=1000.0):
    """
    Učitaj podatke i vrati DFT spektre umjesto vremenskih signala.
    """
    X, y, Z = [], [], []

    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(folder, fname)

        zfault_val, label = parse_metadata_from_filename(fname)

        raw_txt = clean_file_content(fpath)
        df = pd.read_csv(io.StringIO(raw_txt), sep=r"\s+", engine="python")

        if "t[s]" not in df.columns:
            print(f"Preskačem {fname}, nema kolone t[s]")
            continue

        df = df[df["t[s]"] >= 35]
        features = df.drop(columns=["t[s]"]).values

        # normalizacija dužine
        if features.shape[0] < timesteps:
            pad_len = timesteps - features.shape[0]
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
        else:
            features = features[:timesteps, :]

        # DFT
        spectrum, _ = compute_dft_features(features, fs=fs)

        X.append(spectrum)
        y.append(label)
        Z.append(zfault_val)

    X = np.array(X)  # (samples, freq_bins)
    y = np.array(y)
    Z = np.array(Z, dtype=np.float32)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X, Z, y_encoded, encoder
