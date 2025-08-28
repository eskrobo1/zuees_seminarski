import os
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def parse_label_from_filename(filename: str) -> str:
    """
    Parsira labelu iz imena fajla.
    Primjer: Z_fault_5_line_69_50%.txt -> 'line_69_50%'
    """
    base = os.path.basename(filename)
    parts = base.replace(".txt", "").split("_")
    line = parts[3]   # "69"
    loc = parts[4]    # "50%"
    return f"line_{line}_{loc}"


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


def load_data(folder="simulation_results"):
    X, y = [], []

    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(folder, fname)

        # očisti fajl i učitaj
        raw_txt = clean_file_content(fpath)
        df = pd.read_csv(io.StringIO(raw_txt), sep=r"\s+")

        # rezanje vremena (samo 35s do kraja)
        if "t[s]" in df.columns:
            df = df[df["t[s]"] >= 35]

        print(f"Učitano: {fname}, shape={df.shape}")

        # makni vremensku kolonu
        if "t[s]" in df.columns:
            features = df.drop(columns=["t[s]"]).values.flatten()
        else:
            features = df.values.flatten()

        X.append(features)
        y.append(parse_label_from_filename(fname))

    X = np.array(X)
    y = np.array(y)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X, y_encoded, encoder
