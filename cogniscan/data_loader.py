import os
import pandas as pd
from .parser import parse_cha_file # импорт парсера


def load_data(cha_dir: str, csv_path: str):
    df_labels = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    df_labels.columns = df_labels.columns.str.strip()

    df_labels = df_labels.dropna(subset=["label"])
    df_labels["label"] = pd.to_numeric(df_labels["label"], errors="coerce")
    df_labels = df_labels.dropna(subset=["label"])
    df_labels["label"] = df_labels["label"].astype(int)

    records = []

    for _, row in df_labels.iterrows():
        fname = str(row["filename"]).strip()
        if not fname.endswith(".cha"):
            fname += ".cha"

        path = os.path.join(cha_dir, fname)
        if not os.path.exists(path):
            continue

        parsed = parse_cha_file(path)

        records.append({
            "filename": fname,
            "text": parsed["text"],
            "label": row["label"],
            "pauses": parsed["pauses"],
            "pause_duration": parsed["pause_duration"],
            "utterances": parsed["utterances"],
        })

    return pd.DataFrame(records)
