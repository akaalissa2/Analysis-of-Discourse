import os
import pickle
import numpy as np
from cogniscan.data_loader import load_data  # загрузка данных из CHA и CSV
from cogniscan.features import build_feature_matrix  # построение полной матрицы признаков
from cogniscan.model import train_model  # обучение модели регрессии

# пути к файлам для тренировки
CHA_DIR = r"transcripts"
CSV_PATH = r"данные.csv"

# загрузка данных
df = load_data(CHA_DIR, CSV_PATH)
y = df["label"].values

# построение полной матрицы признаков
X_full, tfidf = build_feature_matrix(df)

model_full, scaler_full, metrics_full = train_model(X_full, y)

for k, v in metrics_full.items():
    print(f"{k}: {v}")

X_sent_only = X_full[:, -6:]  # последние 6 признаков — тональность
model_sent, scaler_sent, metrics_sent = train_model(X_sent_only, y)

for k, v in metrics_sent.items():
    print(f"{k}: {v}")

# сохранение моделей вместе со стандартизацией
with open("full_model.pkl", "wb") as f:
    pickle.dump({"model": model_full, "scaler": scaler_full, "tfidf": tfidf}, f)

with open("sent_model.pkl", "wb") as f:
    pickle.dump({"model": model_sent, "scaler": scaler_sent}, f)

print("\nМодели сохранены: full_model.pkl и sent_model.pkl")
