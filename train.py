import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from cogniscan.data_loader import load_data
from cogniscan.features import build_feature_matrix
from cogniscan.model import train_model

# пути к файлам
CHA_DIR = r"transcripts"
CSV_PATH = r"данные.csv"

# загрузка данных
df = load_data(CHA_DIR, CSV_PATH)
y = df["label"].values

# построение полной матрицы признаков
X_full, tfidf = build_feature_matrix(df)

# обучение полной модели
model_full, scaler_full, metrics_full, X_test_full, y_test_full, y_pred_full, y_proba_full = train_model(X_full, y)

print("\n=== Полная модель ===")
for k, v in metrics_full.items():
    print(f"{k}: {v}")

# Матрица ошибок для полной модели
cm_full = confusion_matrix(y_test_full, y_pred_full)
print("\nConfusion Matrix (full model):")
print(cm_full)

disp_full = ConfusionMatrixDisplay(confusion_matrix=cm_full, display_labels=["Здоров", "MCI"])
disp_full.plot()
plt.title("Confusion Matrix - Полная модель")
plt.savefig("cm_full.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------- Модель только тональности ----------
X_sent_only = X_full[:, -6:]  # последние 6 признаков — тональность
model_sent, scaler_sent, metrics_sent, X_test_sent, y_test_sent, y_pred_sent, y_proba_sent = train_model(X_sent_only, y)

print("\n=== Модель только тональности ===")
for k, v in metrics_sent.items():
    print(f"{k}: {v}")

# Матрица ошибок для модели тональности
cm_sent = confusion_matrix(y_test_sent, y_pred_sent)
print("\nConfusion Matrix (sentiment only):")
print(cm_sent)

disp_sent = ConfusionMatrixDisplay(confusion_matrix=cm_sent, display_labels=["Здоров", "MCI"])
disp_sent.plot()
plt.title("Confusion Matrix - Модель тональности")
plt.savefig("cm_sent.png", dpi=150, bbox_inches="tight")
plt.show()

# сохранение моделей
with open("full_model.pkl", "wb") as f:
    pickle.dump({"model": model_full, "scaler": scaler_full, "tfidf": tfidf}, f)

with open("sent_model.pkl", "wb") as f:
    pickle.dump({"model": model_sent, "scaler": scaler_sent}, f)

print("\nМодели сохранены: full_model.pkl и sent_model.pkl")