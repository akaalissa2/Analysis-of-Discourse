import os
import pickle
import numpy as np
from cogniscan.parser import parse_cha_file
from cogniscan.features import linguistic_features
from cogniscan.sentiment import sentiment_features

# загрузка моделей
with open("full_model.pkl", "rb") as f:
    data_full = pickle.load(f)
model_full = data_full["model"]
scaler_full = data_full["scaler"]
tfidf_full = data_full["tfidf"]

with open("sent_model.pkl", "rb") as f:
    data_sent = pickle.load(f)
model_sent = data_sent["model"]
scaler_sent = data_sent["scaler"]


def predict_file(path):
    parsed = parse_cha_file(path)
    text = parsed["text"]

    X_tfidf = tfidf_full.transform([text]).toarray()
    X_ling = np.array([linguistic_features(text)])
    X_pause = np.array([[parsed["pauses"], parsed["pause_duration"], parsed["utterances"]]])
    X_sent = np.array([sentiment_features(text)])

    # полная модель
    X_full = np.hstack([X_tfidf, X_ling, X_pause, X_sent])
    X_full_scaled = scaler_full.transform(X_full)
    pred_full = model_full.predict(X_full_scaled)[0]
    proba_full = model_full.predict_proba(X_full_scaled)[0]

    # только тональность
    X_sent_only_scaled = scaler_sent.transform(X_sent)
    pred_sent = model_sent.predict(X_sent_only_scaled)[0]
    proba_sent = model_sent.predict_proba(X_sent_only_scaled)[0]

    return (pred_full, proba_full), (pred_sent, proba_sent), parsed


# для интерактива
while True:
    path = input("Введите путь к .cha файлу (exit для выхода): ").strip()
    if path.lower() == "exit":
        break
    if not os.path.exists(path):
        print("Файл не найден\n")
        continue

    (pred_full, proba_full), (pred_sent, proba_sent), parsed = predict_file(path)

    print("\nРезультаты")
    print("1. Полная модель:")
    print(f"  Предсказание: {'MCI' if pred_full == 1 else 'Здоров'}")
    print(f"  Вероятность MCI: {proba_full[1]:.3f}")
    print(f"  Вероятность здоров: {proba_full[0]:.3f}")

    print("\n2. Модель только тональность:")
    print(f"  Предсказание: {'MCI' if pred_sent == 1 else 'Здоров'}")
    print(f"  Вероятность MCI: {proba_sent[1]:.3f}")
    print(f"  Вероятность здоров: {proba_sent[0]:.3f}")
    print("-" * 40)
