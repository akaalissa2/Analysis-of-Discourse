import os
import pickle
import csv
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for
from cogniscan.parser import parse_cha_file  # созданный парсер файлов
from cogniscan.features import linguistic_features, sentiment_features  # излечение признаков

# выбираем папки
UPLOAD_FOLDER = "uploads"
RESULTS_FILE = "results.csv"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# загрузка моделек
with open("full_model.pkl", "rb") as f:
    data_full = pickle.load(f)
model_full = data_full["model"]  # полная модель Tf‑idf + лингвистика + паузы + тональность
scaler_full = data_full["scaler"]  # стандартизатор для полной модели
tfidf_full = data_full["tfidf"]  # Tf‑idf векторизатор

with open("sent_model.pkl", "rb") as f:
    data_sent = pickle.load(f)
model_sent = data_sent["model"]  # модель только на тональности
scaler_sent = data_sent["scaler"]  # стандартизатор ее


# предсказываем наличие диагноза
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

    full_model_res = {
        "pred": "MCI" if pred_full == 1 else "Здоров",
        "prob_mci": f"{proba_full[1]:.3f}",
        "prob_healthy": f"{proba_full[0]:.3f}"
    }

    sent_model_res = {
        "pred": "MCI" if pred_sent == 1 else "Здоров",
        "prob_mci": f"{proba_sent[1]:.3f}",
        "prob_healthy": f"{proba_sent[0]:.3f}"
    }

    return parsed, full_model_res, sent_model_res


# записываем результаты
def save_result(filename, parsed, full_model_res, sent_model_res):
    short_text = parsed["text"][:100] + "..." if len(parsed["text"]) > 100 else parsed["text"]
    headers = ["filename", "short_text",
               "full_pred", "full_prob_mci", "full_prob_healthy",
               "sent_pred", "sent_prob_mci", "sent_prob_healthy"]
    row = [
        filename, short_text,
        full_model_res["pred"], full_model_res["prob_mci"], full_model_res["prob_healthy"],
        sent_model_res["pred"], sent_model_res["prob_mci"], sent_model_res["prob_healthy"]
    ]
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "file" not in request.files:
            result = "Файл не выбран"
        else:
            file = request.files["file"]
            if file.filename == "":
                result = "Файл не выбран"
            else:
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)
                parsed, full_model_res, sent_model_res = predict_file(filepath)
                save_result(file.filename, parsed, full_model_res, sent_model_res)
                result = {"full_model": full_model_res, "sentiment_model": sent_model_res}

    return render_template("index.html", result=result, results_url=url_for("view_results"))


@app.route("/results")
def view_results():
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        df["short_text"] = df["short_text"].str.slice(0, 100)
        results_list = df.to_dict(orient="records")
    else:
        results_list = []
    return render_template("results.html", results=results_list)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
