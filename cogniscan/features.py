import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from .sentiment import sentiment_features # функция, вычисляющая признаки тональности

# вычисляет лингвистические признаки для одного текста
def linguistic_features(text: str):
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences)

    unique_ratio = len(set(words)) / word_count if word_count else 0
    avg_word_len = np.mean([len(w) for w in words]) if word_count else 0
    avg_sentence_len = (
        np.mean([len(re.findall(r"\b\w+\b", s)) for s in sentences])
        if sentence_count
        else 0
    )

    return [word_count, sentence_count, unique_ratio, avg_word_len, avg_sentence_len]

# строит полную матрицу признаков для всех образцов в DataFrame
def build_feature_matrix(df):
    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df["text"]).toarray()

    X_ling = np.array([linguistic_features(t) for t in df["text"]])
    X_pause = df[["pauses", "pause_duration", "utterances"]].values.astype(float)
    X_sent = np.array([sentiment_features(t) for t in df["text"]])

    X = np.hstack([X_tfidf, X_ling, X_pause, X_sent])
    return X, tfidf
