import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# используем предобученную модель для анализа тональности
MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"

# глобальные переменные для загрузки модели (загружается один раз при первом вызове)
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


# возвращает тональность одного предложения, от -1 до 1
def sentiment_score(text: str) -> float:
    _load()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    neg, _, pos = probs.tolist()  # 0 - нейтрально, -1 - негатив, 1 -позитив
    return pos - neg


# получение тональности и статистистики: ср, стд, мин, макс, негатив, позитив
def sentiment_features(text: str):
    sentences = re.split(r"[.!?]+", text)
    scores = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        try:
            scores.append(sentiment_score(s))
        except:
            continue

    if not scores:
        return [0, 0, 0, 0, 0, 0]

    scores = np.array(scores)

    mean = scores.mean()
    std = scores.std()
    min_val = scores.min()
    max_val = scores.max()
    neg_ratio = (scores < -0.2).mean()
    pos_ratio = (scores > 0.2).mean()

    return [mean, std, min_val, max_val, neg_ratio, pos_ratio]
