import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# обучает модель
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cv_f1": cross_val_score(
            model, X_train, y_train, cv=5, scoring="f1"
        ).mean(),
    }

    return model, scaler, metrics


def save_pipeline(model, scaler, tfidf, path="cogniscan_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "tfidf": tfidf}, f)


def load_pipeline(path="cogniscan_model.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["tfidf"]
