# файл инициализации проекта
# экспортирует основные функции для удобного импорта из пакета

from .data_loader import load_data
from .features import build_feature_matrix
from .model import train_model, save_pipeline, load_pipeline
from .parser import parse_cha_file
from .sentiment import sentiment_features

__all__ = [
    "load_data",
    "build_feature_matrix",
    "train_model",
    "save_pipeline",
    "load_pipeline",
    "parse_cha_file",
    "sentiment_features",
]
