Scroll down for the English version

# Анализ речевых маркеров когнитивных нарушений 

- Загрузка `.cha` файлов через веб-интерфейс.
- Анализ и отображение результатов двух моделей.
- Просмотр истории всех проведённых анализов.
- Пакетная обработка файлов из командной строки (скрипт `batch_predict.py`).

# Инструкция по запуску
- запустить train.py
- попробовать работоспособность на analyze.py
- открывать веб-приложение через app.py

- либо, скачать файлы формата .pkl с уже загруженными моделями и начинать сразу с запуска app.py


# Использованные библиотеки
- **pandas** — чтение CSV-файлов с метками, создание и обработка DataFrame, сохранение результатов в Excel и отображение таблицы результатов в веб-интерфейсе.
- **numpy** — численные расчёты, работа с массивами признаков, вычисление статистических показателей.
- **scikit-learn** — машинное обучение: Tf-idf векторизация, стандартизация признаков, логистическая регрессия, разделение выборки, метрики качества, кросс-валидация.
- **torch** — фреймворк PyTorch для загрузки предобученной модели sentiment и выполнения анализа тональности.
- **transformers** — библиотека для работы с трансформерами: загрузка токенизатора и модели для анализа тональности русскоязычных текстов.
- **openpyxl** — запись результатов пакетной обработки в формат Excel (используется как движок pandas).
- **Flask** — веб-фреймворк для создания интерфейса загрузки файлов и отображения результатов.
- **os** — работа с файловой системой: создание папок, проверка существования файлов, построение путей.
- **pickle** — сериализация и десериализация обученных моделей, стандартизаторов и Tf-idf векторизатора.
- **csv** — запись результатов единичных предсказаний в файл results.csv в веб-приложении.
- **sklearn** — псевдоним для scikit-learn, используется для импорта конкретных классов (train_test_split, метрики и т.д.).
- **re** — регулярные выражения для обработки текста: извлечение строк участника и пауз из .cha файлов, разбиение текста на предложения, подсчёт слов и предложений.


# Analysis of Speech Markers for Cognitive Impairments

- Upload `.cha` files via a web interface.
- Analyze and display results from two models.
- View history of all conducted analyses.
- Batch processing from the command line (`batch_predict.py` script).

# How to Run
- Run `train.py`
- Test functionality with `analyze.py`
- Launch the web application via `app.py`

- Alternatively, download pre-trained model `.pkl` files and start directly with `app.py`

# Libraries Used
- **pandas** — reading CSV files with labels, creating and manipulating DataFrames, saving results to Excel, and displaying result tables in the web interface.
- **numpy** — numerical computations, feature array handling, calculating statistical measures.
- **scikit-learn** — machine learning: TF-IDF vectorization, feature standardization, logistic regression, train-test split, quality metrics, cross-validation.
- **torch** — PyTorch framework for loading a pre-trained sentiment model and performing sentiment analysis.
- **transformers** — library for working with transformers: loading tokenizer and model for sentiment analysis of Russian texts.
- **openpyxl** — writing batch processing results to Excel format (used as a pandas engine).
- **Flask** — web framework for creating file upload interface and displaying results.
- **os** — file system operations: creating directories, checking file existence, building paths.
- **pickle** — serialization and deserialization of trained models, standardizers, and TF-IDF vectorizer.
- **csv** — writing single prediction results to `results.csv` in the web application.
- **sklearn** — alias for scikit-learn, used to import specific classes (train_test_split, metrics, etc.).
- **re** — regular expressions for text processing: extracting participant utterances and pauses from `.cha` files, splitting text into sentences, counting words and sentences.
