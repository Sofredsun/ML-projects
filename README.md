# ML projects

Мои проекты по машинному обучению: эксперименты, анализ данных, построение и оценка моделей.

## Структура

```
ML-projects/
├── automl/                      # AutoML: мета-признаки, байесовская оптимизация, поиск архитектуры
│   ├── data/                    # данные (не хранятся в репозитории)
│   └── results/                 # результаты экспериментов
├── computer-vision/             # задачи на изображениях
│   └── data/                    # данные (не хранятся в репозитории)
├── tabular/                     # классическое ML на табличных данных
│   └── data/                    # данные (не хранятся в репозитории)
├── time-series/                 # анализ и прогнозирование временных рядов
│   ├── data/                    # небольшие исходные данные
│   └── results/                 # результаты, которые используются в следующих работах
├── environment.yml              # conda-окружение
├── requirements.txt             # список пакетов окружения
└── README.md
```

Все пути в ноутбуках относительные, от папки самого ноутбука. Jupyter по умолчанию запускает ноутбук именно из его папки. В Google Colab нужно заменить путь к данным в начале ноутбука на `/content/drive/MyDrive/...`, в комментарии рядом указано, на какой именно.

## Установка

Одно окружение для всех ноутбуков:

```bash
conda env create -f environment.yml
conda activate ml-projects
python -m ipykernel install --user --name ml-projects --display-name "Python (ml-projects)"
```

После этого в Jupyter (`jupyter notebook`) выбрать ядро **Python (ml-projects)**: Kernel → Change kernel.

Без conda те же пакеты ставятся через pip: `pip install -r requirements.txt`.

PyTorch из pip ставится в CPU-версии. Для GPU нужно поставить сборку с CUDA по инструкции на [pytorch.org](https://pytorch.org/get-started/locally/).

Датасеты с Kaggle скачиваются через `kagglehub`.

## AutoML

* `meta-learning with dataset meta-features.ipynb` — мета-признаки наборов данных OpenML: собственные мета-признаки и PyMFE, проверка инвариантности, мета-классификация (предсказание лучшего алгоритма по мета-признакам). Ожидает наборы в `data/OpenML` (индекс `*.csv` и папку `data/` с файлами `.arff`).
* `bayesian optimization of hyperparameters.ipynb` — подбор гиперпараметров CatBoost на датасете Spotify Songs: случайный поиск, байесовская оптимизация с суррогатом на случайном лесе и на гауссовском процессе, сравнение с Optuna (TPE и random). Результаты сохраняются в `results/bayes_opt`.
* `neural architecture search.ipynb` — поиск архитектуры нейросети для регрессии на датасете elevators (OpenML): Optuna, итерации с LLM, сравнение с MLP-D, RealMLP-TD и CatBoost-TD из статьи [Holzmüller, Grinsztajn, Steinwart. Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data (NeurIPS 2024)](https://arxiv.org/abs/2407.04491) на тесте. Датасет кэшируется в `data/`, результаты сохраняются в `results/`.

## Computer vision

* `detection of brain tumors.ipynb` — нейросеть для обнаружения опухолей мозга на МРТ-снимках. Ожидает `data/images/{train,val}` и `data/labels/{train,val}`.
* `image classification.ipynb` — пайплайн классификации изображений на датасете EuroSAT.

## Tabular

* `movie rating prediction.ipynb` — нейросеть, которая предсказывает рейтинг фильма по его характеристикам. Ожидает `data/mkrf_movies.csv` и `data/mkrf_shows.csv`.
* `termination of a client's cooperation with the bank.ipynb` — модель, которая предсказывает вероятность ухода клиента из банка. Ожидает `data/Bank_data.csv`.

## Time series

Ряды: цены акций Apple, почасовой пассажиропоток метро Нью-Йорка, записи ЭКГ (Fantasia и MIT-BIH), почасовое потребление электроэнергии PJM. Большая часть данных скачивается автоматически через `kagglehub` или по URL. Запись ЭКГ Fantasia скачивается с PhysioNet один раз в `data/fantasia`, при следующих запусках читается с диска.

* `random walk tests and fractal analysis of time series.ipynb` — разведочный анализ, тесты на случайное блуждание, фрактальные и энтропийные характеристики рядов.
* `forecasting with ARIMA and exponential smoothing.ipynb` — прогнозирование моделями AR / ARIMA / SARIMA и экспоненциальным сглаживанием, подбор гиперпараметров, сравнение по MAPE.
* `causality and similarity between time series.ipynb` — связи между рядами: кросс-корреляция, причинность по Грейнджеру, CCM, Dynamic Time Warping.
* `autoregressive models vs neural networks.ipynb` — авторегрессионные модели против нейросетей (LSTM и др.). Сохраняет `results/lab4_results.csv` и `results/lab4_best_hparams.json`.
* `phase space analysis.ipynb` — анализ фазового пространства: корреляционная размерность и размерность вложения, t-SNE траекторий, LSTM с учетом размерности вложения, кластеризация траекторий и граф переходов.
* `local approximation, maximum similarity and SSA forecasting.ipynb` — прогнозирование локальной аппроксимацией, методом максимального подобия и SSA. Сохраняет `results/lab6_results.csv`.
