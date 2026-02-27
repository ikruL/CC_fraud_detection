<div align="center">

# Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-brightgreen?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**End-to-end Machine Learning pipeline for detecting fraudulent credit card transactions**  
Built with focus on **imbalanced data handling**, **production-grade practices**, and **deployable API**.

</div>

## API DEMO

![API Demo](/images/demo.gif)

## Project Highlights

- Complete **ML pipeline**: from raw data to live prediction API
- Effective handling of **extreme class imbalance** (~0.17% fraud cases)
- **Threshold tuning** for optimal precision-recall trade-off in production
- **Stratified cross-validation** + model comparison
- **FastAPI** inference service
- **Explainability** support feature importance

Best model: **XGBoost**  
Achieved excellent fraud-class performance after threshold optimization.

## Dataset

**Credit Card Fraud Detection** (Anonymized transactions – September 2013)  
→ **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Features**:

- `Time` – seconds since first transaction
- `V1`–`V28` – PCA-transformed features
- `Amount` – transaction amount
- `Class` – target (0 = normal, 1 = fraud)

**Key statistics**:

- Total transactions: **284,807**
- Fraud cases: **492** (~**0.172%** – highly imbalanced)

### Download the dataset

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Sign in / create a free Kaggle account
3. Click **Download** (file: `creditcard.csv` ~150 MB)
4. Place the file in the `data/` folder (create if needed)
   > **Note**: The dataset is **not** included in this repository due to size and licensing.

## Project Structure

```text
fraud_detection/
├── data/                    # Put creditcard.csv here
├── notebooks/
│   └── 01_eda.ipynb         # EDA
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluation.py
│   ├── interpret.py
│   └── api.py
├── models/
│   ├── best_model.pkl
│   ├── best_scaler.pkl
│   └── best_threshold.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Installation

```
# Clone repository
git clone https://github.com/YOUR-USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate    # Linux / macOS
# or
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

2. Train the model

```
cd src
python train.py
```

3. Run API

```
uvicorn src.api:app --reload --port 8000
```

→ Open: http://127.0.0.1:8000/docs

## Model Performance (Best – XGBoost)

| Metric            | Value |
| ----------------- | ----- |
| Precision (Fraud) | 0.96  |
| Recall (Fraud)    | 0.78  |
| PR-AUC            | 0.84  |

## License

MIT License – feel free to use this project for learning purposes.
