# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Overview

This project detects fraudulent credit card transactions using Machine Learning models and exposes predictions through a FastAPI REST API.

The project compares multiple classification algorithms and selects the best-performing model based on evaluation metrics.

---

## 🚀 Features

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Scaling
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Hyperparameter Tuning
- FastAPI REST API
- Swagger UI Documentation

---

## 📂 Project Structure

```
fraud-detection/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── data/
│   └── creditcard.csv (not included)
│
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── api/
├── dashboard/
├── database/
├── src/
└── tests/
```

---

## 📊 Dataset

Dataset: **Credit Card Fraud Detection**

Download from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the downloaded file in:

```
data/creditcard.csv
```

> The dataset is not included in this repository because it exceeds GitHub's file size recommendations.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Joblib
- Matplotlib
- Seaborn

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|---------:|----------:|-------:|---------:|--------:|
| Logistic Regression | 99.92% | 90% | 79% | 0.84 | 0.960 |
| Random Forest | **99.96%** | **96%** | **79%** | **0.87** | **0.963** |
| XGBoost | 99.95% | 90% | 79% | 0.84 | 0.961 |

**Selected Model:** Random Forest

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/apurva119/fraud-detection.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app:app --reload
```

Open your browser:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### POST `/predict`

Returns:

```json
{
  "prediction": 0,
  "fraud_probability": 0.0002
}
```

---

## 📌 Future Improvements

- Deploy on Render
- Docker support
- Streamlit Dashboard
- CI/CD Pipeline
- Model Monitoring

---

## 👨‍💻 Author

**Apurva Jaiswal**

GitHub: https://github.com/apurva119