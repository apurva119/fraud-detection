# 💳 Credit Card Fraud Detection API

A Machine Learning-powered REST API built with **FastAPI** that detects fraudulent credit card transactions in real time.

## 🚀 Live Demo

- **API:** https://fraud-detection-41y3.onrender.com/
- **Swagger Docs:** https://fraud-detection-41y3.onrender.com/docs

---

## 📌 Project Overview

This project uses a trained Machine Learning model to classify credit card transactions as either:

- ✅ Legitimate Transaction
- 🚨 Fraudulent Transaction

The model is exposed through a FastAPI REST API and deployed on Render for real-time inference.

---

## 🛠 Tech Stack

- Python 3
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Uvicorn
- Render
- Git & GitHub

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
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── 01_eda.ipynb
│
└── data/
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/apurva119/fraud-detection.git
cd fraud-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Windows**

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoint

### POST `/predict`

Predict whether a transaction is fraudulent.

### Sample Request

```json
{
  "Time": 0,
  "V1": 0,
  "V2": 0,
  "V3": 0,
  "V4": 0,
  "V5": 0,
  "V6": 0,
  "V7": 0,
  "V8": 0,
  "V9": 0,
  "V10": 0,
  "V11": 0,
  "V12": 0,
  "V13": 0,
  "V14": 0,
  "V15": 0,
  "V16": 0,
  "V17": 0,
  "V18": 0,
  "V19": 0,
  "V20": 0,
  "V21": 0,
  "V22": 0,
  "V23": 0,
  "V24": 0,
  "V25": 0,
  "V26": 0,
  "V27": 0,
  "V28": 0,
  "Amount": 0
}
```

### Sample Response

```json
{
  "prediction": 0,
  "fraud_probability": 0.0002
}
```

---

## 📊 Features

- REST API with FastAPI
- Machine Learning fraud detection
- Interactive Swagger documentation
- Render cloud deployment
- GitHub version control

---

## 🔮 Future Improvements

- Streamlit dashboard
- Docker support
- GitHub Actions (CI/CD)
- Unit tests with Pytest
- Model performance dashboard
- Authentication and API keys

---

## 👩‍💻 Author

**Apurva**

GitHub: https://github.com/apurva119
