# 💳 Credit Card Fraud Detection System

A machine learning system built with **FastAPI** and **Streamlit** that scores credit card transactions for fraud. A tuned Random Forest model is served through a REST API, and a Streamlit dashboard lets anyone enter transaction details and get a prediction with a fraud probability.

## 🚀 Live Demo

- **Streamlit Dashboard:** https://fraud-detection-uiia.streamlit.app/
- **API:** https://fraud-detection-41y3.onrender.com/
- **Swagger Docs:** https://fraud-detection-41y3.onrender.com/docs

> ⏳ The API runs on Render's free tier and sleeps when idle. The first request after a quiet period can take **30–60 seconds** to wake up.

---

## 📌 Project Overview

The model classifies each transaction as:

- ✅ Legitimate
- 🚨 Fraudulent

and returns a **fraud probability** with every prediction. The model is exposed through a FastAPI REST API (Render) and consumed by a Streamlit dashboard (Streamlit Community Cloud).

---

## 📈 Model & Results

- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), 284,807 transactions, of which only 0.17% are fraud. Download `creditcard.csv` and place it in `data/` to run the notebook.
- **Models compared:** Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Final model:** Random Forest (`max_depth=8`, `n_estimators=50`), tuned with 5-fold GridSearchCV
- **Split:** 80/20 stratified; the scaler is fit on training data only, and SMOTE (where used) is applied to the training split only

### Test-set results (fraud class, default 0.5 threshold)

| Model | Precision | Recall | Missed frauds | False alarms |
|---|---|---|---|---|
| Random Forest (final) | 0.96 | 0.79 | 21 | 3 |
| Logistic Regression + SMOTE | 0.06 | 0.92 | 8 | 1,467 |

The final model catches 77 of 98 frauds with 3 false alarms among 56,864 legitimate transactions. SMOTE and `class_weight="balanced"` were evaluated on a Logistic Regression baseline: both raised recall but flooded the system with false alarms, which is why they are not used in the final model.

ROC-AUC 0.970 and Average Precision 0.867 on the holdout test set (same model, same run). A random ranker would score about 0.0017 on Average Precision, the fraud rate.

### Cost-based decision threshold

Accuracy is a poor guide on data this imbalanced, so the decision threshold was chosen by **expected cost** instead of F1.

- **Assumed costs (illustrative):** ₹10,000 per missed fraud, ₹200 per false alarm
- **Break-even threshold:** `A / (A + F) = 200 / 10,200 ≈ 0.0196`
- **Selection method:** out-of-fold predictions on the training set (5-fold), sweeping thresholds and picking the midpoint of the flat low-cost band (0.013 to 0.218, midpoint ≈ 0.116), *before* touching the test set
- **Verification:** evaluated once on the held-out test set

| Threshold | Missed | False alarms | Test cost |
|---|---|---|---|
| 0.5 (default) | 21 | 3 | ₹2,10,600 |
| 0.0196 (theoretical break-even) | 10 | 160 | ₹1,32,000 |
| 0.116 (selected) | 13 | 29 | ₹1,35,800 |

Both lower thresholds cut expected cost by roughly 35–37% versus the default 0.5. The gap between 0.0196 and 0.116 (₹3,800) is smaller than the cost of a single missed fraud, so it is within noise on a test set with only 98 frauds.

**Status:** the live API uses the selected cost-sensitive threshold of **0.116**.

---

## 🛠 Tech Stack

- Python 3, FastAPI, Uvicorn
- Scikit-learn, imbalanced-learn (SMOTE), XGBoost (comparison experiments)
- Pandas, NumPy, Joblib
- Streamlit
- Render, Streamlit Community Cloud
- Git & GitHub

---

## 📂 Project Structure

```
fraud-detection/
│
├── app.py                  # FastAPI service
├── requirements.txt
├── README.md
├── LICENSE
│
├── models/
│   ├── fraud_model.pkl     # tuned Random Forest
│   └── scaler.pkl          # scaler fit on training data
│
├── streamlit_app/
│   ├── app.py              # dashboard
│   └── requirements.txt
│
├── notebooks/
│   └── 01_eda.ipynb        # EDA, model comparison, threshold analysis
│
└── data/                   # place creditcard.csv here (not tracked)
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/apurva119/fraud-detection.git
cd fraud-detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

**Windows**

```powershell
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

Install dependencies and run the API:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000/docs

---

## 📡 API Endpoint

### POST `/predict`

Predicts whether a transaction is fraudulent. The request takes `Time`, `V1`–`V28` and `Amount`.

### Sample Request

```json
{
  "Time": 0,
  "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0, "V7": 0,
  "V8": 0, "V9": 0, "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V14": 0,
  "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0, "V21": 0,
  "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0,
  "Amount": 0
}
```

### Sample Response

```json
{
  "prediction": 0,
  "fraud_probability": 0.0002,
  "threshold": 0.116
}
```

---

## ✨ Features

- Interactive Streamlit dashboard
- Fraud probability returned with every prediction
- REST API with interactive Swagger docs
- Imbalance-aware evaluation (precision, recall, cost) instead of accuracy
- Cloud deployment on Render and Streamlit Community Cloud

---

## ⚠ Limitations

- Hyperparameters were tuned for F1, not for cost. Under the cost view, an untuned Random Forest was marginally cheaper on the test set, within noise.
- The costs above are assumptions, not real bank figures. In practice, the cost of a missed fraud scales with the transaction amount.
- Random Forest probabilities are not guaranteed to be calibrated, so a probability should be read as a risk score rather than an exact likelihood.
- The test set contains only 98 fraud cases, so recall and cost figures carry noticeable sampling noise.

---

## 🔮 Future Improvements

- Unit tests with Pytest
- GitHub Actions (CI/CD)
- Probability calibration and an amount-weighted cost function
- Model performance dashboard
- Docker support
- Authentication and API keys

---

## 👤 Author

**Apurva Jaiswal**

GitHub: https://github.com/apurva119