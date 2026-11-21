"""Credit Card Fraud Detection dashboard (Streamlit front-end for a FastAPI model)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =======================
# Configuration
# =======================
BASE_URL = os.getenv(
    "FRAUD_API_URL", "https://fraud-detection-41y3.onrender.com"
).rstrip("/")
API_URL = f"{BASE_URL}/predict"
GITHUB_URL = "https://github.com/apurva119/fraud-detection"

FEATURES: list[str] = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
PCA_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]

REQUEST_TIMEOUT = 60  # seconds; generous for Render free-tier cold starts
HEALTH_TIMEOUT = 20
BATCH_MAX_ROWS = 500
BATCH_WORKERS = 4
HISTORY_LIMIT = 50

# (exclusive upper bound, label)
RISK_LEVELS = [(0.10, "🟢 Low"), (0.50, "🟡 Medium"), (1.01, "🔴 High")]

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Colors for the page itself come from .streamlit/config.toml.
       Custom CSS only styles self-contained components, so text is never
       left unreadable if the viewer's theme (light/dark) differs. */
    .stButton > button[kind="primary"] {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    }
    [data-testid="stMetric"] * { color: #1f2937 !important; }
    .footer { text-align: center; opacity: 0.7; padding: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =======================
# API layer
# =======================
@dataclass(frozen=True)
class PredictionResult:
    probability: float
    is_fraud: bool
    raw: dict[str, Any]


@st.cache_resource
def get_session() -> requests.Session:
    """One shared session with automatic retries for transient gateway errors."""
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=8))
    session.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=8))
    return session


@st.cache_data(ttl=60, show_spinner=False)
def check_api_status() -> str:
    """Return 'online', 'waking' (timeout / cold start) or 'offline'."""
    try:
        r = requests.get(BASE_URL, timeout=HEALTH_TIMEOUT)
        return "online" if r.status_code == 200 else "offline"
    except requests.Timeout:
        return "waking"
    except requests.RequestException:
        return "offline"


def parse_result(data: dict[str, Any]) -> PredictionResult:
    try:
        probability = float(data["fraud_probability"])
        is_fraud = bool(data["prediction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unexpected response format: {data!r}") from exc
    return PredictionResult(probability, is_fraud, data)


def call_predict(payload: dict[str, float]) -> PredictionResult:
    response = get_session().post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return parse_result(response.json())


def describe_error(exc: Exception) -> str:
    if isinstance(exc, requests.Timeout):
        return "The API timed out. It may be waking up; please try again."
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"API error {exc.response.status_code}: {exc.response.text[:300]}"
    if isinstance(exc, requests.ConnectionError):
        return "Could not reach the API. Check your connection or the API status."
    return str(exc)


def predict_batch(
    df: pd.DataFrame, on_progress: Callable[[float], None]
) -> pd.DataFrame:
    """Score every row concurrently; failed rows keep an error message."""
    total = len(df)
    outcomes: dict[Any, tuple[float | None, int | None, str | None]] = {}

    with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as pool:
        futures = {
            pool.submit(call_predict, {f: float(row[f]) for f in FEATURES}): idx
            for idx, row in df.iterrows()
        }
        for done, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]
            try:
                res = future.result()
                outcomes[idx] = (res.probability, int(res.is_fraud), None)
            except Exception as exc:  # noqa: BLE001 - report per-row failures
                outcomes[idx] = (None, None, describe_error(exc))
            on_progress(done / total)

    out = df.copy()
    out["fraud_probability"] = [outcomes[i][0] for i in out.index]
    out["prediction"] = [outcomes[i][1] for i in out.index]
    out["error"] = [outcomes[i][2] for i in out.index]
    return out


# =======================
# State & helpers
# =======================
def init_state() -> None:
    for f in FEATURES:
        st.session_state.setdefault(f, 0.0)
    st.session_state.setdefault("history", [])


def reset_inputs() -> None:
    for f in FEATURES:
        st.session_state[f] = 0.0


def randomize_inputs() -> None:
    rng = np.random.default_rng()
    st.session_state["Time"] = float(rng.integers(0, 172_800))
    st.session_state["Amount"] = round(float(rng.exponential(90)), 2)
    for f in PCA_FEATURES:
        st.session_state[f] = round(float(rng.normal(0, 1.5)), 4)


def risk_label(probability: float) -> str:
    return next(label for bound, label in RISK_LEVELS if probability < bound)


def add_to_history(amount: float, result: PredictionResult) -> None:
    history: list[dict[str, Any]] = st.session_state["history"]
    history.append(
        {
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Amount": amount,
            "Fraud Probability": result.probability,
            "Prediction": "Fraud" if result.is_fraud else "Legitimate",
            "Risk": risk_label(result.probability),
        }
    )
    del history[:-HISTORY_LIMIT]


def render_result(result: PredictionResult) -> None:
    confidence = max(result.probability, 1 - result.probability)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud Probability", f"{result.probability:.2%}")
    c2.metric("Prediction", "🚨 Fraud" if result.is_fraud else "✅ Legitimate")
    c3.metric("Risk Level", risk_label(result.probability))
    c4.metric("Confidence", f"{confidence:.2%}")

    st.progress(min(max(result.probability, 0.0), 1.0))

    if result.is_fraud:
        st.error("🚨 Fraudulent transaction detected!")
    else:
        st.success("✅ Legitimate transaction")

    with st.expander("View raw API response"):
        st.json(result.raw)


def template_csv() -> bytes:
    return pd.DataFrame(columns=FEATURES).to_csv(index=False).encode("utf-8")


# =======================
# Sidebar
# =======================
def render_sidebar() -> None:
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/942/942748.png", width=90)
        st.title("💳 Fraud Detection")

        with st.spinner("Checking API..."):
            status = check_api_status()
        if status == "online":
            st.success("🟢 API online")
        elif status == "waking":
            st.warning("🟡 API is waking up (cold start). Try again shortly.")
        else:
            st.error("🔴 API offline")

        if st.button("🔄 Recheck status", width="stretch"):
            check_api_status.clear()
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
### About
Detect fraudulent credit card transactions using machine learning.

### Tech stack
- 🐍 Python
- ⚡ FastAPI
- 🎈 Streamlit
- 🤖 Scikit-learn
"""
        )
        st.link_button("📄 API documentation", f"{BASE_URL}/docs", width="stretch")
        st.link_button("💻 GitHub repository", GITHUB_URL, width="stretch")


# =======================
# Tabs
# =======================
def tab_single() -> None:
    st.write("Enter the transaction details, then click **Predict transaction**.")

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("🎲 Random values", on_click=randomize_inputs, width="stretch")
    b2.button("♻️ Reset", on_click=reset_inputs, width="stretch")

    t1, t2 = st.columns(2)
    t1.number_input("Time (seconds)", key="Time", min_value=0.0)
    t2.number_input("Amount", key="Amount", min_value=0.0)

    with st.expander("PCA features (V1 – V28)", expanded=True):
        cols = st.columns(4)
        for i, name in enumerate(PCA_FEATURES):
            cols[i // 7].number_input(name, key=name, format="%.4f")

    if st.button("🔍 Predict transaction", type="primary", key="predict_single"):
        payload = {f: float(st.session_state[f]) for f in FEATURES}
        try:
            with st.spinner("Analyzing transaction (first call may take a minute)..."):
                result = call_predict(payload)
        except (requests.RequestException, ValueError) as exc:
            st.error(describe_error(exc))
            return

        add_to_history(payload["Amount"], result)
        st.divider()
        render_result(result)


def tab_batch() -> None:
    st.write(
        f"Upload a CSV with the columns **{', '.join(FEATURES[:3])}, …, Amount** "
        f"(up to {BATCH_MAX_ROWS} rows) to score many transactions at once."
    )
    st.download_button(
        "⬇️ Download CSV template", template_csv(), "template.csv", "text/csv"
    )

    file = st.file_uploader("Transactions CSV", type="csv", key="batch_file")
    if file is None:
        return

    try:
        df = pd.read_csv(file)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read CSV: {exc}")
        return

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    df = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    bad = df.isna().any(axis=1)
    if bad.any():
        st.warning(f"Skipping {int(bad.sum())} row(s) with missing or non-numeric values.")
        df = df[~bad]
    if df.empty:
        st.error("No valid rows to score.")
        return
    if len(df) > BATCH_MAX_ROWS:
        st.warning(f"Only the first {BATCH_MAX_ROWS} rows will be scored.")
        df = df.head(BATCH_MAX_ROWS)

    st.caption(f"{len(df)} valid row(s) ready.")
    with st.expander("Preview"):
        st.dataframe(df.head(10), width="stretch")

    if st.button("🚀 Run batch prediction", type="primary", key="predict_batch"):
        bar = st.progress(0.0, text="Scoring transactions...")
        scored = predict_batch(df, lambda p: bar.progress(p, text="Scoring transactions..."))
        bar.empty()
        st.session_state["batch_result"] = scored

    scored = st.session_state.get("batch_result")
    if scored is None:
        return

    ok = scored[scored["error"].isna()]
    flagged = ok[ok["prediction"] == 1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scored", len(ok))
    m2.metric("Flagged as fraud", len(flagged))
    m3.metric("Fraud rate", f"{(len(flagged) / len(ok)):.2%}" if len(ok) else "n/a")
    m4.metric("Failed rows", len(scored) - len(ok))

    st.dataframe(
        scored.sort_values("fraud_probability", ascending=False, na_position="last"),
        width="stretch",
        column_config={
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud probability", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )
    st.download_button(
        "⬇️ Download results",
        scored.to_csv(index=False).encode("utf-8"),
        "fraud_predictions.csv",
        "text/csv",
    )


def tab_history() -> None:
    history = st.session_state["history"]
    if not history:
        st.info("No predictions yet. Run one from the **Single prediction** tab.")
        return

    df = pd.DataFrame(history)
    st.line_chart(df["Fraud Probability"], height=200)
    st.dataframe(
        df.iloc[::-1],
        width="stretch",
        hide_index=True,
        column_config={
            "Fraud Probability": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )
    if st.button("🗑️ Clear history"):
        st.session_state["history"] = []
        st.rerun()


# =======================
# App
# =======================
init_state()
render_sidebar()

st.title("💳 Credit Card Fraud Detection Dashboard")

st.markdown(
    """
Detect fraudulent credit card transactions with a machine learning model served by **FastAPI**.

- ⚡ FastAPI backend
- 🤖 Scikit-learn model
- 🎈 Streamlit dashboard
- ☁️ Hosted on Render
"""
)

single, batch, history = st.tabs(["🔍 Single prediction", "📦 Batch prediction", "🕑 History"])
with single:
    tab_single()
with batch:
    tab_batch()
with history:
    tab_history()

st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <h4>Built with ❤️ by Apurva Jaiswal</h4>
        FastAPI • Streamlit • Scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)