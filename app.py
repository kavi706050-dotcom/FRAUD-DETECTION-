# ============================================================
# FRAUD SHIELD AI | V2.2
# Real-Time Fraud Detection & Intelligence Platform
# ============================================================

import os
import random
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="Fraud Shield AI | V2.2",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detection_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "creditcard.csv")

# ----------------------------- Theme -----------------------------
# Dashboard metric cards use native Streamlit components (st.metric).
# This avoids HTML card strings accidentally appearing as code/text.
st.markdown(
    """
    <style>
    .stApp {
        background:
          radial-gradient(circle at 8% 8%, rgba(130, 0, 38, .18), transparent 35%),
          radial-gradient(circle at 92% 90%, rgba(72, 0, 125, .15), transparent 36%),
          #080910;
        color: #f4f4f5;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #100b13, #090910);
        border-right: 1px solid rgba(255, 23, 68, .35);
    }
    .hero-title {
        font-size: clamp(28px, 4vw, 44px);
        font-weight: 900;
        letter-spacing: 1.5px;
        background: linear-gradient(90deg, #fff, #ff1744, #ff7895, #fff);
        background-size: 250% 250%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleShift 7s ease infinite;
        margin-bottom: 2px;
    }
    @keyframes titleShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .subtitle { color: #a1a1aa; margin: 4px 0 22px 0; }
    .release-pill {
        display: inline-block;
        color: #ff4167;
        background: rgba(255, 23, 68, .10);
        border: 1px solid #ff1744;
        border-radius: 10px;
        padding: 7px 13px;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1px;
        animation: pillGlow 3s ease-in-out infinite;
    }
    @keyframes pillGlow {
        0%, 100% { box-shadow: 0 0 4px rgba(255, 23, 68, .12); }
        50% { box-shadow: 0 0 16px rgba(255, 23, 68, .38); }
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #171923, #0c0d12);
        border: 1px solid rgba(255, 40, 80, .26);
        border-radius: 16px;
        padding: 18px 18px;
        min-height: 118px;
        transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
        animation: riseIn .5s ease both;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: #ff1744;
        box-shadow: 0 0 22px rgba(255, 23, 68, .18);
    }
    @keyframes riseIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    div[data-testid="stMetricLabel"] { color: #a1a1aa; }
    div[data-testid="stMetricValue"] { color: #fff; font-weight: 850; }
    .alert-panel {
        background: rgba(255, 23, 68, .09);
        border: 1px solid rgba(255, 23, 68, .65);
        border-radius: 14px;
        padding: 16px 18px;
    }
    .section-heading { font-size: 25px; font-weight: 850; margin: 14px 0 5px; }
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(255, 23, 68, .55);
        background: linear-gradient(135deg, #9b0028, #ff1744);
        color: white;
        font-weight: 750;
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 16px rgba(255, 23, 68, .35);
        border-color: #ff7895;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Load model/data -----------------------------
@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None, None, None, None
    try:
        saved = joblib.load(MODEL_PATH)
        if isinstance(saved, dict):
            return (
                saved.get("model"),
                saved.get("scaler"),
                saved.get("feature_columns"),
                saved.get("scaler_columns"),
            )
        return saved, None, None, None
    except Exception as exc:
        return None, None, None, str(exc)

model, scaler, feature_columns, scaler_columns = load_model()

@st.cache_data
def load_dataset():
    if not os.path.isfile(DATA_PATH):
        return None
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None

df = load_dataset()

# ----------------------------- Utilities -----------------------------
def prepare_features(data):
    features = data.copy()
    if "Class" in features.columns:
        features = features.drop(columns=["Class"])

    if feature_columns is not None:
        for col in feature_columns:
            if col not in features.columns:
                features[col] = 0
        features = features.loc[:, feature_columns]

    if scaler is not None:
        if scaler_columns is not None:
            for col in scaler_columns:
                if col not in features.columns:
                    features[col] = 0
            features = features.loc[:, scaler_columns]
        transformed = scaler.transform(features)
        names = list(features.columns)
        features = pd.DataFrame(transformed, columns=names, index=features.index)

    return features

def predict_transactions(data):
    if model is None:
        raise RuntimeError("Model is not loaded. Check models/fraud_detection_model.joblib.")
    features = prepare_features(data)
    preds = model.predict(features)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        probs = 1 / (1 + np.exp(-scores))
    else:
        probs = np.asarray(preds, dtype=float)
    return np.asarray(preds), np.clip(np.asarray(probs, dtype=float), 0, 1)

def header(title, subtitle):
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)

def show_metrics(items):
    # items is a list of (label, value, optional_delta)
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        delta = item[2] if len(item) > 2 else None
        with col:
            st.metric(label, value, delta=delta)

def sample_transaction():
    row = {"Time": random.randint(0, 172800), "Amount": round(random.uniform(1, 5000), 2)}
    for i in range(1, 29):
        row[f"V{i}"] = random.uniform(-5, 5)
    return row

def plotly_dark(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=55, b=15),
    )
    if height:
        fig.update_layout(height=height)
    return fig

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:26px">FRAUD SHIELD AI</div>', unsafe_allow_html=True)
    st.markdown('<span class="release-pill">V2.2 RELEASE</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### SYSTEM STATUS")
    if model is not None:
        st.success("● MODEL ONLINE")
    else:
        st.error("● MODEL OFFLINE")
        st.caption("Expected: models/fraud_detection_model.joblib")
    if isinstance(scaler, str):
        st.error(f"Model loading error: {scaler}")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        [
            "V2 Dashboard",
            "V1 Fraud Detection",
            "V3 Batch Analysis",
            "V4 Transaction Intelligence",
            "V5 Live Simulator",
            "V6 Model Performance",
            "About & Versions",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("AI-powered transaction risk analysis")
    st.caption("V2.2 • Educational demo")

# ----------------------------- V2 Dashboard -----------------------------
if page == "V2 Dashboard":
    st.markdown('<div class="hero-title">FRAUD SHIELD AI</div>', unsafe_allow_html=True)
    st.markdown('<span class="release-pill">V2.2 | TRANSACTION INTELLIGENCE</span>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Monitor transaction activity and fraud risk.</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Dataset not found. Place creditcard.csv beside app.py.")
    elif "Class" not in df.columns or "Amount" not in df.columns:
        st.error("Dataset must contain Class and Amount columns.")
    else:
        total = len(df)
        fraud = int(df["Class"].sum())
        legitimate = total - fraud
        rate = fraud / total * 100 if total else 0
        show_metrics([
            ("TOTAL TRANSACTIONS", f"{total:,}", "Dataset"),
            ("FRAUD TRANSACTIONS", f"{fraud:,}", "Known fraud"),
            ("LEGITIMATE TRANSACTIONS", f"{legitimate:,}", "Known legitimate"),
            ("FRAUD RATE", f"{rate:.3f}%", "Share of dataset"),
        ])
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            dist = pd.DataFrame({"Transaction Type": ["Legitimate", "Fraud"], "Count": [legitimate, fraud]})
            fig = px.pie(dist, names="Transaction Type", values="Count", hole=.55, title="Fraud vs Legitimate")
            st.plotly_chart(plotly_dark(fig), use_container_width=True)
        with c2:
            chart_df = df.copy()
            chart_df["Transaction Type"] = chart_df["Class"].map({0: "Legitimate", 1: "Fraud"})
            fig = px.histogram(chart_df, x="Amount", color="Transaction Type", nbins=50, title="Transaction Amount Distribution")
            st.plotly_chart(plotly_dark(fig), use_container_width=True)
        st.markdown("### Recent Dataset Transactions")
        st.dataframe(df.tail(10), use_container_width=True)

# ----------------------------- V1 Single prediction -----------------------------
elif page == "V1 Fraud Detection":
    header("V1 | FRAUD DETECTION", "Enter transaction details to estimate fraud risk.")
    if model is None:
        st.error("Model is not loaded.")
    else:
        st.markdown('<div class="alert-panel">AI TRANSACTION ANALYSIS — Submit a transaction for model classification.</div>', unsafe_allow_html=True)
        with st.form("fraud_detection_form"):
            c1, c2 = st.columns(2)
            with c1:
                amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=10.0)
            with c2:
                tx_time = st.number_input("Transaction Time", min_value=0.0, value=10000.0, step=100.0)
            st.markdown("### PCA Features")
            st.caption("V1–V28 are anonymized PCA components; their individual real-world meanings are not provided by the dataset.")
            feature_values = {}
            with st.expander("Advanced inputs: V1–V28", expanded=False):
                feature_cols = st.columns(4)
                for i in range(1, 29):
                    with feature_cols[(i - 1) % 4]:
                        feature_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1, key=f"single_v{i}")
            submitted = st.form_submit_button("ANALYZE TRANSACTION", use_container_width=True)
        if submitted:
            row = {"Time": tx_time, "Amount": amount, **feature_values}
            try:
                preds, probs = predict_transactions(pd.DataFrame([row]))
                pred, prob = int(preds[0]), float(probs[0])
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Estimated Fraud Probability", f"{prob * 100:.2f}%")
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": "Fraud Risk Score"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#ff1744"},
                            "steps": [
                                {"range": [0, 30], "color": "#153a2b"},
                                {"range": [30, 70], "color": "#55401b"},
                                {"range": [70, 100], "color": "#551523"},
                            ],
                        },
                    ))
                    st.plotly_chart(plotly_dark(gauge, 300), use_container_width=True)
                with c2:
                    if pred == 1:
                        st.error("⚠️ FRAUD RISK DETECTED")
                        st.markdown(f'<div class="alert-panel"><b>Model classified this as potentially fraudulent.</b><br>Estimated probability: {prob * 100:.2f}%</div>', unsafe_allow_html=True)
                    else:
                        st.success("CLASSIFIED AS LEGITIMATE")
                        st.info(f"Estimated fraud probability: {prob * 100:.2f}%")
                    st.warning("Prediction is not a guarantee that a transaction is safe.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ----------------------------- V3 Batch analysis -----------------------------
elif page == "V3 Batch Analysis":
    header("V3 | BATCH ANALYSIS", "Upload a CSV file and analyze multiple transactions.")
    if model is None:
        st.error("Model is not loaded.")
    else:
        upload = st.file_uploader("Upload transaction CSV", type=["csv"])
        if upload is not None:
            try:
                batch = pd.read_csv(upload)
                st.markdown("### Uploaded Transactions")
                st.dataframe(batch.head(10), use_container_width=True)
                if st.button("RUN BATCH FRAUD ANALYSIS", use_container_width=True):
                    with st.spinner("Analyzing transactions..."):
                        preds, probs = predict_transactions(batch)
                        result = batch.copy()
                        result["Fraud Prediction"] = preds
                        result["Fraud Probability"] = probs
                        result["Risk Level"] = np.where(probs >= .70, "HIGH", np.where(probs >= .30, "MEDIUM", "LOW"))
                    flagged = int((preds == 1).sum())
                    show_metrics([
                        ("TOTAL ANALYZED", f"{len(result):,}"),
                        ("FRAUD FLAGGED", f"{flagged:,}"),
                        ("LEGITIMATE", f"{len(result) - flagged:,}"),
                    ])
                    st.markdown("### Analysis Results")
                    st.dataframe(result, use_container_width=True)
                    st.download_button("DOWNLOAD ANALYSIS CSV", result.to_csv(index=False).encode("utf-8"), "fraud_analysis_results.csv", "text/csv", use_container_width=True)
            except Exception as exc:
                st.error(f"Batch analysis failed: {exc}")

# ----------------------------- V4 Intelligence -----------------------------
elif page == "V4 Transaction Intelligence":
    header("V4 | TRANSACTION INTELLIGENCE", "Explore transaction patterns and fraud behavior.")
    if df is None:
        st.warning("Dataset not found.")
    elif "Class" not in df.columns or "Amount" not in df.columns:
        st.error("Dataset must contain Class and Amount.")
    else:
        min_amt = float(df["Amount"].min())
        max_amt = float(df["Amount"].max())
        if min_amt == max_amt:
            selected_range = (min_amt, max_amt)
            st.caption(f"All transactions have amount {min_amt}.")
        else:
            selected_range = st.slider("Transaction Amount Range", min_amt, max_amt, (min_amt, max_amt))
        filtered = df[(df["Amount"] >= selected_range[0]) & (df["Amount"] <= selected_range[1])]
        st.write(f"Transactions matching filter: {len(filtered):,}")
        chart_df = filtered.copy()
        chart_df["Transaction Type"] = chart_df["Class"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.histogram(chart_df, x="Amount", color="Transaction Type", nbins=40, title="Amount Distribution by Class")
        st.plotly_chart(plotly_dark(fig), use_container_width=True)
        if "Time" in filtered.columns:
            st.markdown("### Transaction Activity Over Time")
            time_df = filtered.copy()
            time_df["Time Group"] = (time_df["Time"] // 3600).astype(int)
            summary = time_df.groupby(["Time Group", "Class"]).size().reset_index(name="Count")
            fig = px.line(summary, x="Time Group", y="Count", color="Class", markers=True, title="Transaction Count by Time Group")
            st.plotly_chart(plotly_dark(fig), use_container_width=True)
        st.markdown("### Filtered Transaction Data")
        st.dataframe(filtered.head(500), use_container_width=True)

# ----------------------------- V5 Simulator -----------------------------
elif page == "V5 Live Simulator":
    header("V5 | LIVE TRANSACTION SIMULATOR", "Generate synthetic transactions and test model predictions.")
    if model is None:
        st.error("Model is not loaded.")
    else:
        st.warning("Demo mode: synthetic transactions only. This is not connected to a real bank or payment gateway.")
        count = st.slider("Number of Transactions", 5, 100, 20)
        if st.button("START SIMULATION", use_container_width=True):
            simulation = pd.DataFrame([sample_transaction() for _ in range(count)])
            try:
                preds, probs = predict_transactions(simulation)
                simulation["Fraud Prediction"] = preds
                simulation["Fraud Probability"] = probs
                simulation["Risk Level"] = np.where(probs >= .70, "HIGH", np.where(probs >= .30, "MEDIUM", "LOW"))
                flagged = int((preds == 1).sum())
                show_metrics([
                    ("SIMULATED TRANSACTIONS", f"{count:,}"),
                    ("FLAGGED TRANSACTIONS", f"{flagged:,}"),
                    ("LEGITIMATE CLASSIFICATIONS", f"{count - flagged:,}"),
                ])
                st.markdown("### Simulator Results")
                st.dataframe(simulation, use_container_width=True)
                if flagged:
                    st.markdown('<div class="alert-panel"><b>SIMULATED FRAUD ALERT</b><br>The model flagged synthetic transactions.</div>', unsafe_allow_html=True)
                    st.dataframe(simulation[simulation["Fraud Prediction"] == 1], use_container_width=True)
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")

# ----------------------------- V6 Model performance -----------------------------
elif page == "V6 Model Performance":
    header("V6 | MODEL PERFORMANCE", "Inspect classification metrics and model behavior.")
    if model is None:
        st.error("Model is not loaded.")
    elif df is None:
        st.warning("Dataset not found.")
    elif "Class" not in df.columns:
        st.error("Dataset must contain a Class target column.")
    else:
        st.warning("For a trustworthy score, evaluate on held-out data that was not used to train the model. This page uses creditcard.csv as provided, so results may be optimistic if it overlaps training.")
        if st.button("EVALUATE MODEL", use_container_width=True):
            try:
                X_eval = df.drop(columns=["Class"])
                y_true = df["Class"].astype(int)
                with st.spinner("Evaluating model in a batch..."):
                    y_pred, _ = predict_transactions(X_eval)
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                show_metrics([
                    ("ACCURACY", f"{acc * 100:.2f}%"),
                    ("PRECISION", f"{prec * 100:.2f}%"),
                    ("RECALL", f"{rec * 100:.2f}%"),
                    ("F1 SCORE", f"{f1 * 100:.2f}%"),
                ])
                st.markdown("### Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                fig = px.imshow(
                    cm, text_auto=True,
                    labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                    x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
                    title="Confusion Matrix",
                )
                st.plotly_chart(plotly_dark(fig), use_container_width=True)
                st.markdown("### Classification Report")
                report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

# ----------------------------- About & versions -----------------------------
elif page == "About & Versions":
    header("ABOUT FRAUD SHIELD AI", "AI-powered fraud detection and transaction intelligence.")
    st.markdown("""
    ## Project Overview
    Fraud Shield AI is a machine-learning demonstration that classifies transaction records
    and estimates the likelihood of fraud using a trained model.

    Features:
    - Individual transaction analysis
    - Batch CSV analysis and export
    - Transaction pattern charts
    - Synthetic transaction simulator
    - Model metrics and confusion matrix

    **Important:** This application is educational. It is not a production banking system,
    and its output should not be used as the sole basis for financial decisions.
    """)
    st.markdown("## Version History")
    versions = pd.DataFrame({
        "Version": ["V1.0", "V2.0", "V2.1", "V2.2", "V3.0"],
        "Meaning": [
            "Initial working release",
            "Major redesign and feature additions",
            "UI and interaction improvements",
            "HTML rendering fix using native metric cards",
            "Planned future major release",
        ],
        "Status": ["Completed", "Completed", "Completed", "Current", "Planned"],
    })
    st.dataframe(versions, use_container_width=True)
    st.markdown("""
    **Version numbers do not represent model accuracy.**
    V1.0 is the first release, V2.0 is a major release, V2.1/V2.2 are incremental improvements,
    and V3.0 would be a future major release.
    """)

# ----------------------------- Footer -----------------------------
st.markdown("---")
st.caption("FRAUD SHIELD AI • V2.2 • Educational project")
