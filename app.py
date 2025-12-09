# ================================================================
#  BANK CUSTOMER CHURN — CLUSTERING + LOGISTIC REGRESSION PIPELINE
#  Streamlit Version — CLEAN & WORKING
# ================================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PATH = "Bank Customer Churn Prediction.csv"
RANDOM_STATE = 42


# ----------------------------
# LOAD & BASIC CLEANING
# ----------------------------
def load_data():
    df = pd.read_csv(DATASET_PATH)

    # Drop identifier if exists
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # Ensure binary columns
    for col in ["churn", "credit_card", "active_member"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


# ----------------------------
# PREPROCESSING FOR CLUSTERING
# ----------------------------
def clustering_features(df):
    df["balance_log"] = np.log1p(df["balance"])
    df["estimated_salary_log"] = np.log1p(df["estimated_salary"])

    cluster_cols = ["credit_score", "age", "tenure", "balance_log",
                    "products_number", "estimated_salary_log"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_cols])

    return X_scaled, df


# ----------------------------
# RUN CLUSTERING
# ----------------------------
def run_clustering(df):
    X_scaled, df = clustering_features(df)

    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE)
    df["cluster_id"] = kmeans.fit_predict(X_scaled)

    return df


# ----------------------------
# LOGISTIC REGRESSION
# ----------------------------
def run_logistic_regression(df):
    y = df["churn"]
    X = df.drop(columns=["churn"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("logreg", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "AUC": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }

    return metrics


# ================================================================
#  STREAMLIT SECTION
# ================================================================
st.title("📌 Bank Customer Churn — Clustering + Logistic Regression")

st.write("🔄 **Loading dataset...**")
df = load_data()

st.write("🔍 **Running clustering...**")
df_clustered = run_clustering(df.copy())

# Logistic Regression WITHOUT clustering feature
st.write("⚙ **Evaluating Logistic Regression (TANPA clustering)...**")
metrics_no_cluster = run_logistic_regression(df.copy())

# Logistic Regression WITH clustering feature
st.write("⚙ **Evaluating Logistic Regression (DENGAN clustering)...**")
metrics_with_cluster = run_logistic_regression(df_clustered.copy())

# Results
st.subheader("📊 Hasil Evaluasi Model")
st.write("AUC Logistic Regression (tanpa clustering):", metrics_no_cluster["AUC"])
st.write("AUC Logistic Regression (dengan clustering):", metrics_with_cluster["AUC"])

st.write("Confusion Matrix (dengan clustering):")
st.write(metrics_with_cluster["confusion_matrix"])

st.success("Selesai ✔ Pipeline berhasil dijalankan!")
