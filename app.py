import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Bank Churn Modeling", layout="wide")

# State awal
if "step" not in st.session_state:
    st.session_state.step = 1
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clustered" not in st.session_state:
    st.session_state.df_clustered = None
if "preprocess_cols" not in st.session_state:
    st.session_state.preprocess_cols = None


@st.cache_data
def load_data():
    return pd.read_csv("Bank Customer Churn Prediction.csv")


# ==============================================
# STEP 1 — PREPROCESSING
# ==============================================
def preprocessing_page():
    st.header("🧹 Tahap 1 — Preprocessing")

    df = st.session_state.df
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # pastikan churn tidak bisa dipilih
    if "churn" in num_cols:
        num_cols.remove("churn")

    selected = st.multiselect("Pilih kolom numerik untuk preprocessing (minimal 2):", num_cols)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Jalankan Preprocessing"):
            if len(selected) < 2:
                st.error("Pilih minimal 2 kolom!")
                return
            scaler = StandardScaler()
            df[selected] = scaler.fit_transform(df[selected])
            st.session_state.preprocess_cols = selected
            st.success("Preprocessing selesai!")
            st.session_state.step = 2

    st.write("Dataset (preview):")
    st.dataframe(df.head())


# ==============================================
# STEP 2 — CLUSTERING
# ==============================================
def clustering_page():
    st.header("🔮 Tahap 2 — Clustering")

    df = st.session_state.df
    cols = st.session_state.preprocess_cols

    cluster_k = st.slider("Jumlah cluster", min_value=2, max_value=10, value=4)

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("🚀 Jalankan Clustering"):
            kmeans = KMeans(n_clusters=cluster_k, random_state=42)
            df["cluster_id"] = kmeans.fit_predict(df[cols])
            st.session_state.df_clustered = df
            st.success("Clustering selesai!")
            st.session_state.step = 3
    with col2:
        if st.button("⬅ Kembali ke Preprocessing"):
            st.session_state.step = 1

    st.write("Dataset (preview):")
    st.dataframe(df.head())


# ==============================================
# STEP 3 — MODELING
# ==============================================
def modeling_page():
    st.header("📌 Tahap 3 — Modeling Logistic Regression")

    df_no_cluster = st.session_state.df.drop(columns=["cluster_id"])
    df_cluster = st.session_state.df_clustered
    df_cluster["churn"] = df_cluster["churn"].astype(int)

    y = df_cluster["churn"]
    X_no = df_no_cluster.drop(columns=["churn"])
    X_cluster = df_cluster.drop(columns=["churn"])

    # pipeline tanpa clustering
    cat_no = X_no.select_dtypes(include=["object"]).columns.tolist()
    num_no = X_no.select_dtypes(include=["number"]).columns.tolist()
    preprocess_no = ColumnTransformer(
        [("num", StandardScaler(), num_no),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_no)]
    )
    model = Pipeline([("preprocess", preprocess_no),
                      ("logreg", LogisticRegression(max_iter=500, class_weight="balanced"))])

    X_train, X_test, y_train, y_test = train_test_split(X_no, y, test_size=0.25, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    auc_no = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # pipeline dengan clustering
    cat_cl = X_cluster.select_dtypes(include=["object"]).columns.tolist()
    num_cl = X_cluster.select_dtypes(include=["number"]).columns.tolist()
    preprocess_cl = ColumnTransformer(
        [("num", StandardScaler(), num_cl),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cl)]
    )
    model2 = Pipeline([("preprocess", preprocess_cl),
                       ("logreg", LogisticRegression(max_iter=500, class_weight="balanced"))])

    X_train, X_test, y_train, y_test = train_test_split(X_cluster, y, test_size=0.25, random_state=42, stratify=y)
    model2.fit(X_train, y_train)
    auc_cluster = roc_auc_score(y_test, model2.predict_proba(X_test)[:, 1])
    cm_cluster = confusion_matrix(y_test, model2.predict(X_test))

    # HASIL
    st.success("Modeling selesai!")
    st.write(f"🔥 **AUC tanpa clustering:** `{auc_no:.4f}`")
    st.write(f"🔥 **AUC dengan clustering:** `{auc_cluster:.4f}`")

    # CONFUSION MATRIX — VISUAL
    st.subheader("📊 Confusion Matrix (dengan clustering)")
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm_cluster, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig_cm)

    # ROC CURVE — VISUAL
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, model2.predict_proba(X_test)[:, 1])
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    st.pyplot(fig_roc)

    # BACK BUTTON
    if st.button("⬅ Kembali ke Clustering"):
        st.session_state.step = 2


# ==============================================
# MAIN LAYOUT
# ==============================================
st.sidebar.title("📌 Bank Customer Churn Prediction")

if st.session_state.df is None:
    st.session_state.df = load_data()
    st.sidebar.success("Dataset dimuat")

st.sidebar.write(f"Jumlah data: {len(st.session_state.df)} baris")

if st.session_state.step == 1:
    preprocessing_page()
elif st.session_state.step == 2:
    clustering_page()
elif st.session_state.step == 3:
    modeling_page()
