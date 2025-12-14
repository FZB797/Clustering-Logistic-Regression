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

# =====================================================
# SESSION STATE
# =====================================================
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


# =====================================================
# STEP 1 — PREPROCESSING
# =====================================================
def preprocessing_page():
    st.header("🧹 Tahap 1 — Preprocessing")

    df = st.session_state.df
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if "churn" in num_cols:
        num_cols.remove("churn")

    selected = st.multiselect(
        "Pilih kolom numerik untuk preprocessing (minimal 2):",
        num_cols
    )

    if st.button("🔄 Jalankan Preprocessing"):
        if len(selected) < 2:
            st.error("Minimal pilih 2 kolom numerik!")
            return

        scaler = StandardScaler()
        df[selected] = scaler.fit_transform(df[selected])

        st.session_state.preprocess_cols = selected
        st.session_state.step = 2
        st.success("Preprocessing berhasil!")

    st.dataframe(df.head())


# =====================================================
# STEP 2 — CLUSTERING
# =====================================================
def clustering_page():
    st.header("🔮 Tahap 2 — Clustering (K-Means)")

    df = st.session_state.df
    cols = st.session_state.preprocess_cols

    k = st.slider("Jumlah Cluster", 2, 10, 4)

    if st.button("🚀 Jalankan Clustering"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["cluster_id"] = kmeans.fit_predict(df[cols])
        st.session_state.df_clustered = df.copy()
        st.session_state.step = 3
        st.success("Clustering selesai!")

    # ===== OUTPUT VISUAL CLUSTERING =====
    if "cluster_id" in df.columns and len(cols) >= 2:
        st.subheader("📊 Visualisasi Clustering")

        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x=cols[0],
            y=cols[1],
            hue="cluster_id",
            palette="tab10",
            ax=ax
        )
        ax.set_title("Scatter Plot Hasil Clustering")
        st.pyplot(fig)

        st.subheader("📈 Distribusi Data per Cluster")
        fig2, ax2 = plt.subplots()
        df["cluster_id"].value_counts().sort_index().plot(
            kind="bar", ax=ax2
        )
        ax2.set_xlabel("Cluster ID")
        ax2.set_ylabel("Jumlah Data")
        st.pyplot(fig2)

    if st.button("⬅ Kembali ke Preprocessing"):
        st.session_state.step = 1



# =====================================================
# STEP 3 — LOGISTIC REGRESSION
# =====================================================
def modeling_page():
    st.header("📌 Tahap 3 — Logistic Regression (Dengan Clustering)")

    df = st.session_state.df_clustered.copy()
    df["churn"] = df["churn"].astype(int)

    y = df["churn"]
    X = df.drop(columns=["churn"])

    # cluster_id dianggap kategorikal
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    cat_cols.append("cluster_id")

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if "cluster_id" in num_cols:
        num_cols.remove("cluster_id")

    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("logreg", LogisticRegression(
                max_iter=500,
                class_weight="balanced"
            )),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    st.success("Modeling selesai!")
    st.write(f"🔥 **AUC Logistic Regression + Clustering:** `{auc:.4f}`")

    # ===== CONFUSION MATRIX =====
    st.subheader("📊 Confusion Matrix")
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig_cm)

    # ===== ROC CURVE =====
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig_roc)

    if st.button("⬅ Kembali ke Clustering"):
        st.session_state.step = 2


# =====================================================
# MAIN APP
# =====================================================
st.sidebar.title("📌 Navigasi")

if st.session_state.df is None:
    st.session_state.df = load_data()
    st.sidebar.success("Dataset dimuat")

st.sidebar.write(f"Jumlah data: {len(st.session_state.df)}")

if st.session_state.step == 1:
    preprocessing_page()
elif st.session_state.step == 2:
    clustering_page()
elif st.session_state.step == 3:
    modeling_page()
