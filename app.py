import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Customer Churn Pipeline", layout="wide")
st.title("🔍 Customer Churn — Preprocessing → Clustering → Modelling → Visualisasi")

# SESSION STATE
if "page" not in st.session_state: st.session_state.page = 1
if "df" not in st.session_state: st.session_state.df = None
if "df_pre" not in st.session_state: st.session_state.df_pre = None
if "clusters" not in st.session_state: st.session_state.clusters = None
if "model_result" not in st.session_state: st.session_state.model_result = None


# BUTTON NAVIGATION
def next_page(): st.session_state.page += 1


# LOAD DATA LOCAL TANPA UPLOAD
def load_data():
    return pd.read_csv("Bank Customer Churn Prediction.csv")

if "df" not in st.session_state or st.session_state.df is None:
    try:
        st.session_state.df = load_data()
        st.success(f"Dataset dimuat otomatis ✔ ({len(st.session_state.df)} baris)")
    except FileNotFoundError:
        st.error("❌ File 'Bank Customer Churn Prediction.csv' tidak ditemukan.")



############################################################
# 1️⃣ PREPROCESSING
############################################################
if st.session_state.page == 1 and st.session_state.df is not None:
    st.subheader("1️⃣ Preprocessing Data")

    numeric_columns = st.session_state.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_cols = st.multiselect("Pilih kolom numerik untuk pemodelan (minimal 2 kolom):",
                                   numeric_columns)

    if len(selected_cols) >= 2:
        X = st.session_state.df[selected_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state.df_pre = pd.DataFrame(X_scaled, columns=selected_cols)
        st.write("📌 **Hasil Preprocessing (Standard Scaling):**")
        st.dataframe(st.session_state.df_pre.head())

        if st.button("➡️ Lanjut ke Clustering"):
            next_page()

############################################################
# 2️⃣ CLUSTERING
############################################################
elif st.session_state.page == 2 and st.session_state.df_pre is not None:
    st.subheader("2️⃣ Clustering (K-Means)")

    k = st.slider("Tentukan jumlah cluster:", 2, 10, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(st.session_state.df_pre)
    st.session_state.clusters = clusters
    st.session_state.df_pre["cluster"] = clusters

    st.write("📌 **Label Cluster Ditambahkan**")
    st.dataframe(st.session_state.df_pre.head())

    # Scatter Visualization
    fig, ax = plt.subplots()
    ax.scatter(st.session_state.df_pre.iloc[:, 0], st.session_state.df_pre.iloc[:, 1], c=clusters)
    ax.set_xlabel(st.session_state.df_pre.columns[0])
    ax.set_ylabel(st.session_state.df_pre.columns[1])
    st.pyplot(fig)

    if st.button("➡️ Lanjut ke Logistic Regression"):
        next_page()

############################################################
# 3️⃣ LOGISTIC REGRESSION
############################################################
elif st.session_state.page == 3 and st.session_state.clusters is not None:
    st.subheader("3️⃣ Training Logistic Regression")

    if "Churn" not in st.session_state.df.columns:
        st.error("Kolom target `Churn` wajib ada di dataset!")
    else:
        X = st.session_state.df_pre.drop(columns=["cluster"])
        y = st.session_state.df["Churn"]

        logreg = LogisticRegression(max_iter=300)
        logreg.fit(X, y)
        preds = logreg.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, preds)
        cm = confusion_matrix(y, (preds > 0.5).astype(int))

        st.session_state.model_result = {"auc": auc, "cm": cm, "preds": preds}

        st.success("🎉 Model selesai dilatih!")
        st.write("AUC:", auc)
        st.write("Confusion Matrix:")
        st.write(cm)

        if st.button("➡️ Lanjut ke Visualisasi Akhir"):
            next_page()

############################################################
# 4️⃣ VISUALISASI AKHIR
############################################################
elif st.session_state.page == 4 and st.session_state.model_result is not None:
    st.subheader("4️⃣ Visualisasi Hasil Akhir")

    preds = st.session_state.model_result["preds"]
    df_vis = st.session_state.df_pre.copy()
    df_vis["prediksi_churn"] = preds

    fig, ax = plt.subplots()
    sc = ax.scatter(df_vis.iloc[:, 0], df_vis.iloc[:, 1],
                    c=df_vis["prediksi_churn"], cmap="coolwarm")
    plt.colorbar(sc, label="Probabilitas Churn")
    ax.set_xlabel(df_vis.columns[0])
    ax.set_ylabel(df_vis.columns[1])
    st.pyplot(fig)

    st.success("📊 Visualisasi akhir selesai — pipeline lengkap!")

