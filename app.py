import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Customer Churn ML Pipeline", layout="wide")

# State
if "page" not in st.session_state:
    st.session_state.page = "preprocessing"

if "df" not in st.session_state:
    st.session_state.df = None

if "processed" not in st.session_state:
    st.session_state.processed = None

if "clustered" not in st.session_state:
    st.session_state.clustered = None


st.title("💠 End-to-End ML Pipeline — Customer Churn")

# UPLOAD DATA
if st.session_state.df is None:
    file = st.file_uploader("Upload dataset (.csv)", type="csv")
    if file:
        st.session_state.df = pd.read_csv(file)
        st.success("Dataset berhasil dimuat!")
        st.write("Preview dataset:")
        st.dataframe(st.session_state.df.head())

else:
    st.sidebar.success("Dataset telah dimuat")
    st.sidebar.write(f"Jumlah data: {len(st.session_state.df)} baris")

    # PREPROCESSING
    if st.session_state.page == "preprocessing":
        st.header("🔹 Preprocessing Data")

        num_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        selected_cols = st.multiselect(
            "Pilih kolom numerik minimal 2 kolom:", num_cols
        )

        if selected_cols and len(selected_cols) >= 2:
            scaler = StandardScaler()
            processed = pd.DataFrame(
                scaler.fit_transform(st.session_state.df[selected_cols]),
                columns=selected_cols
            )
            st.session_state.processed = processed
            st.write("📌 Hasil Scaling:")
            st.dataframe(processed.head())

            if st.button("➡ Lanjut ke Clustering"):
                st.session_state.page = "clustering"

        else:
            st.warning("Pilih minimal 2 kolom numerik.")

    # CLUSTERING
    elif st.session_state.page == "clustering":
        st.header("🔹 Clustering (K-Means)")

        X = st.session_state.processed

        # Saran cluster optimal (Elbow)
        distortions = []
        K = range(2, 11)
        for k in K:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X)
            distortions.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K, distortions, marker="o")
        ax.set_title("Elbow Method")
        ax.set_xlabel("Jumlah Cluster")
        ax.set_ylabel("Inertia")
        st.pyplot(fig)

        cluster_num = st.slider("Pilih jumlah cluster:", 2, 10, 3)
        kmeans = KMeans(n_clusters=cluster_num, random_state=42)
        labels = kmeans.fit_predict(X)

        df_clustered = st.session_state.df.copy()
        df_clustered["Cluster"] = labels
        st.session_state.clustered = df_clustered

        st.success("Clustering Berhasil!")
        st.dataframe(df_clustered.head())

        # PCA Visual
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        df_plot = pd.DataFrame({
            "PCA1": pca_result[:, 0],
            "PCA2": pca_result[:, 1],
            "Cluster": labels
        })
        st.subheader("📌 Visualisasi PCA Clustering")
        st.plotly_chart(px.scatter(df_plot, x="PCA1", y="PCA2", color="Cluster"))

        if st.button("➡ Lanjut ke Modelling Logistic Regression"):
            st.session_state.page = "modeling"

    # MODELING
    elif st.session_state.page == "modeling":
        st.header("🔹 Logistic Regression Modelling")

        df = st.session_state.clustered

        target_col = st.selectbox("Pilih kolom target:", df.columns)
        feature_cols = [c for c in df.columns if c not in [target_col]]

        X = pd.get_dummies(df[feature_cols], drop_first=True)
        y = df[target_col]

        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, output_dict=False)

        st.success("Modeling selesai!")
        st.write("🔹 AUC Score:", auc)
        st.write("🔹 Confusion Matrix:")
        st.write(cm)
        st.text("🔹 Classification Report:\n" + report)

        # ROC curve
        fpr, tpr, _ = roc_curve(y, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        st.pyplot(fig)

