import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 🎨 App Config
st.set_page_config(page_title="🛍️ Retail Intelligence App", layout="wide")

# 🧾 Check working directory and files
st.sidebar.title("📂 App Diagnostics")
cwd = os.getcwd()
files = os.listdir(cwd)

# 📦 Load RFM Model
@st.cache_resource
def load_rfm_model():
    try:
        with open("rfm_kmeans_model.pkl", "rb") as f:
            model = pickle.load(f)
            return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading RFM model: {e}")
        return None

# 📦 Load Product Recommender Model
@st.cache_resource
def load_product_model():
    try:
        with open("product_recommender.pkl", "rb") as f:
            recommender = pickle.load(f)
            return recommender
    except Exception as e:
        st.sidebar.error(f"❌ Error loading recommender: {e}")
        return None

# Load models
rfm_bundle = load_rfm_model()
product_bundle = load_product_model()

# Unpack RFM model
scaler, kmeans = None, None
if rfm_bundle:
    scaler = rfm_bundle.get("scaler")
    kmeans = rfm_bundle.get("model")

# Unpack Product Recommender
similarity_df, code_to_desc, desc_to_code = None, None, None
if product_bundle:
    similarity_df = product_bundle.get("product_similarity_df")
    code_to_desc = product_bundle.get("code_to_desc")
    desc_to_code = product_bundle.get("desc_to_code")

# Sidebar Navigation
page = st.sidebar.radio("📌 Select Module", ["📊 Customer Segmentation", "🛍️ Product Recommendation"])

# ========================
# 📊 Customer Segmentation
# ========================
if page == "📊 Customer Segmentation":
    st.title("📊 Predict Customer Segment")

    if scaler is None or kmeans is None:
        st.warning("⚠️ Model not loaded. Please check your pickle file.")
    else:
        recency = st.number_input("Recency (days)", min_value=0, value=30)
        frequency = st.number_input("Frequency (purchases)", min_value=0, value=5)
        monetary = st.number_input("Monetary Value (₹)", min_value=0.0, value=200.0)

        if st.button("Predict Cluster"):
            input_df = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
            input_log = np.log1p(input_df)
            input_scaled = scaler.transform(input_log)
            cluster = kmeans.predict(input_scaled)[0]
            
            cluster_names = {
                0: "🔴 At-Risk",
                1: "🟢 High-Value",
                2: "🟡 Regular",
                3: "🔵 Occasional"}


            st.success(f"📌 Predicted Segment: **{cluster_names.get(cluster, f'Cluster {cluster}')}**")

# =========================
# 🛍️ Product Recommendation
# =========================
elif page == "🛍️ Product Recommendation":
    st.title("🛍️ Product Recommendation System")

    if similarity_df is None or desc_to_code is None:
        st.warning("⚠️ Recommender model not loaded.")
    else:
        # Use dropdown (selectbox)
        all_products = sorted(desc_to_code.keys())
        product_name = st.selectbox("Select a Product to Recommend Similar Ones:", all_products)

        if st.button("Get Recommendations"):
            code = desc_to_code[product_name]
            sims = similarity_df[code].sort_values(ascending=False)[1:6]  # Skip self

            recs = pd.DataFrame({
                "StockCode": sims.index,
                "Similarity": sims.values,
                "Description": [code_to_desc.get(c, "N/A") for c in sims.index]
            })

            st.subheader("🔎 Recommended Products")
            for _, row in recs.iterrows():
                st.markdown(f"**{row['Description']}**  \nSimilarity: `{row['Similarity']:.2f}`")
