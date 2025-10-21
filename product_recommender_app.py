import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="🛍️ Retail Intelligence", layout="centered")

# ---------------- DEBUG: CURRENT DIRECTORY ----------------
# st.write("📂 Current working directory:", os.getcwd())
# st.write("📁 Files in directory:", os.listdir())

# ---------------- TITLE ----------------
st.title("🛍️ Retail Intelligence Dashboard")
st.markdown("Choose a module from the sidebar to begin.")

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_rfm_model():
    try:
        with open("rfm_kmeans_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading rfm_kmeans_model.pkl: {e}")
        return None

@st.cache_resource
def load_product_model():
    try:
        with open("product_recommender.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading product_recommender.pkl: {e}")
        return None

# Load Models
rfm_bundle = load_rfm_model()
product_bundle = load_product_model()

if rfm_bundle:
    scaler = rfm_bundle["scaler"]
    kmeans = rfm_bundle["model"]
else:
    st.stop()

if product_bundle:
    similarity_df = product_bundle["product_similarity_df"]
    code_to_desc = product_bundle["code_to_desc"]
    desc_to_code = product_bundle["desc_to_code"]
else:
    st.stop()

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("📂 Select Module", ["📊 Customer Segmentation", "🔍 Product Recommendation"])

# ---------------- CUSTOMER SEGMENTATION ----------------
if page == "📊 Customer Segmentation":
    st.header("📊 Customer Segmentation")

    recency = st.number_input("📅 Recency (days)", min_value=0, value=30)
    frequency = st.number_input("🔁 Frequency (purchases)", min_value=0, value=5)
    monetary = st.number_input("💸 Monetary Value (₹)", min_value=0.0, value=200.0)

    if st.button("🔍 Predict Cluster"):
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
        input_log = np.log1p(input_df)
        input_scaled = scaler.transform(input_log)
        cluster = kmeans.predict(input_scaled)[0]

        cluster_labels = {
            0: "🔵 Occasional",
            1: "🟢 High-Value",
            2: "🟡 Regular",
            3: "🔴 At-Risk"
        }

        label = cluster_labels.get(cluster, f"Cluster {cluster}")
        st.success(f"📌 Predicted Segment: **{label}**")

# ---------------- PRODUCT RECOMMENDATION ----------------
elif page == "🔍 Product Recommendation":
    st.header("🔍 Product Recommendation")

    product_list = sorted(desc_to_code.index.tolist())
    selected_product = st.selectbox("Select a product", product_list)

    if st.button("🎁 Get Recommendations"):
        if selected_product in desc_to_code:
            seed_code = desc_to_code[selected_product]

            if seed_code in similarity_df.columns:
                sim_series = similarity_df[seed_code].sort_values(ascending=False)
                top_codes = sim_series.index[1:6]

                recs = pd.DataFrame({
                    'StockCode': top_codes,
                    'Similarity': [sim_series[c] for c in top_codes],
                    'Description': [code_to_desc.get(c, "Unknown") for c in top_codes]
                })

                st.markdown("### 🧠 Top 5 Similar Products:")
                for _, row in recs.iterrows():
                    st.markdown(f"- **{row['Description']}** _(Similarity: {row['Similarity']:.2f})_")
            else:
                st.error("❌ Product code not found in similarity matrix.")
        else:
            st.error("❌ Product name not found in mapping.")
