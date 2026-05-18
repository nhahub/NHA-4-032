"""
Egypt Real Estate Appraiser — Streamlit Web Application
Predicts property prices based on user-provided features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 Egypt Real Estate Appraiser",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(model_path):
        st.error("❌ model.pkl not found. Run `python train.py` first.")
        st.stop()
    return joblib.load(model_path)

bundle = load_model()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — About
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fe/Flag_of_Egypt.svg", width=80)
    st.title("About")
    st.markdown("""
**Egypt Real Estate Appraiser** uses machine learning to estimate property prices
in the Egyptian market.

**Model:** {}  
**R² Score:** {:.4f}  
**MAE:** {:.3f}M EGP  
**Dataset:** ~16,000 cleaned listings
    """.format(
        bundle["model_name"],
        bundle["metrics"]["R² Score"],
        bundle["metrics"]["MAE (M EGP)"],
    ))
    st.markdown("---")
    st.markdown("Built with ❤️ using **Scikit-learn** + **Streamlit**")

# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("🏠 Egypt Real Estate Appraiser")
st.markdown("Enter the property details below to get an estimated market price.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    governorate = st.selectbox(
        "📍 Governorate",
        options=bundle["governorates"],
        help="Select the governorate where the property is located."
    )
    property_type = st.selectbox(
        "🏗️ Property Type",
        options=bundle["property_types"],
        help="Select the type of property."
    )
    payment_method = st.selectbox(
        "💳 Payment Method",
        options=bundle["payment_methods"],
        help="Select the payment method."
    )

with col2:
    size_sqm = st.number_input(
        "📐 Size (sqm)",
        min_value=20, max_value=1000, value=120, step=5,
        help="Property area in square meters."
    )
    bedrooms = st.number_input(
        "🛏️ Bedrooms",
        min_value=1, max_value=7, value=2, step=1
    )
    bathrooms = st.number_input(
        "🚿 Bathrooms",
        min_value=1, max_value=7, value=2, step=1
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────
if st.button("💰 Estimate Price", type="primary", use_container_width=True):
    try:
        gov_enc  = bundle["le_gov"].transform([governorate])[0]
        type_enc = bundle["le_type"].transform([property_type])[0]
        pay_enc  = bundle["le_pay"].transform([payment_method])[0]

        X_input = pd.DataFrame([{
            "size_sqm"    : size_sqm,
            "bedrooms_num": bedrooms,
            "bathrooms"   : bathrooms,
            "gov_enc"     : gov_enc,
            "type_enc"    : type_enc,
            "pay_enc"     : pay_enc,
        }])

        price_m = bundle["model"].predict(X_input)[0]
        price_egp = price_m * 1_000_000

        # Show result
        st.success("✅ Estimation Complete")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Estimated Price", f"{price_m:.2f}M EGP")
        with col_b:
            st.metric("In Egyptian Pounds", f"{price_egp:,.0f} EGP")

        # Property summary
        st.markdown("#### Property Summary")
        summary_data = {
            "Feature": ["Location", "Type", "Payment", "Size", "Bedrooms", "Bathrooms"],
            "Value"  : [governorate, property_type, payment_method,
                        f"{size_sqm} sqm", bedrooms, bathrooms],
        }
        st.table(pd.DataFrame(summary_data))

        st.info(
            f"📊 This estimate is based on a **{bundle['model_name']}** model "
            f"trained on ~16,000 Egyptian property listings. "
            f"The model has a Mean Absolute Error of **{bundle['metrics']['MAE (M EGP)']}M EGP** "
            f"and an R² score of **{bundle['metrics']['R² Score']}**."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🎓 Egypt Real Estate Appraiser | Data Science Graduation Project")
