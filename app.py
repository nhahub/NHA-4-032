"""
app.py — Egypt Real Estate Appraiser (Streamlit)
=================================================
Cascading dropdowns: Governorate → City → District
Per-type model accuracy shown after type selection.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile

st.set_page_config(
    page_title="🏠 Egypt Real Estate Appraiser",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    # path = os.path.join(os.path.dirname(__file__), "model_final.pkl")
    path = os.path.join(os.path.dirname(__file__), "model_final")
    if not os.path.exists(path + ".zip"):
        st.error("❌ model_final.zip not found. Run:  python train.py")
        st.stop()
    with zipfile.ZipFile(path +".zip", 'r') as zip_ref:
        print(os.path.dirname(__file__))
        zip_ref.extractall(os.path.dirname(__file__))

    if not os.path.exists(path + ".pkl"):
        st.error("❌ model_final.pkl was not contained in the zip file. Run:  python train.py")
        st.stop()    
    return joblib.load(path + ".pkl")

B = load_bundle()

# Build cascading maps
gov_city_map      = B['gov_city_map']       # governorate → [gov — city, ...]
city_district_map = B['city_district_map']  # city → [city / district, ...]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fe/Flag_of_Egypt.svg", width=80)
    st.title("About")
    gm = B['global_metrics']
    st.markdown(f"""
**Egypt Real Estate Appraiser**
*Dubizzle Scrape — Stratified by Property Type*

**Global R²:** `{gm['R2']}`
**Global MAE:** `{gm['MAE']}M EGP`
**Global MAPE:** `{gm['MAPE']}%`
**Models:** `{len(B['type_models'])}` (one per type)

**Features:**
- Area & Bedrooms
- Governorate → City → District
- Ownership (Primary / Resale)
- Payment (Cash / Installment)
- Completion (Ready / Off-plan)

*(Bathrooms omitted — collinear with bedrooms)*
    """)
    st.markdown("---")
    st.markdown("**Per-type accuracy:**")
    for ptype, res in B['type_results'].items():
        st.markdown(f"- **{ptype}**: R²=`{res['R2']}` MAPE=`{res['MAPE']}%`")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🏠 Egypt Real Estate Appraiser")
st.markdown("**Step 1:** Select type → **Step 2:** Choose location → **Step 3:** Enter details")
st.markdown("---")

# STEP 1 — Property Type
st.subheader("① Property Type")
property_type = st.radio(
    "Select the property type:",
    options=B['valid_types'],
    horizontal=True,
)

# Show type-specific accuracy metrics
if property_type in B['type_results']:
    res = B['type_results'][property_type]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",        f"{res['R2']:.4f}")
    c2.metric("MAE",       f"{res['MAE']:.2f}M EGP")
    c3.metric("MAPE",      f"{res['MAPE']:.1f}%")
    c4.metric("Trained on", f"{res['count']:,} listings")

st.markdown("---")

# STEP 2 — Cascading Location
st.subheader("② Location")
col1, col2, col3 = st.columns(3)

with col1:
    governorate = st.selectbox("📍 Governorate", sorted(gov_city_map.keys()))

with col2:
    gc_options = sorted(gov_city_map.get(governorate, []))
    gov_city   = st.selectbox(
        "🏙️ City",
        options=gc_options,
        format_func=lambda x: x.split(' — ')[-1] if ' — ' in x else x,
    )

with col3:
    city_name  = gov_city.split(' — ')[-1] if ' — ' in gov_city else gov_city
    cd_options = sorted(city_district_map.get(city_name, [city_name + ' / ' + city_name]))
    city_dist  = st.selectbox(
        "🗺️ District / Compound",
        options=cd_options,
        format_func=lambda x: x.split(' / ')[-1] if ' / ' in x else x,
        help="Bounded to the selected city.",
    )

st.markdown("---")

# STEP 3 — Property Details
st.subheader(f"③ {property_type} Details")
col1, col2 = st.columns(2)

with col1:
    area_sqm   = st.number_input("📐 Area (m²)",   min_value=20,  max_value=1200, value=150, step=5)
    bedrooms   = st.number_input("🛏️ Bedrooms",    min_value=1,   max_value=8,    value=3,   step=1)

with col2:
    ownership   = st.selectbox("🔑 Ownership",          ["Primary", "Resale"])
    payment     = st.selectbox("💳 Payment Option",     ["Cash", "Installment", "Cash or Installment"])
    completion  = st.selectbox("🏗️ Completion Status",  ["Ready", "Off-plan"])

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("💰 Estimate Price", type="primary", use_container_width=True):
    try:
        gc_enc   = B['le_gc'].transform([gov_city])[0]
        cd_enc   = B['le_cd'].transform([city_dist])[0]
        own_enc  = {'Primary': 1, 'Resale': 0}.get(ownership, 0.5)
        pay_enc  = {'Cash': 1, 'Installment': 0, 'Cash or Installment': 0.5}.get(payment, 0.5)
        comp_enc = {'Ready': 1, 'Off-plan': 0}.get(completion, 0.5)

        Xi = pd.DataFrame([{
            'Built-Up Area (m²)': area_sqm,
            'Bedrooms'          : bedrooms,
            'gc_enc'            : gc_enc,
            'cd_enc'            : cd_enc,
            'ownership_enc'     : own_enc,
            'payment_enc'       : pay_enc,
            'completion_enc'    : comp_enc,
        }])

        model   = B['type_models'][property_type]
        price_m = np.exp(model.predict(Xi)[0])

        st.success("✅ Estimation Complete")
        col_a, col_b = st.columns(2)
        with col_a: st.metric("Estimated Price",    f"{price_m:.2f}M EGP")
        with col_b: st.metric("In Egyptian Pounds", f"{price_m * 1_000_000:,.0f} EGP")

        # Summary table
        city_display = gov_city.split(' — ')[-1]  if ' — ' in gov_city  else gov_city
        dist_display = city_dist.split(' / ')[-1] if ' / ' in city_dist else city_dist
        st.markdown("#### Property Summary")
        st.table(pd.DataFrame({
            "Feature": ["Type","Governorate","City","District","Area","Bedrooms","Ownership","Payment","Completion"],
            "Value"  : [property_type, governorate, city_display, dist_display,
                        f"{area_sqm} m²", bedrooms, ownership, payment, completion],
        }))

        res = B['type_results'][property_type]
        st.info(
            f"📊 **{property_type} model** ({res['model_name']})  |  "
            f"R² = `{res['R2']}`  |  "
            f"MAE = `{res['MAE']}M EGP`  |  "
            f"MAPE = `{res['MAPE']}%`  |  "
            f"Trained on `{res['count']:,}` listings"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check that the selected location combination exists in the training data.")

st.markdown("---")
st.caption("🎓 Egypt Real Estate Appraiser · Dubizzle Dataset · DEPI Data Science Track · May 2026")
