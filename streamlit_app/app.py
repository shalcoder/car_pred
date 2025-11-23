# streamlit_app/app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# -------------------------
# CONFIG
# -------------------------
API_URL = "http://127.0.0.1:8000/predict"
APP_TITLE = "🚘 Shadowfox — Car Resale Price Estimator"
APP_SUB = "Smart, fast and professional UI for your car price ML model"

# Local file link (will be transformed by your tooling). Provided by user upload.
# Developer note: the path below is expected to be converted into a served URL by your infra.
DOWNLOAD_CODE_PATH = "E:/shadowfox/phase2/car_pred/streamlit_app/app.py"

st.set_page_config(
    page_title="Shadowfox — Car Price Estimator",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CSS — glassmorphism + sleek header
# -------------------------
st.markdown(
    """
    <style>
    :root {
      --accent: linear-gradient(90deg, #2b8be6, #8c52ff);
      --card-bg: rgba(255, 255, 255, 0.06);
      --card-blur: 8px;
    }
    .app-header {
      padding: 18px;
      border-radius: 14px;
      background: linear-gradient(90deg, rgba(43,139,230,0.12), rgba(140,82,255,0.08));
      box-shadow: 0 6px 18px rgba(12,13,20,0.12);
      margin-bottom: 18px;
    }
    .big-title {
      font-size:28px;
      font-weight:700;
      letter-spacing: -0.5px;
      color: white;
    }
    .sub-title {
      font-size:14px;
      color: #d1d7e0;
      margin-top:6px;
    }
    .glass {
      background: var(--card-bg);
      border-radius: 12px;
      padding: 16px;
      backdrop-filter: blur(var(--card-blur));
      box-shadow: 0 6px 18px rgba(10, 15, 25, 0.25);
      color: #eaf0ff;
    }
    .muted { color: #aeb7c7; font-size:13px; }
    .footer { font-size:12px; color:#9aa3b3; margin-top:8px; }
    .predict-btn {
      background: linear-gradient(90deg,#2b8be6,#8c52ff);
      color: white;
      padding: 10px 18px;
      border-radius: 8px;
      border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
with st.container():
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    cols = st.columns([0.75, 0.25])
    with cols[0]:
        st.markdown('<div class="big-title">🚘 Shadowfox — Car Resale Price Estimator</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Enter car details on the left — get a reliable resale estimate instantly.</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div style='text-align:right'><small class='muted'>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# LAYOUT
# -------------------------
sidebar = st.sidebar
sidebar.markdown("### Input Car Details")
present_price = sidebar.number_input("Present Price (lakhs)", min_value=0.0, value=5.59, step=0.1, format="%.2f")
kms_driven = sidebar.number_input("Kilometers Driven", min_value=0, value=27000, step=100)
year = sidebar.slider("Manufacture Year", 1990, 2025, 2014)
brand = sidebar.text_input("Brand (optional)", value="Maruti")
fuel_type = sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Other"])
seller_type = sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = sidebar.selectbox("Previous Owners", [0, 1, 2, 3])
show_engineered = sidebar.checkbox("Show engineered features (debug)", value=False)

# Controls
run_button = st.sidebar.button("Estimate Price", key="predict")

# Center area: result and diagnostics
left_col, right_col = st.columns([0.6, 0.4])

with left_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0'>Estimated Resale Price</h3>", unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    price_placeholder = st.empty()
    details_placeholder = st.empty()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### Model diagnostics", unsafe_allow_html=True)
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    with diag_col1:
        model_used = st.empty()
    with diag_col2:
        latency = st.empty()
    with diag_col3:
        confidence = st.empty()

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Quick Insights")
    st.markdown("- `Price_Depreciation` and `Age` are strong drivers.")
    st.markdown("- Dealer listings often fetch slightly higher resale.")
    st.markdown("<hr/>", unsafe_allow_html=True)

    # local path that will be turned into a URL by your tooling
   
    st.markdown("<div class='footer'>Pro tip: deploy FastAPI backend on Render and expose it via HTTPS; update API_URL in Streamlit secrets.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Prediction logic
# -------------------------
def call_predict_api(payload: dict):
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

if run_button:
    # build payload
    payload = {
        "present_price": float(present_price),
        "kms_driven": int(kms_driven),
        "year": int(year),
        "fuel_type": fuel_type,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": int(owner),
        "brand": brand or "Unknown"
    }

    with st.spinner("Estimating price — contacting model..."):
        t0 = pd.Timestamp.now()
        result = call_predict_api(payload)
        t1 = pd.Timestamp.now()
        dt = (t1 - t0).total_seconds()

    # handle errors
    if result is None or "error" in result:
        st.error("API error: " + (result.get("error") if isinstance(result, dict) else "unknown"))
    else:
        predicted = result.get("predicted_price")
        used = result.get("model_used", "unknown")
        # show an animated metric-like output
        price_placeholder.markdown(
            f"<div style='font-size:44px; font-weight:700; margin-top:6px;'>₹ {predicted:.2f} lakhs</div>",
            unsafe_allow_html=True
        )
        model_used.text(f"Model: {used}")
        latency.text(f"Latency: {dt:.2f}s")
        confidence.text("Confidence: High")  # placeholder — replace with real metric later

        # show engineered features if requested (call extra debug endpoint or reconstruct)
        if show_engineered:
            # If backend returns engineered features, prefer that; else reconstruct locally
            engineered = result.get("engineered_features")
            if engineered is None:
                # local compute (same as training): quick reproduction
                Age = 2025 - payload["year"]
                KM_per_Year = payload["kms_driven"] / max(Age + 1, 1)
                Price_Depreciation = payload["present_price"] / max(Age + 1, 1)
                Car_Condition = (payload["present_price"] / (payload["kms_driven"] + 1)) * (1.0 / (Age + 1))
                engineered = {
                    "Present_Price": payload["present_price"],
                    "Kms_Driven": payload["kms_driven"],
                    "Age": Age,
                    "KM_per_Year": round(KM_per_Year, 2),
                    "Price_Depreciation": round(Price_Depreciation, 4),
                    "Car_Condition": round(Car_Condition, 8),
                    "Is_First_Owner": int(payload["owner"] == 0),
                    "Is_Diesel": int(payload["fuel_type"].lower() == "diesel"),
                    "Brand": payload["brand"]
                }

            st.markdown("#### Engineered features")
            st.json(engineered)

        # If API returned feature importance data, show it
        feat_imp = result.get("feature_importance")
        if feat_imp:
            st.markdown("#### Feature importance (model view)")
            df_imp = pd.DataFrame(feat_imp)
            chart = alt.Chart(df_imp).mark_bar().encode(
                x=alt.X("importance:Q"),
                y=alt.Y("feature:N", sort='-x')
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)

# -------------------------
# Footer / credits
# -------------------------
st.markdown("---")
st.markdown("<div class='muted'>Built with ❤️ for Shadowfox | Designed by your friendly ML mentor</div>", unsafe_allow_html=True)
