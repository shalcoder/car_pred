import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# Configuration
API_URL = "https://car-pred-le3d.onrender.com/predict"
APP_TITLE = "Shadowfox Car Resale Price Estimator"
APP_SUB = "Intelligent vehicle valuation powered by machine learning"

DOWNLOAD_CODE_PATH = "E:/shadowfox/phase2/car_pred/streamlit_app/app.py"

st.set_page_config(
    page_title="Shadowfox Car Price Estimator",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    :root {
        --primary-blue: #0066CC;
        --primary-dark: #004C99;
        --secondary-slate: #475569;
        --accent-teal: #0891B2;
        --bg-card: rgba(255, 255, 255, 0.03);
        --bg-elevated: rgba(255, 255, 255, 0.06);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --text-primary: #F8FAFC;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
        --success: #10B981;
        --warning: #F59E0B;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    }
    
    .app-header {
        background: linear-gradient(135deg, rgba(0, 102, 204, 0.08) 0%, rgba(8, 145, 178, 0.05) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 32px 28px;
        margin-bottom: 28px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary-blue), transparent);
    }
    
    .header-title {
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.8px;
        color: var(--text-primary);
        margin: 0;
        line-height: 1.2;
    }
    
    .header-subtitle {
        font-size: 15px;
        color: var(--text-secondary);
        margin-top: 8px;
        font-weight: 400;
        letter-spacing: 0.2px;
    }
    
    .header-meta {
        text-align: right;
        font-size: 13px;
        color: var(--text-muted);
        font-weight: 500;
    }
    
    .glass-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 102, 204, 0.3);
        box-shadow: 0 8px 32px rgba(0, 102, 204, 0.15);
    }
    
    .card-title {
        font-size: 20px;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 20px 0;
        letter-spacing: -0.3px;
    }
    
    .price-display {
        background: linear-gradient(135deg, rgba(0, 102, 204, 0.15), rgba(8, 145, 178, 0.1));
        border: 1px solid rgba(0, 102, 204, 0.3);
        border-radius: 12px;
        padding: 32px 24px;
        text-align: center;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }
    
    .price-display::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 102, 204, 0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .price-value {
        font-size: 48px;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
    }
    
    .price-currency {
        font-size: 24px;
        color: var(--text-secondary);
        margin-right: 4px;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-top: 24px;
    }
    
    .metric-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 12px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 20px;
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .insight-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .insight-item {
        padding: 12px 16px;
        margin-bottom: 8px;
        background: var(--bg-elevated);
        border-left: 3px solid var(--primary-blue);
        border-radius: 6px;
        color: var(--text-secondary);
        font-size: 14px;
        line-height: 1.6;
        transition: all 0.2s ease;
    }
    
    .insight-item:hover {
        background: rgba(255, 255, 255, 0.08);
        border-left-color: var(--accent-teal);
    }
    
    .divider {
        height: 1px;
        background: var(--border-subtle);
        margin: 24px 0;
        border: none;
    }
    
    .footer-text {
        font-size: 13px;
        color: var(--text-muted);
        margin-top: 12px;
        line-height: 1.6;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.15);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.15);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* Streamlit Component Overrides */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary-blue), var(--primary-dark));
        color: white;
        font-weight: 600;
        padding: 14px 24px;
        border-radius: 8px;
        border: none;
        font-size: 15px;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-dark), var(--accent-teal));
        box-shadow: 0 6px 20px rgba(0, 102, 204, 0.4);
        transform: translateY(-2px);
    }
    
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        color: var(--text-primary);
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 14px;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid var(--border-subtle);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }
    
    .element-container {
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
with st.container():
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    cols = st.columns([0.75, 0.25])
    with cols[0]:
        st.markdown('<h1 class="header-title">Shadowfox Car Resale Price Estimator</h1>', unsafe_allow_html=True)
        st.markdown('<p class="header-subtitle">Intelligent vehicle valuation powered by machine learning algorithms</p>', unsafe_allow_html=True)
    with cols[1]:
        current_time = datetime.now().strftime('%B %d, %Y')
        st.markdown(f"<div class='header-meta'>Updated<br/>{current_time}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar Configuration
sidebar = st.sidebar
sidebar.markdown("### Vehicle Information")
sidebar.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

present_price = sidebar.number_input(
    "Current Market Price (lakhs)", 
    min_value=0.0, 
    value=5.59, 
    step=0.1, 
    format="%.2f",
    help="Enter the current market price of the vehicle"
)

kms_driven = sidebar.number_input(
    "Kilometers Driven", 
    min_value=0, 
    value=27000, 
    step=500,
    help="Total distance traveled by the vehicle"
)

year = sidebar.slider(
    "Year of Manufacture", 
    1990, 
    2025, 
    2014,
    help="Select the year the vehicle was manufactured"
)

brand = sidebar.text_input(
    "Brand Name", 
    value="Maruti",
    help="Enter the vehicle manufacturer name"
)

sidebar.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
sidebar.markdown("### Vehicle Specifications")
sidebar.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

fuel_type = sidebar.selectbox(
    "Fuel Type", 
    ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid"],
    help="Select the type of fuel used"
)

seller_type = sidebar.selectbox(
    "Seller Type", 
    ["Dealer", "Individual"],
    help="Type of seller offering the vehicle"
)

transmission = sidebar.selectbox(
    "Transmission Type", 
    ["Manual", "Automatic"],
    help="Gearbox transmission type"
)

owner = sidebar.selectbox(
    "Number of Previous Owners", 
    [0, 1, 2, 3],
    help="How many owners has this vehicle had"
)

sidebar.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

show_engineered = sidebar.checkbox(
    "Display Feature Engineering", 
    value=False,
    help="Show the calculated features used by the model"
)

sidebar.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
run_button = sidebar.button("Calculate Resale Value")

# Main Content Layout
left_col, right_col = st.columns([0.6, 0.4])

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Predicted Resale Value</h2>", unsafe_allow_html=True)
    
    price_placeholder = st.empty()
    
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
    st.markdown("<h3 class='card-title'>Model Performance Metrics</h3>", unsafe_allow_html=True)
    
    metrics_placeholder = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Key Insights</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <ul class='insight-list'>
        <li class='insight-item'><strong>Depreciation Rate:</strong> Primary factor affecting vehicle value over time</li>
        <li class='insight-item'><strong>Vehicle Age:</strong> Strong inverse correlation with resale price</li>
        <li class='insight-item'><strong>Seller Premium:</strong> Dealer listings typically command higher valuations</li>
        <li class='insight-item'><strong>Usage Pattern:</strong> Annual mileage impacts overall condition assessment</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
    
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Logic
def call_predict_api(payload: dict):
    """Call the prediction API endpoint with error handling"""
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

if run_button:
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

    with st.spinner("Calculating resale value using machine learning model..."):
        t0 = pd.Timestamp.now()
        result = call_predict_api(payload)
        t1 = pd.Timestamp.now()
        elapsed_time = (t1 - t0).total_seconds()

    if result is None or "error" in result:
        error_msg = result.get("error", "Unknown error occurred") if isinstance(result, dict) else "API connection failed"
        st.error(f"Error: {error_msg}")
    else:
        predicted_price = result.get("predicted_price", 0.0)
        model_name = result.get("model_used", "Unknown Model")
        
        # Display Price
        price_placeholder.markdown(f"""
        <div class='price-display'>
            <div class='price-value'>
                <span class='price-currency'>₹</span>{predicted_price:.2f}
            </div>
            <div style='margin-top: 8px; color: var(--text-secondary); font-size: 14px;'>lakhs (Indian Rupees)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Metrics
        confidence_level = "High" if predicted_price > 0 else "Low"
        status_class = "status-success" if confidence_level == "High" else "status-warning"
        
        metrics_placeholder.markdown(f"""
        <div class='metric-grid'>
            <div class='metric-card'>
                <div class='metric-label'>Model Type</div>
                <div class='metric-value'>{model_name}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Response Time</div>
                <div class='metric-value'>{elapsed_time:.3f}s</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Confidence</div>
                <div class='metric-value'>
                    <span class='status-badge {status_class}'>{confidence_level}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display Engineered Features
        if show_engineered:
            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h3 class='card-title'>Engineered Features</h3>", unsafe_allow_html=True)
            
            engineered = result.get("engineered_features")
            if engineered is None:
                vehicle_age = 2025 - payload["year"]
                km_per_year = payload["kms_driven"] / max(vehicle_age, 1)
                price_depreciation = payload["present_price"] / max(vehicle_age, 1)
                car_condition = (payload["present_price"] / (payload["kms_driven"] + 1)) * (1.0 / (vehicle_age + 1))
                
                engineered = {
                    "Present_Price": payload["present_price"],
                    "Kms_Driven": payload["kms_driven"],
                    "Age": vehicle_age,
                    "KM_per_Year": round(km_per_year, 2),
                    "Price_Depreciation": round(price_depreciation, 4),
                    "Car_Condition": round(car_condition, 8),
                    "Is_First_Owner": int(payload["owner"] == 0),
                    "Is_Diesel": int(payload["fuel_type"].lower() == "diesel"),
                    "Brand": payload["brand"]
                }
            
            st.json(engineered)
            st.markdown('</div>', unsafe_allow_html=True)

        # Display Feature Importance
        feat_imp = result.get("feature_importance")
        if feat_imp:
            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h3 class='card-title'>Feature Importance Analysis</h3>", unsafe_allow_html=True)
            
            df_imp = pd.DataFrame(feat_imp)
            chart = alt.Chart(df_imp).mark_bar(color='#0066CC').encode(
                x=alt.X("importance:Q", title="Importance Score"),
                y=alt.Y("feature:N", sort='-x', title="Feature"),
                tooltip=['feature', 'importance']
            ).properties(height=320).configure_axis(
                labelColor='#CBD5E1',
                titleColor='#F8FAFC'
            ).configure_view(
                strokeWidth=0
            )
            
            st.altair_chart(chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: var(--text-muted); font-size: 13px; padding: 20px 0;'>
    <strong>Shadowfox Technology Solutions</strong><br/>
    Advanced Machine Learning for Automotive Valuation
</div>
""", unsafe_allow_html=True)