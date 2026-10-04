"""
Streamlit Web Application for Resistor Band Detection & Resistance Calculation.
Provides an interactive AI scanner and a manual color code calculator.
"""

import os
import glob
import streamlit as st
import numpy as np
from PIL import Image
import cv2

from resistance_calculator import (
    calculate_4band_resistance,
    calculate_5band_resistance,
    format_resistance_value,
    COLOR_DIGIT_MAP,
    COLOR_MULTIPLIER_MAP,
    COLOR_TOLERANCE_MAP,
    COLOR_HEX_MAP,
    order_and_calculate_from_detections
)
from predict import ResistorPredictor, COLOR_BGR_MAP

# Configure page
st.set_page_config(
    page_title="Resistor Band AI Detector & Calculator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2.3rem;
        font-weight: 800;
        color: #4ECDC4;
    }
    .metric-subtitle {
        font-size: 0.95rem;
        color: #aaa;
        margin-top: 5px;
    }
    .color-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 2px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .band-step-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #4ECDC4;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return ResistorPredictor(weights_path='weights/resistor_detector.pth')

predictor = load_predictor()

# App Header
st.markdown('<div class="main-header">⚡ Resistor Band AI Detector & Resistance Calculator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">MobileNet CNN Deep Learning Model for Automated Resistor Color Band Detection and Real-Time Resistance Computation</div>', unsafe_allow_html=True)

# Tabs
tab_scanner, tab_calc, tab_docs = st.tabs(["📷 AI Image Scanner", "🎨 Manual Band Calculator", "ℹ️ Model & Color Code Guide"])

# ================= TAB 1: AI IMAGE SCANNER =================
with tab_scanner:
    st.sidebar.header("⚙️ Scanner Settings")
    conf_threshold = st.sidebar.slider("Detection Confidence Threshold", min_value=0.10, max_value=0.95, value=0.30, step=0.05)
    
    col_input, col_results = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("1. Upload or Select Resistor Image")
        input_source = st.radio("Choose Input Method:", ["Upload Image", "Select Sample from Test Dataset"], horizontal=True)
        
        selected_img = None
        
        if input_source == "Upload Image":
            uploaded_file = st.file_uploader("Upload Resistor Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_img = Image.open(uploaded_file)
        else:
            test_images = sorted(glob.glob('resistor_images/test/images/*'))
            if test_images:
                sample_names = [os.path.basename(p) for p in test_images]
                chosen_sample = st.selectbox("Choose a sample resistor:", sample_names, index=0)
                sample_path = os.path.join('resistor_images/test/images', chosen_sample)
                selected_img = Image.open(sample_path)
            else:
                st.info("No test images found in resistor_images/test/images/")

        if selected_img is not None:
            st.image(selected_img, caption="Input Resistor Image", use_column_width=True)

    with col_results:
        st.subheader("2. AI Analysis & Resistance Output")
        
        if selected_img is not None:
            with st.spinner("Analyzing resistor image with MobileNet CNN..."):
                result = predictor.predict(selected_img, score_thresh=conf_threshold)
                
            if result['status'] == 'success':
                calc = result['calculation']
                annotated_rgb = result['annotated_rgb']
                
                # Show Annotated Image
                st.image(annotated_rgb, caption="Detected Bands & Bounding Boxes", use_column_width=True)
                
                # Big Metric Display
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{calc['formatted_value']}</div>
                    <div class="metric-subtitle">Nominal: <b>{calc['nominal_formatted']}</b> | Range: <b>{calc['min_formatted']} – {calc['max_formatted']}</b></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                
                # Band Breakdown
                st.write("### 🏷️ Detected Band Sequence")
                sorted_dets = calc.get('sorted_detections', [])
                
                flip_order = st.checkbox("🔄 Reverse Band Order (Flip Orientation)", value=False)
                if flip_order:
                    sorted_dets = list(reversed(sorted_dets))
                    lbls = [d['label'] for d in sorted_dets]
                    if len(lbls) == 4:
                        calc = calculate_4band_resistance(lbls[0], lbls[1], lbls[2], lbls[3])
                    elif len(lbls) == 5:
                        calc = calculate_5band_resistance(lbls[0], lbls[1], lbls[2], lbls[3], lbls[4])
                
                cols = st.columns(len(sorted_dets))
                for i, (col, det) in enumerate(zip(cols, sorted_dets)):
                    label = det['label']
                    color_hex = COLOR_HEX_MAP.get(label.lower(), '#777777')
                    role = "1st Digit" if i == 0 else ("2nd Digit" if i == 1 else ("Multiplier" if i == len(sorted_dets)-2 else "Tolerance"))
                    with col:
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; text-align: center; border-top: 4px solid {color_hex};">
                            <span style="font-size: 0.8rem; color: #888;">Band {i+1} ({role})</span><br/>
                            <b style="color: {color_hex}; font-size: 1.1rem;">{label}</b><br/>
                            <span style="font-size: 0.75rem; color: #aaa;">Conf: {det['score']:.0%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                st.write("")
                # Detailed Calculation Table
                with st.expander("🔍 View Detailed Calculation Formula"):
                    st.write(f"- **Band 1 (1st Digit):** {calc['bands'][0]} → `{calc['digits'][0]}`")
                    st.write(f"- **Band 2 (2nd Digit):** {calc['bands'][1]} → `{calc['digits'][1]}`")
                    st.write(f"- **Band 3 (Multiplier):** {calc['bands'][2]} → `× {calc['multiplier']:g}`")
                    st.write(f"- **Band 4 (Tolerance):** {calc['bands'][3]} → `±{calc['tolerance_pct']}%`")
                    st.write(f"- **Formula:** `({calc['digits'][0]} × 10 + {calc['digits'][1]}) × {calc['multiplier']} = {calc['nominal_ohms']} Ω`")
            else:
                st.warning("⚠️ No bands detected above the confidence threshold. Try lowering the confidence threshold slider on the left sidebar.")
        else:
            st.info("👈 Upload an image or select a sample from the test set to begin.")

# ================= TAB 2: MANUAL CALCULATOR =================
with tab_calc:
    st.subheader("Interactive Resistor Color Code Calculator")
    st.write("Manually pick band colors to calculate exact resistance, tolerance, and range.")
    
    band_mode = st.radio("Resistor Type:", ["4-Band Resistor", "5-Band Resistor"], horizontal=True)
    
    color_options = ['Black', 'Brown', 'Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Violet', 'Gray', 'White']
    mult_options = ['Black (×1)', 'Brown (×10)', 'Red (×100)', 'Orange (×1k)', 'Yellow (×10k)', 'Green (×100k)', 'Blue (×1M)', 'Gold (×0.1)', 'Silver (×0.01)']
    tol_options = ['Gold (±5%)', 'Silver (±10%)', 'Brown (±1%)', 'Red (±2%)', 'Green (±0.5%)', 'Blue (±0.25%)', 'None (±20%)']
    
    if band_mode == "4-Band Resistor":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            b1 = st.selectbox("1st Band (Digit 1)", color_options, index=2) # Red
        with c2:
            b2 = st.selectbox("2nd Band (Digit 2)", color_options, index=5) # Green
        with c3:
            b3_raw = st.selectbox("3rd Band (Multiplier)", mult_options, index=3) # Orange (x1k)
            b3 = b3_raw.split()[0]
        with c4:
            b4_raw = st.selectbox("4th Band (Tolerance)", tol_options, index=0) # Gold (+-5%)
            b4 = b4_raw.split()[0]
            
        calc_manual = calculate_4band_resistance(b1, b2, b3, b4)
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            b1 = st.selectbox("1st Band", color_options, index=1) # Brown
        with c2:
            b2 = st.selectbox("2nd Band", color_options, index=0) # Black
        with c3:
            b3 = st.selectbox("3rd Band", color_options, index=0) # Black
        with c4:
            b4_raw = st.selectbox("4th Band (Multiplier)", mult_options, index=2) # Red (x100)
            b4 = b4_raw.split()[0]
        with c5:
            b5_raw = st.selectbox("5th Band (Tolerance)", tol_options, index=1) # Brown (+-1%)
            b5 = b5_raw.split()[0]
            
        calc_manual = calculate_5band_resistance(b1, b2, b3, b4, b5)
        
    st.write("")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{calc_manual['formatted_value']}</div>
        <div class="metric-subtitle">Nominal: <b>{calc_manual['nominal_formatted']}</b> ({calc_manual['nominal_ohms']:g} Ω) | Operating Range: <b>{calc_manual['min_formatted']} – {calc_manual['max_formatted']}</b></div>
    </div>
    """, unsafe_allow_html=True)

# ================= TAB 3: MODEL & COLOR GUIDE =================
with tab_docs:
    st.subheader("📚 Resistor Color Code Standard & Architecture")
    
    st.markdown("""
    ### Standard Resistor Color Code Table
    | Color | Digit | Multiplier | Tolerance |
    |---|---|---|---|
    | **Black** | 0 | $10^0$ (1) | — |
    | **Brown** | 1 | $10^1$ (10) | ±1% |
    | **Red** | 2 | $10^2$ (100) | ±2% |
    | **Orange** | 3 | $10^3$ (1 kΩ) | — |
    | **Yellow** | 4 | $10^4$ (10 kΩ) | — |
    | **Green** | 5 | $10^5$ (100 kΩ) | ±0.5% |
    | **Blue** | 6 | $10^6$ (1 MΩ) | ±0.25% |
    | **Violet** | 7 | $10^7$ (10 MΩ) | ±0.1% |
    | **Gray** | 8 | $10^8$ (100 MΩ) | ±0.05% |
    | **White** | 9 | $10^9$ (1 GΩ) | — |
    | **Gold / Side-Gold** | — | $10^{-1}$ (0.1) | ±5% |
    | **Silver / Side-Silver**| — | $10^{-2}$ (0.01)| ±10% |
    """)
    
    st.markdown("""
    ### Model & Inference Pipeline:
    1. **MobileNet CNN Backbone**: Efficient lightweight feature extractor suitable for mobile, embedded, and CPU deployment.
    2. **Multi-Scale Feature Pyramid**: Detects color bands of varying widths and aspect ratios.
    3. **Spatial Band Ordering**: Projects detected bounding box centroids along the primary orientation axis (horizontal/vertical).
    4. **Orientation Correction**: Automatically detects tolerance bands (`Side-Gold` / `Side-Silver`) to determine whether the resistor is oriented left-to-right or right-to-left.
    5. **Resistance Calculator**: Applies standard IEC 60062 / EIA-RS-279 electronic color code formulas.
    """)
