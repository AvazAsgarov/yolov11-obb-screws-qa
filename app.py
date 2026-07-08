"""
YOLOv11-OBB QA Screws & Nuts Analyzer (Streamlit Entry Point)

Description:
    Core orchestrator that initializes page configurations, applies custom styles,
    builds the sidebar controls, and routes layout rendering to modular tab scripts.
"""

import streamlit as st
from src.utils import get_inference_device, load_yolo_model, MODEL_PATHS
from src.tabs import render_single_tab, render_compare_tab, render_batch_tab, render_report_tab

# Page Configuration
st.set_page_config(
    page_title="YOLO-OBB Screws and Nuts Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark-themed UI hints, clean spacing, tab styling)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: bold;
        font-size: 16px;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
    div.stButton > button:first-child {
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #27ae60;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 4px solid #3498db;
    }
    .device-badge {
        background-color: #34495e;
        color: #ecf0f1;
        padding: 6px 12px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: bold;
        font-size: 14px;
        text-align: center;
        margin-bottom: 10px;
    }
    /* Compact Sidebar overrides */
    [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }
    [data-testid="stSidebarUserContent"] .element-container {
        margin-bottom: 0.4rem !important;
    }
    [data-testid="stSidebarUserContent"] div[data-testid="stWidgetLabel"] p {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        margin-bottom: 2px !important;
    }
    [data-testid="stSidebarUserContent"] hr {
        margin-top: 0.6rem !important;
        margin-bottom: 0.6rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Hardware Detection
device_id, device_desc = get_inference_device()

# --- SIDEBAR NAV & CONFIGURATION ---
with st.sidebar:
    st.markdown("### Navigation")
    selected_page = st.radio(
        "Select Dashboard View",
        [
            "Single Image Analyzer",
            "Side-by-Side Comparison",
            "Batch Processor & Export",
            "Model Benchmarks & Report"
        ]
    )
    
    st.markdown("---")
    st.markdown("### Runtime Environment")
    st.markdown(f"<div class='device-badge'>Device: {device_desc}</div>", unsafe_allow_html=True)
    
    # Conditional Sidebar Controls based on active page
    if selected_page in ["Single Image Analyzer", "Batch Processor & Export"]:
        st.markdown("### Model Configuration")
        selected_model_name = st.selectbox("Active YOLO-OBB Model", list(MODEL_PATHS.keys()))
        
        # Load Selected Model
        yolo_model = load_yolo_model(selected_model_name)
        if yolo_model is None:
            st.error(f"Model weights for '{selected_model_name}' not found. Please verify the 'models/' directory.")
            st.stop()
            
        st.markdown("### Inference Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.10, 1.00, 0.25, 0.05)
        iou_threshold = st.slider("IoU (NMS) Threshold", 0.10, 0.90, 0.45, 0.05)
        
    elif selected_page == "Side-by-Side Comparison":
        st.markdown("### Comparison Configurations")
        model1_name = st.selectbox("Select Model 1 (Left)", list(MODEL_PATHS.keys()), index=0)
        model2_name = st.selectbox("Select Model 2 (Right)", list(MODEL_PATHS.keys()), index=1)
        
        st.markdown("### Inference Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.10, 1.00, 0.25, 0.05)
        iou_threshold = st.slider("IoU (NMS) Threshold", 0.10, 0.90, 0.45, 0.05)
        
    else:  # Report page
        # Hide all model configuration inputs (static page)
        pass
    
    st.markdown("---")
    st.markdown("### Screws OBB Project Dashboard")
    st.markdown("Developed for high-precision screw class and orientation QA inspection.")

# --- MAIN PAGE HEADLINE ---
st.markdown("# YOLO-OBB QA Screws & Nuts Analyzer")

# --- PAGE ROUTING ---
if selected_page == "Single Image Analyzer":
    render_single_tab(yolo_model, device_id, conf_threshold, iou_threshold)

elif selected_page == "Side-by-Side Comparison":
    render_compare_tab(model1_name, model2_name, conf_threshold, iou_threshold, device_id)

elif selected_page == "Batch Processor & Export":
    render_batch_tab(yolo_model, device_id, conf_threshold, iou_threshold)

elif selected_page == "Model Benchmarks & Report":
    render_report_tab()
