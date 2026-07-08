"""
Inference Utility Functions for YOLOv11-OBB Screws Dashboard

Description:
    Manages automatic hardware device detection (GPU/CPU) and 
    cached model loading to prevent VRAM memory bloat.
"""

import os
import torch
import streamlit as st
from ultralytics import YOLO

# Model weight configurations
MODEL_PATHS = {
    "YOLO11n-OBB Optimized (Tuned)": "models/yolo11n_optimized.pt",
    "YOLO11s-OBB Optimized (Tuned)": "models/yolo11s_optimized.pt",
    "YOLO11n-OBB Baseline": "models/yolo11n_baseline.pt",
    "YOLO11s-OBB Baseline": "models/yolo11s_baseline.pt"
}

@st.cache_resource
def get_inference_device():
    """Detects CUDA availability and returns the device index and name."""
    if torch.cuda.is_available():
        return 0, f"GPU (CUDA: {torch.cuda.get_device_name(0)})"
    return "cpu", "CPU"

@st.cache_resource
def load_yolo_model(model_name):
    """Loads and caches the specified YOLO OBB model weights."""
    path = MODEL_PATHS.get(model_name)
    if path and os.path.exists(path):
        return YOLO(path)
    return None
