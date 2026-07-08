"""
UI Tabs Components for YOLOv11-OBB Screws Dashboard

Description:
    Contains separate UI layout and execution functions for each of
    the four Streamlit dashboard tabs.
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from src.drawing import draw_obb_predictions
from src.utils import load_yolo_model, MODEL_PATHS

def render_single_tab(yolo_model, device_id, conf_threshold, iou_threshold):
    """Renders the Single Image Analyzer tab layout and logic."""
    st.markdown("### Single Image Analysis")
    uploaded_file = st.file_uploader("Upload inspection image...", type=["png", "jpg", "jpeg", "webp"], key="single_upload")
    
    if uploaded_file is not None:
        # Load and convert image
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        
        # Run prediction
        start_time = time.time()
        results = yolo_model.predict(img_np, conf=conf_threshold, iou=iou_threshold, imgsz=1024, device=device_id, verbose=False)
        latency = (time.time() - start_time) * 1000
        
        # Draw bounding boxes and arrows
        annotated_img, detections = draw_obb_predictions(img_np, results, conf_threshold)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.image(annotated_img, caption="Analyzed Bounding Boxes & Orientations", use_column_width=True)
            
        with col2:
            st.markdown("#### Performance Metrics")
            st.markdown(f"<div class='metric-card'><b>Inference Latency:</b> {latency:.1f} ms<br><b>Detected Objects:</b> {len(detections)}</div>", unsafe_allow_html=True)
            
            if len(detections) > 0:
                st.markdown("#### Detections Summary")
                df_det = pd.DataFrame(detections).drop(columns=["corners"])
                st.dataframe(df_det, use_container_width=True, hide_index=True)
                
                # Crop Viewer Section
                st.markdown("#### Crop Inspector")
                selected_id = st.selectbox("Select Object ID to inspect", df_det["Object ID"].tolist())
                
                if selected_id:
                    # Find corresponding corners
                    selected_det = next(item for item in detections if item["Object ID"] == selected_id)
                    corners = selected_det["corners"]
                    
                    # Axis-aligned crop with padding margin
                    margin = 15
                    ymin = max(0, int(np.min(corners[:, 1])) - margin)
                    ymax = min(img_np.shape[0], int(np.max(corners[:, 1])) + margin)
                    xmin = max(0, int(np.min(corners[:, 0])) - margin)
                    xmax = min(img_np.shape[1], int(np.max(corners[:, 0])) + margin)
                    
                    crop_img = img_np[ymin:ymax, xmin:xmax]
                    
                    c_col1, c_col2 = st.columns([1, 1])
                    with c_col1:
                        st.image(crop_img, caption=f"ID {selected_id} Crop View", use_column_width=True)
                    with c_col2:
                        st.markdown(f"""
                            *   **Class:** {selected_det['Class Name']}
                            *   **Confidence:** {selected_det['Confidence']:.2%}
                            *   **Angle:** {selected_det['Angle (Deg)']}° ({selected_det['Angle (Rad)']} rad)
                            *   **Dimensions:** {selected_det['Width']} px x {selected_det['Height']} px
                        """)
            else:
                st.info("No objects detected above the confidence threshold.")

def render_compare_tab(model1_name, model2_name, conf_threshold, iou_threshold, device_id):
    """Renders the Side-by-Side Model Comparison tab layout and logic."""
    st.markdown("### Real-Time Side-by-Side Model Comparison")
    
    comp_file = st.file_uploader("Upload image to compare...", type=["png", "jpg", "jpeg", "webp"], key="comp_upload")
    
    if comp_file is not None:
        image = Image.open(comp_file).convert("RGB")
        img_np = np.array(image)
        
        # Load both models
        m1 = load_yolo_model(model1_name)
        m2 = load_yolo_model(model2_name)
        
        if m1 is None or m2 is None:
            st.error("One or both selected models could not be loaded. Please verify the 'models/' directory.")
            return
            
        if m1 and m2:
            # Model 1 Inference
            start_m1 = time.time()
            res_m1 = m1.predict(img_np, conf=conf_threshold, iou=iou_threshold, imgsz=1024, device=device_id, verbose=False)
            lat_m1 = (time.time() - start_m1) * 1000
            ann_m1, det_m1 = draw_obb_predictions(img_np, res_m1, conf_threshold)
            
            # Model 2 Inference
            start_m2 = time.time()
            res_m2 = m2.predict(img_np, conf=conf_threshold, iou=iou_threshold, imgsz=1024, device=device_id, verbose=False)
            lat_m2 = (time.time() - start_m2) * 1000
            ann_m2, det_m2 = draw_obb_predictions(img_np, res_m2, conf_threshold)
            
            # Display comparison side-by-side
            c_col1, c_col2 = st.columns(2)
            
            with c_col1:
                st.markdown(f"#### {model1_name}")
                st.markdown(f"<div class='metric-card' style='border-left-color: #2ecc71;'><b>Latency:</b> {lat_m1:.1f} ms<br><b>Objects Count:</b> {len(det_m1)}</div>", unsafe_allow_html=True)
                st.image(ann_m1, use_column_width=True)
                if len(det_m1) > 0:
                    df1 = pd.DataFrame(det_m1).drop(columns=["corners", "Angle (Rad)"])
                    st.dataframe(df1, use_container_width=True, hide_index=True)
                    
            with c_col2:
                st.markdown(f"#### {model2_name}")
                st.markdown(f"<div class='metric-card' style='border-left-color: #9b59b6;'><b>Latency:</b> {lat_m2:.1f} ms<br><b>Objects Count:</b> {len(det_m2)}</div>", unsafe_allow_html=True)
                st.image(ann_m2, use_column_width=True)
                if len(det_m2) > 0:
                    df2 = pd.DataFrame(det_m2).drop(columns=["corners", "Angle (Rad)"])
                    st.dataframe(df2, use_container_width=True, hide_index=True)

def render_batch_tab(yolo_model, device_id, conf_threshold, iou_threshold):
    """Renders the Batch Processor & CSV Export tab layout and logic."""
    st.markdown("### Batch Processing Pipeline")
    uploaded_files = st.file_uploader("Upload multiple images to process...", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="batch_upload")
    
    if uploaded_files:
        st.markdown(f"[INFO] Loaded {len(uploaded_files)} images. Press button to run batch inference.")
        
        if st.button("Run Batch Inference"):
            batch_detections = []
            cols_gallery = st.columns(4)  # 4 thumbnails per row
            
            for idx, file in enumerate(uploaded_files):
                image = Image.open(file).convert("RGB")
                img_np = np.array(image)
                
                # Run prediction
                res = yolo_model.predict(img_np, conf=conf_threshold, iou=iou_threshold, imgsz=1024, device=device_id, verbose=False)
                ann_img, det = draw_obb_predictions(img_np, res, conf_threshold)
                
                # Accumulate for CSV export
                for d in det:
                    batch_detections.append({
                        "Image Name": file.name,
                        "Object ID": d["Object ID"],
                        "Class Name": d["Class Name"],
                        "Confidence": d["Confidence"],
                        "Center X": d["Center X"],
                        "Center Y": d["Center Y"],
                        "Width": d["Width"],
                        "Height": d["Height"],
                        "Angle (Deg)": d["Angle (Deg)"]
                    })
                    
                # Render gallery thumbnail
                col_idx = idx % 4
                with cols_gallery[col_idx]:
                    st.image(ann_img, caption=file.name, use_column_width=True)
            
            # Show summary and export button if anything is found
            if batch_detections:
                st.markdown("#### Consolidated Batch Results")
                df_batch = pd.DataFrame(batch_detections)
                st.dataframe(df_batch, use_container_width=True)
                
                # Convert DataFrame to CSV for download
                csv_data = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results Table as CSV",
                    data=csv_data,
                    file_name="screws_batch_detections.csv",
                    mime="text/csv"
                )
            else:
                st.info("No objects detected in any of the uploaded images.")

def render_report_tab():
    """Renders the Benchmarks & Report tab displaying training stats and charts."""
    st.markdown("### Model Benchmarks & Development Training Report")
    
    st.markdown("""
    #### 1. Performance Metrics Summary (Test Split @ 1024x1024 px)
    
    | Model | Variant | Epochs | Input Size | Precision | Recall | mAP50 | mAP50-95 |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | **YOLO11n-OBB** (Nano) | Baseline | 50 | 1024 | 99.0% | 98.6% | 99.1% | 93.2% |
    | **YOLO11n-OBB** (Nano) | Optimized (Tuned) | 100 | 1024 | **98.7%** | **99.1%** | **99.4%** | **94.1%** |
    | **YOLO11s-OBB** (Small) | Baseline | 50 | 1024 | 99.2% | 99.5% | 99.2% | 94.5% |
    | **YOLO11s-OBB** (Small) | Optimized (Tuned) | 100 | 1024 | **99.5%** | **99.4%** | **99.2%** | **95.4%** |
    """)
    
    # Load and display Speed Benchmark plot
    speed_plot_path = "results/plots/inference_speed_benchmark.png"
    if os.path.exists(speed_plot_path):
        st.markdown("#### 2. Real-Time Inference Speed Bar Chart")
        st.image(speed_plot_path, caption="Throughput (FPS) & Latency (ms) on CPU vs GPU", use_column_width=True)
        
    # Load and display training curves and confusion matrix side-by-side
    st.markdown("#### 3. Diagnostic & Training History Curves")
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        s_curves_path = "results/plots/yolo11s_optimized_results.png"
        if os.path.exists(s_curves_path):
            st.image(s_curves_path, caption="Optimized YOLO11s-OBB Training History Loss & mAP Curves", use_column_width=True)
            
    with col_c2:
        s_conf_path = "results/plots/yolo11s_optimized_confusion_matrix_normalized.png"
        if os.path.exists(s_conf_path):
            st.image(s_conf_path, caption="Optimized YOLO11s-OBB Confusion Matrix (Normalized)", use_column_width=True)

    # Load and display xAI feature map activation overlay
    st.markdown("#### 4. Model Explainability (xAI) - Feature Map Heatmap")
    xai_path = "results/plots/results_gradcam.png"
    if os.path.exists(xai_path):
        st.image(xai_path, caption="Channel-Averaged Feature Activations from SPPF layer overlay (xAI)", use_column_width=True)
