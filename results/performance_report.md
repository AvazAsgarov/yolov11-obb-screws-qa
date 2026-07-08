# MVTec Screws & Nuts oriented Object Detection (OBB)
## YOLOv11-OBB Performance & Optimization Report

This report evaluates and compares **YOLOv11n-OBB** (Nano) and **YOLOv11s-OBB** (Small) models trained on the preprocessed **MVTec Screws** dataset. Models were trained both with default configurations (baseline) and optimized hyperparameters discovered via Genetic Algorithm tuning.

---

## 1. Performance Metrics Summary

All models were evaluated on the validation and test splits using an oriented bounding box (OBB) metric at a resolution of **1024x1024** pixels.

### Overall Performance Comparison (Test Set)

| Model | Variant | Epochs | Input Size | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv11n-OBB** (Nano) | Baseline | 50 | 1024 | 99.0% | 98.6% | 99.1% | 93.2% |
| **YOLOv11n-OBB** (Nano) | Optimized (Tuned) | 100 | 1024 | **98.7%** | **99.1%** | **99.4%** | **94.1%** |
| **YOLOv11s-OBB** (Small) | Baseline | 50 | 1024 | 99.2% | 99.5% | 99.2% | 94.5% |
| **YOLOv11s-OBB** (Small) | Optimized (Tuned) | 100 | 1024 | **99.5%** | **99.4%** | **99.2%** | **95.4%** |

*Note: The OBB mAP50-95 metric represents the mean Average Precision computed across multiple Intersection-over-Union (IoU) thresholds from 0.50 to 0.95, which highly penalizes errors in both box boundaries and rotation angles (orientations).*

### Key Observations
1. **Hyperparameter Tuning Impact:** The genetic hyperparameter search combined with longer training (100 epochs) improved the crucial `mAP50-95` metric by **+0.9%** for both Nano (93.2% $\rightarrow$ 94.1%) and Small (94.5% $\rightarrow$ 95.4%) variants.
2. **Nano vs. Small:** The Small variant (`yolo11s-obb`) achieved the best overall performance with **99.5% Precision, 99.4% Recall, and 95.4% mAP50-95**, resolving complex orientation boundaries on overlapping or tiny parts slightly better than the Nano model.

---

## 2. Genetic Hyperparameter Tuning Results

The genetic tuner ran for **15 iterations of 10 epochs each** using `YOLOv11n-OBB` to find the optimal balance of regularization and data augmentations. The best hyperparameters were found during **iteration 7** (Fitness Score: `0.49173`).

These optimized values were used for the final 100-epoch training runs:

| Hyperparameter | Tuned Value | Description |
| :--- | :--- | :--- |
| `lr0` | `0.01` | Initial learning rate |
| `lrf` | `0.01` | Final learning rate fraction |
| `momentum` | `0.94077` | Optimizer momentum |
| `weight_decay` | `0.00034` | L2 weight regularization decay |
| `warmup_epochs` | `3.01076` | Epochs spent in warmup phase |
| `box` | `7.47175` | Box loss gain factor |
| `cls` | `0.4793` | Class loss gain factor |
| `dfl` | `1.53551` | Distribution focal loss gain |
| `degrees` | `0.01119` | Maximum rotation angle (degrees) |
| `translate` | `0.13523` | Translation augmentation range |
| `scale` | `0.50521` | Scale scaling range |
| `mosaic` | `0.9287` | Probability of applying mosaic augmentation |
| `fliplr` | `0.47882` | Horizontal flip probability |
| `mixup` | `0.00038` | Mixup augmentation probability |

---

## 3. Real-Time Inference Speed Benchmark

Speed benchmarks were measured on test images of size **1024x1024** pixels.

| Model | Device | Preprocessing (ms) | Inference (ms) | Postprocessing (ms) | Total Latency (ms) | FPS (Frames/Sec) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv11n-OBB** (Nano) | CPU | 7.30 ms | 126.18 ms | 24.10 ms | 157.58 ms | 6.3 |
| **YOLOv11n-OBB** (Nano) | GPU (A100) | 0.20 ms | 15.01 ms | 1.90 ms | 17.11 ms | 58.4 |
| **YOLOv11s-OBB** (Small) | CPU | 7.30 ms | 306.38 ms | 24.10 ms | 337.78 ms | 2.9 |
| **YOLOv11s-OBB** (Small) | GPU (A100) | 0.20 ms | 14.26 ms | 2.00 ms | 16.46 ms | 60.7 |

*Note: On GPU, parallel execution and Tensor Core acceleration allow the Small model to achieve throughput (~60 FPS) comparable to the Nano model, while the Nano model is over 2x faster when deployed on low-power CPU environments.*

![Inference Speed Grouped Bar Chart](plots/inference_speed_benchmark.png)

---

## 4. Diagnostic & Training Visualizations

All generated figures, curves, and validation batches are stored in the [results/plots/](plots/) folder of the repository.

### Training History (Loss & mAP Curves)
*   **YOLOv11n-OBB (Nano) Optimized:** [yolo11n_optimized_results.png](plots/yolo11n_optimized_results.png)
*   **YOLOv11s-OBB (Small) Optimized:** [yolo11s_optimized_results.png](plots/yolo11s_optimized_results.png)
*   **YOLOv11n-OBB (Nano) Baseline:** [yolo11n_baseline_results.png](plots/yolo11n_baseline_results.png)
*   **YOLOv11s-OBB (Small) Baseline:** [yolo11s_baseline_results.png](plots/yolo11s_baseline_results.png)

### Confusion Matrices
Confusion matrices evaluate class-wise confusion on the validation split:
*   **YOLOv11s-OBB Optimized (Normalized):** [yolo11s_optimized_confusion_matrix_normalized.png](plots/yolo11s_optimized_confusion_matrix_normalized.png) | [Standard](plots/yolo11s_optimized_confusion_matrix.png)
*   **YOLOv11n-OBB Optimized (Normalized):** [yolo11n_optimized_confusion_matrix_normalized.png](plots/yolo11n_optimized_confusion_matrix_normalized.png) | [Standard](plots/yolo11n_optimized_confusion_matrix.png)

### Precision-Recall (PR) and F1-Curves
*   **YOLOv11s-OBB Optimized Box PR Curve:** [yolo11s_optimized_BoxPR_curve.png](plots/yolo11s_optimized_BoxPR_curve.png)
*   **YOLOv11s-OBB Optimized Box F1 Curve:** [yolo11s_optimized_BoxF1_curve.png](plots/yolo11s_optimized_BoxF1_curve.png)
*   **YOLOv11n-OBB Optimized Box PR Curve:** [yolo11n_optimized_BoxPR_curve.png](plots/yolo11n_optimized_BoxPR_curve.png)
*   **YOLOv11n-OBB Optimized Box F1 Curve:** [yolo11n_optimized_BoxF1_curve.png](plots/yolo11n_optimized_BoxF1_curve.png)

### Ground Truth vs. Model Prediction Samples
View how predictions match the ground-truth orientation on validation batches:
*   **Ground Truth Labels (Batch 0):** [yolo11s_optimized_val_batch0_labels.jpg](plots/yolo11s_optimized_val_batch0_labels.jpg)
*   **OBB Predictions (Batch 0):** [yolo11s_optimized_val_batch0_pred.jpg](plots/yolo11s_optimized_val_batch0_pred.jpg)

### Model Explainability (xAI) Heatmap
Spatial activation overlays from the SPPF layer (Backbone Index 9) demonstrating what regions the model focuses on to determine screw thread orientations:
*   **xAI Heatmap Overlay:** [results_gradcam.png](plots/results_gradcam.png)
