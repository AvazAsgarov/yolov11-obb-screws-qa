"""
Plotting Script for YOLO OBB Inference Speed Benchmarks

Description:
    Generates a high-quality grouped bar chart comparing the latency (ms) and
    throughput (FPS) of YOLO11n-OBB and YOLO11s-OBB models across CPU and GPU.
    Saves the output to 'results/plots/inference_speed_benchmark.png'.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure results directory exists
plots_dir = os.path.join("results", "plots")
os.makedirs(plots_dir, exist_ok=True)
out_path = os.path.join(plots_dir, "inference_speed_benchmark.png")

# Data
models = ["YOLO11n-OBB", "YOLO11s-OBB"]
cpu_latencies = [126.18, 306.38]
gpu_latencies = [15.01, 14.26]

cpu_fps = [7.9, 3.3]
gpu_fps = [66.6, 70.1]

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(models))
width = 0.35

# Color palette
cpu_color = "#34495e"  # Dark Slate Blue
gpu_color = "#2ecc71"  # Emerald Green

# 1. Latency Plot (Lower is Better)
rects1 = ax1.bar(x - width/2, cpu_latencies, width, label="CPU", color=cpu_color, edgecolor="black", alpha=0.9)
rects2 = ax1.bar(x + width/2, gpu_latencies, width, label="GPU (A100)", color=gpu_color, edgecolor="black", alpha=0.9)

ax1.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
ax1.set_title("Inference Latency\n(Lower is Better)", fontsize=14, fontweight="bold", pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11, fontweight="bold")
ax1.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.6)

# Add values on top of bars
def autolabel_latency(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.2f} ms",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

autolabel_latency(rects1, ax1)
autolabel_latency(rects2, ax1)

# 2. FPS Plot (Higher is Better)
rects3 = ax2.bar(x - width/2, cpu_fps, width, label="CPU", color=cpu_color, edgecolor="black", alpha=0.9)
rects4 = ax2.bar(x + width/2, gpu_fps, width, label="GPU (A100)", color=gpu_color, edgecolor="black", alpha=0.9)

ax2.set_ylabel("Throughput (FPS)", fontsize=12, fontweight="bold")
ax2.set_title("Throughput (FPS)\n(Higher is Better)", fontsize=14, fontweight="bold", pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=11, fontweight="bold")
ax2.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.6)

# Add values on top of bars
def autolabel_fps(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.1f} FPS",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

autolabel_fps(rects3, ax2)
autolabel_fps(rects4, ax2)

# Adjust layout and save
plt.suptitle("YOLOv11-OBB Inference Speed Comparison: CPU vs. GPU", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f"[OK] Saved speed benchmark bar chart to: '{out_path}'")
