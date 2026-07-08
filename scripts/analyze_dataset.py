"""
MVTec Screws Dataset Exploratory Data Analysis (EDA) Script

Description:
    This script performs detailed exploratory analysis on the MVTec Screws dataset
    by reading annotations directly from the 'mvtec_screws_v1.1.tar.gz' archive
    in-memory. This prevents extracting large image assets and speeds up execution.
    
    It generates the following visualizations and logs:
    1. Class distribution bar chart across splits (train/val/test) -> 'results/class_distribution.png'
    2. Bounding box area distribution histogram (scale analysis) -> 'results/box_area_distribution.png'
    3. Box aspect ratio distribution histogram (shape analysis) -> 'results/aspect_ratio_distribution.png'
    4. Box orientation polar histogram (rotation analysis) -> 'results/orientation_distribution.png'
    5. Object count density per image histogram (crowdedness analysis) -> 'results/objects_per_image.png'
    6. Terminal text report of class distribution.
"""

import os
import tarfile
import json
import numpy as np
import matplotlib.pyplot as plt

# Resolve project root relative to script file location for robustness
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

TAR_PATH = os.path.join(PROJECT_ROOT, "mvtec_screws_v1.1.tar.gz")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def analyze_dataset():
    """Load JSON annotation files from the tar archive and analyze statistics."""
    if not os.path.exists(TAR_PATH):
        raise FileNotFoundError(f"Dataset archive '{TAR_PATH}' not found in the project directory.")
        
    print(f"Reading annotations directly from '{os.path.basename(TAR_PATH)}'...")
    
    splits = ["train", "val", "test"]
    class_stats = {split: {} for split in splits}
    class_names = {}
    
    # Lists for aggregated analysis metrics
    all_areas = []
    all_aspect_ratios = []
    all_angles = []
    all_counts_per_image = []
    
    # Open tarball and extract JSON files in-memory
    with tarfile.open(TAR_PATH, "r:gz") as tf:
        for split in splits:
            member_name = f"v1.1/mvtec_screws_{split}.json"
            try:
                member = tf.getmember(member_name)
            except KeyError:
                print(f"[Warning] Annotation '{member_name}' not found in archive.")
                continue
                
            f = tf.extractfile(member)
            coco_data = json.load(f)
            
            # Map category IDs to names (0-indexed for YOLO)
            for cat in coco_data["categories"]:
                cat_id = cat["id"]
                cat_name = cat["name"]
                class_names[cat_id - 1] = cat_name
                
            # Group annotations by image_id to calculate objects per image
            annotations_by_img = {}
            for ann in coco_data["annotations"]:
                img_id = ann["image_id"]
                annotations_by_img.setdefault(img_id, []).append(ann)
                
                # Gather box scale and shape metrics
                row, col, w, h, phi = ann["bbox"]
                all_areas.append(w * h)
                all_aspect_ratios.append(w / h if h > 0 else 1.0)
                all_angles.append(phi)
                
                cat_id = ann["category_id"]
                class_idx = cat_id - 1
                class_stats[split][class_idx] = class_stats[split].get(class_idx, 0) + 1
                
            # Record object count per image for this split
            for img in coco_data["images"]:
                img_id = img["id"]
                count = len(annotations_by_img.get(img_id, []))
                all_counts_per_image.append(count)
                
    # Create results folder if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate all plots
    plot_class_distribution(class_names, class_stats)
    plot_advanced_metrics(all_areas, all_aspect_ratios, all_angles, all_counts_per_image)
    
    # Print console report
    print("\n" + "="*50)
    print("               DATA DISTRIBUTION REPORT")
    print("="*50)
    print(f"{'Class Name':<12} | {'Class ID':<8} | {'Train':<7} | {'Val':<7} | {'Test':<7}")
    print("-"*50)
    for idx in sorted(class_names.keys()):
        c_name = class_names[idx]
        train_cnt = class_stats["train"].get(idx, 0)
        val_cnt = class_stats["val"].get(idx, 0)
        test_cnt = class_stats["test"].get(idx, 0)
        print(f"{c_name:<12} | {idx:<8} | {train_cnt:<7} | {val_cnt:<7} | {test_cnt:<7}")
    print("="*50)
    print(f"Total processed classes: {len(class_names)}")
    print("="*50 + "\n")

def plot_class_distribution(class_names, class_stats):
    """Generate and save grouped bar chart showing split distribution."""
    print("Plotting class distribution...")
    classes = sorted(class_names.keys())
    x_labels = [class_names[c] for c in classes]
    
    train_counts = [class_stats["train"].get(c, 0) for c in classes]
    val_counts = [class_stats["val"].get(c, 0) for c in classes]
    test_counts = [class_stats["test"].get(c, 0) for c in classes]
    
    x = np.arange(len(x_labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Modern colors: Train(Blue), Val(Orange), Test(Green)
    rects1 = ax.bar(x - width, train_counts, width, label='Train', color='#3498db', edgecolor='grey', alpha=0.9)
    rects2 = ax.bar(x, val_counts, width, label='Val', color='#e67e22', edgecolor='grey', alpha=0.9)
    rects3 = ax.bar(x + width, test_counts, width, label='Test', color='#2ecc71', edgecolor='grey', alpha=0.9)
    
    # Setup labels, titles, grids
    ax.set_ylabel('Number of Instances', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlabel('Screw / Nut Type', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('MVTec Screws Dataset Class Distribution', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Label value on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] Saved class distribution to '{plot_path}'")

def plot_advanced_metrics(all_areas, all_aspect_ratios, all_angles, all_counts):
    """Generate and save plots for area, aspect ratio, polar orientation, and density."""
    print("Plotting advanced analysis charts...")
    
    # 1. Bounding Box Area Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_areas, bins=30, color='#3498db', edgecolor='black', alpha=0.8)
    ax.set_title('Bounding Box Area (Size) Distribution', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Box Area (pixels$^2$)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "box_area_distribution.png"), dpi=300)
    plt.close()

    # 2. Bounding Box Aspect Ratio Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_aspect_ratios, bins=30, color='#9b59b6', edgecolor='black', alpha=0.8)
    ax.set_title('Bounding Box Aspect Ratio Distribution (Width / Height)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Aspect Ratio (Aspect ratio of 1.0 means square-like)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "aspect_ratio_distribution.png"), dpi=300)
    plt.close()

    # 3. Orientation Angle Distribution (Polar Plot)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    angles_rad = np.array(all_angles)
    
    # Group angles into 24 bins (15 degrees each)
    bins = np.linspace(-np.pi, np.pi, 25)
    counts, _ = np.histogram(angles_rad, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = np.diff(bins)[0]
    
    ax.bar(bin_centers, counts, width=width, bottom=0.0, color='#e74c3c', edgecolor='black', alpha=0.8)
    ax.set_title(r'Orientation Angle ($\phi$) Polar Distribution (Radians)', fontsize=13, fontweight='bold', pad=20)
    ax.set_theta_zero_location("E")  # 0 radians is East (right side of image)
    ax.set_theta_direction(1)       # Counter-clockwise rotation
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "orientation_distribution.png"), dpi=300)
    plt.close()

    # 4. Object Density per Image (Crowdedness analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins_range = np.arange(min(all_counts) - 0.5, max(all_counts) + 1.5, 1)
    ax.hist(all_counts, bins=bins_range, color='#2ecc71', edgecolor='black', alpha=0.8)
    ax.set_title('Objects Per Image Density', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Screws / Nuts per Image', fontsize=11)
    ax.set_ylabel('Frequency (Image Count)', fontsize=11)
    ax.set_xticks(np.arange(min(all_counts), max(all_counts) + 1, 2))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "objects_per_image.png"), dpi=300)
    plt.close()
    
    print("[OK] All advanced metrics plotted successfully.")

if __name__ == "__main__":
    try:
        analyze_dataset()
        print("\n[Done] Exploratory Data Analysis completed successfully!")
    except Exception as e:
        print(f"\n[Error] Error occurred during analysis: {e}")
