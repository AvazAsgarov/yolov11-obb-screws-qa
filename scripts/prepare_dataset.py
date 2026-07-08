"""
MVTec Screws Dataset Preparation and Preprocessing Script

Description:
    This script automates the preprocessing pipeline for the MVTec Screws dataset
    to train an Oriented Bounding Box (OBB) object detection model (YOLO OBB).
    
    It performs the following steps:
    1. Extracts the raw dataset from the 'mvtec_screws_v1.1.tar.gz' archive.
    2. Converts COCO-based oriented bounding boxes [row, col, width, height, phi] to 
       YOLO-OBB normalized 4-corner coordinates [x1, y1, x2, y2, x3, y3, x4, y4].
       It handles coordinate clipping to ensure points stay within [0, 1] range.
    3. Partitions images and label files into standard 'train', 'val', and 'test' splits.
    4. Writes the YOLO training configuration file 'data.yaml'.
    5. Generates visual sanity checks by drawing predicted rotated boxes on sample images 
       and saving them to 'dataset/sanity_checks/'.
    6. Cleans up temporary raw extracted directories to save local disk space.
    7. Packages the final formatted YOLO dataset into a zip file for easy upload to Google Colab.
"""

import os
import tarfile
import json
import zipfile
import shutil
import cv2
import numpy as np

# Resolve project root relative to script file location for robustness
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

TAR_PATH = os.path.join(PROJECT_ROOT, "mvtec_screws_v1.1.tar.gz")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RAW_EXTRACT_DIR = os.path.join(DATASET_DIR, "tmp_raw")
YOLO_DIR = DATASET_DIR
ZIP_OUT_PATH = os.path.join(PROJECT_ROOT, "yolo_obb_dataset.zip")

def create_folders():
    """Create directory structure for YOLO OBB and temporary extractions."""
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(YOLO_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, "sanity_checks"), exist_ok=True)
    os.makedirs(RAW_EXTRACT_DIR, exist_ok=True)
    print("[OK] Folder structure created.")

def extract_dataset():
    """Extract raw dataset from tar.gz archive to a temporary directory."""
    if not os.path.exists(TAR_PATH):
        raise FileNotFoundError(f"Dataset archive '{TAR_PATH}' not found in the project directory.")
    
    print(f"Extracting '{os.path.basename(TAR_PATH)}' to '{RAW_EXTRACT_DIR}'...")
    with tarfile.open(TAR_PATH, "r:gz") as tf:
        tf.extractall(path=RAW_EXTRACT_DIR)
    print("[OK] Dataset extracted successfully.")

def convert_coco_to_yolo_obb():
    """
    Parse COCO JSON annotations, convert rotated bounding boxes to YOLO OBB format,
    save text labels, copy images to their split directories, and output sanity checks.
    """
    splits = ["train", "val", "test"]
    class_names = {}
    sanity_check_samples = []

    for split in splits:
        json_path = os.path.join(RAW_EXTRACT_DIR, f"v1.1/mvtec_screws_{split}.json")
        if not os.path.exists(json_path):
            print(f"[Warning] JSON annotation file '{json_path}' not found.")
            continue
            
        print(f"Processing '{split}' split...")
        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
            
        # Map category IDs to names (0-indexed for YOLO)
        for cat in coco_data["categories"]:
            cat_id = cat["id"]
            cat_name = cat["name"]
            class_names[cat_id - 1] = cat_name
            
        # Create image mappings
        images_map = {img["id"]: img for img in coco_data["images"]}
        
        # Group annotations by image
        annotations_by_img = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            annotations_by_img.setdefault(img_id, []).append(ann)
            
        # Process each image and its annotations
        for img_id, img_info in images_map.items():
            file_name = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]
            
            src_img_path = os.path.join(RAW_EXTRACT_DIR, "v1.1/images", file_name)
            dst_img_path = os.path.join(YOLO_DIR, "images", split, file_name)
            
            if not os.path.exists(src_img_path):
                print(f"[Warning] Image file '{src_img_path}' not found.")
                continue
                
            # Copy image to YOLO directory split
            shutil.copy2(src_img_path, dst_img_path)
            
            # Create label text file
            base_name = os.path.splitext(file_name)[0]
            label_file_path = os.path.join(YOLO_DIR, "labels", split, f"{base_name}.txt")
            
            img_annots = annotations_by_img.get(img_id, [])
            
            # Select first 3 train images for OBB sanity check drawing
            collect_sanity = (split == "train" and len(sanity_check_samples) < 3)
            current_sanity_boxes = []
            
            with open(label_file_path, "w", encoding="utf-8") as label_file:
                for ann in img_annots:
                    # bbox: [row, col, width, height, phi]
                    row, col, w, h, phi = ann["bbox"]
                    cat_id = ann["category_id"]
                    
                    class_idx = cat_id - 1
                    
                    # Compute local box corners relative to center
                    half_w = w / 2.0
                    half_h = h / 2.0
                    local_pts = np.array([
                        [-half_w, -half_h],  # Top-left (local)
                        [half_w, -half_h],   # Top-right (local)
                        [half_w, half_h],    # Bottom-right (local)
                        [-half_w, half_h]    # Bottom-left (local)
                    ])
                    
                    # Rotate and translate corners to image (col, row) coordinates
                    # col_new = col + u * cos(phi) - v * sin(phi)
                    # row_new = row - (u * sin(phi) + v * cos(phi))
                    cos_phi = np.cos(phi)
                    sin_phi = np.sin(phi)
                    
                    global_pts = []
                    for u, v in local_pts:
                        c_new = col + u * cos_phi - v * sin_phi
                        r_new = row - (u * sin_phi + v * cos_phi)
                        
                        # Normalize corners relative to image dimensions
                        x_norm = c_new / img_w
                        y_norm = r_new / img_h
                        
                        # Clip boundary cases to stay inside [0.0, 1.0]
                        x_norm = max(0.0, min(1.0, x_norm))
                        y_norm = max(0.0, min(1.0, y_norm))
                        
                        global_pts.append(f"{x_norm:.6f} {y_norm:.6f}")
                        
                        if collect_sanity:
                            current_sanity_boxes.append((c_new, r_new))
                    
                    # Write label in YOLO-OBB format: class_idx x1 y1 x2 y2 x3 y3 x4 y4
                    label_file.write(f"{class_idx} {' '.join(global_pts)}\n")
            
            if collect_sanity and current_sanity_boxes:
                sanity_check_samples.append({
                    "img_path": dst_img_path,
                    "boxes": current_sanity_boxes,
                    "file_name": file_name
                })
                
    # Draw OBB corners on sanity check samples to verify conversion accuracy
    print("Generating visual sanity check samples...")
    for sample in sanity_check_samples:
        img = cv2.imread(sample["img_path"])
        pts = np.array(sample["boxes"], dtype=np.int32).reshape(-1, 4, 2)
        for poly in pts:
            cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            # Color-code corners to verify ordering: Red(P1) -> Yellow(P2) -> Blue(P3) -> Magenta(P4)
            cv2.circle(img, tuple(poly[0]), 5, (0, 0, 255), -1)   # P1: Red
            cv2.circle(img, tuple(poly[1]), 5, (0, 255, 255), -1) # P2: Yellow
            cv2.circle(img, tuple(poly[2]), 5, (255, 0, 0), -1)   # P3: Blue
            cv2.circle(img, tuple(poly[3]), 5, (255, 0, 255), -1) # P4: Magenta
            
        out_path = os.path.join(YOLO_DIR, "sanity_checks", f"sanity_{sample['file_name']}")
        cv2.imwrite(out_path, img)
    print(f"[OK] Visual sanity checks saved to {os.path.join(YOLO_DIR, 'sanity_checks')}.")
    
    # Generate data.yaml file
    yaml_path = os.path.join(YOLO_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(f"path: {os.path.abspath(YOLO_DIR).replace(os.sep, '/')}\n")
        yf.write("train: images/train\n")
        yf.write("val: images/val\n")
        yf.write("test: images/test\n\n")
        yf.write("names:\n")
        for idx in sorted(class_names.keys()):
            yf.write(f"  {idx}: {class_names[idx]}\n")
    print("[OK] data.yaml created.")

def cleanup_raw_files():
    """Remove raw temporary extracted files to optimize workspace size."""
    if os.path.exists(RAW_EXTRACT_DIR):
        print(f"Cleaning up raw extracted directory '{RAW_EXTRACT_DIR}'...")
        shutil.rmtree(RAW_EXTRACT_DIR)
        print("[OK] Cleanup completed.")

def zip_yolo_dataset():
    """Compress the formatted YOLO directory into a zip archive for Colab."""
    print(f"Zipping YOLO OBB dataset into '{ZIP_OUT_PATH}'...")
    with zipfile.ZipFile(ZIP_OUT_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(YOLO_DIR):
            for file in files:
                if "sanity_checks" in root:
                    continue
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, YOLO_DIR)
                zipf.write(file_path, arc_name)
    print("[OK] Dataset compressed to zip successfully.")

if __name__ == "__main__":
    try:
        create_folders()
        extract_dataset()
        convert_coco_to_yolo_obb()
        cleanup_raw_files()
        zip_yolo_dataset()
        print("\n[Done] Dataset preprocessing completed successfully!")
    except Exception as e:
        print(f"\n[Error] Error occurred during execution: {e}")
