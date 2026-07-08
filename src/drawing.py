"""
Drawing & Visualization Utilities for YOLOv11-OBB Screws Dashboard

Description:
    Handles oriented bounding box (OBB) contour drawing, transparent highlights,
    orientation direction arrows, and class labels on numpy images.
"""

import cv2
import numpy as np

# 13-Class BGR Palette for high-contrast OBB drawing
COLORS_BGR = [
    (219, 152, 52),  # Blue
    (113, 204, 46),  # Green
    (182, 89, 155),  # Purple
    (34, 126, 230),  # Orange
    (15, 196, 241),  # Yellow
    (60, 76, 231),   # Red
    (156, 188, 26),  # Teal
    (18, 156, 243),  # Dark Orange
    (96, 174, 39),   # Dark Green
    (173, 68, 142),  # Dark Purple
    (185, 128, 41),  # Dark Blue
    (43, 57, 192),   # Dark Red
    (133, 160, 22)   # Dark Teal
]

def draw_obb_predictions(image_np, results, conf_thresh):
    """
    Annotates the input image with rotated boxes, confidence scores,
    and high-contrast orientation arrows pointing in the direction of rotation.
    
    Returns:
        annotated_image (np.ndarray): The drawn image.
        detections (list): List of dictionaries containing extracted object metrics.
    """
    img_draw = image_np.copy()
    obb_data = []
    
    if len(results) == 0 or results[0].obb is None:
        return img_draw, obb_data
        
    obb = results[0].obb
    class_names = results[0].names
    
    for i in range(len(obb)):
        conf = float(obb.conf[i].item())
        if conf < conf_thresh:
            continue
            
        cls_idx = int(obb.cls[i].item())
        class_name = class_names.get(cls_idx, f"Class {cls_idx}")
        
        # Rotated Box corner points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        corners = obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
        
        # Center, Dimensions, and Angle [cx, cy, w, h, rotation_angle_radians]
        xywhr = obb.xywhr[i].cpu().numpy()
        cx, cy, w, h, r_angle = xywhr
        
        # Color setup
        color = COLORS_BGR[cls_idx % len(COLORS_BGR)]
        
        # Draw Box boundaries (polygon using positional args for OpenCV version safety)
        cv2.polylines(img_draw, [corners], True, color, 2, cv2.LINE_AA)
        
        # Transparent filled overlay for the bounding box
        overlay = img_draw.copy()
        cv2.fillPoly(overlay, [corners], color)
        cv2.addWeighted(overlay, 0.15, img_draw, 0.85, 0, dst=img_draw)
        
        # Draw Orientation Arrow (direction representing the rotation angle)
        arrow_len = int(max(w, h) / 2)
        arrow_end_x = int(cx + arrow_len * np.cos(r_angle))
        arrow_end_y = int(cy + arrow_len * np.sin(r_angle))
        
        # Double-pass arrow drawing for high-contrast on any background (using positional args for OpenCV compatibility)
        cv2.arrowedLine(img_draw, (int(cx), int(cy)), (arrow_end_x, arrow_end_y), (0, 0, 0), 4, cv2.LINE_AA, 0, 0.25)
        cv2.arrowedLine(img_draw, (int(cx), int(cy)), (arrow_end_x, arrow_end_y), (255, 255, 255), 2, cv2.LINE_AA, 0, 0.25)
        
        # Draw Text Label at the topmost corner of the bounding box
        top_corner = corners[np.argmin(corners[:, 1])]
        label_text = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        # Draw label background and text
        cv2.rectangle(img_draw, (top_corner[0], top_corner[1] - th - 6), (top_corner[0] + tw + 4, top_corner[1]), color, -1)
        cv2.putText(img_draw, label_text, (top_corner[0] + 2, top_corner[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Save metrics for tables (convert radians to degrees)
        deg_angle = float(np.degrees(r_angle)) % 360
        obb_data.append({
            "Object ID": len(obb_data) + 1,
            "Class Name": class_name,
            "Confidence": conf,
            "Center X": int(cx),
            "Center Y": int(cy),
            "Width": int(w),
            "Height": int(h),
            "Angle (Deg)": round(deg_angle, 1),
            "Angle (Rad)": round(float(r_angle), 3),
            "corners": corners
        })
        
    return img_draw, obb_data
