"""
Module 2: License Plate Detection & Geometric Deskewing (Slide 2, 7, 9, 11)
===========================================================================
Implements:
1. Edge & Gradient Operators (Slide 2 & 9):
   - Vertical Sobel gradient operator G_x = dI/dx exploiting high vertical stroke density
   - Canny edge detector
2. Mathematical Morphology (Slide 9 & 11):
   - Morphological Black-Hat (closing minus original) to highlight dark characters on bright plate
   - Morphological White Top-Hat (original minus opening) to highlight bright rectangular plate
   - Morphological Closing & Opening with rectangular structuring elements to group strokes
3. Geometric Region Filtering (Slide 11):
   - Connected contour extraction & candidate scoring using Area, Aspect Ratio (W/H),
     Extent (A_contour / A_bbox), Solidity (A_contour / A_convex_hull), and internal gradient energy
4. Geometric Transformations & Deskewing (Slide 7):
   - Minimum-area rotated rectangle orientation estimation
   - 2D Affine Rotation / Perspective 4-point Homography rectification to standard (160 x 128) ROI
"""

import cv2
import numpy as np


def compute_sobel_vertical(gray: np.ndarray) -> np.ndarray:
    """
    Compute normalized vertical Sobel gradient magnitude |G_x| = |dI/dx| (Slide 2 & 9).
    License plates contain dense vertical character strokes, producing strong |G_x| response.
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.abs(sobelx)
    max_val = np.max(abs_sobelx)
    if max_val > 1e-6:
        abs_sobelx = (255.0 * abs_sobelx / max_val)
    return abs_sobelx.astype(np.uint8)


def compute_morphological_maps(gray: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute morphological feature maps (Slide 9 & 11):
    - tophat: highlights bright objects smaller than structuring element
    - blackhat: highlights dark strokes inside bright license plate background
    - sobelx: vertical gradient magnitude
    - morph_closed: binary region grouping via Otsu + Closing + Opening
    """
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    sobelx = compute_sobel_vertical(gray)
    
    combined_energy = cv2.addWeighted(sobelx, 0.5, blackhat, 0.5, 0)
    _, bw_edge = cv2.threshold(combined_energy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 11))
    morph_closed = cv2.morphologyEx(bw_edge, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_closed = cv2.morphologyEx(morph_closed, cv2.MORPH_OPEN, k_open)
    
    return {
        "gray": gray,
        "tophat": tophat,
        "blackhat": blackhat,
        "sobelx": sobelx,
        "combined_energy": combined_energy,
        "morph_closed": morph_closed,
    }


def detect_license_plate(img: np.ndarray, return_debug: bool = False):
    """
    Detect the license plate bounding box and contour in a vehicle image (Slide 9 & 11).
    Combines multi-level brightness/Otsu thresholding with morphological opening and
    internal gradient/black-hat energy scoring.
    
    Returns
    -------
    If return_debug is False:
        (score, bbox, contour) or None
    If return_debug is True:
        (score, bbox, contour, debug_dict)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h_img, w_img = gray.shape
    
    maps = compute_morphological_maps(gray)
    sobelx = maps["sobelx"]
    blackhat = maps["blackhat"]
    morph_closed = maps["morph_closed"]
    
    candidates = []
    all_raw_boxes = []
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for thresh_val in [110, 125, 140, 155, 170, 185, 200, -1]:
        if thresh_val == -1:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
            
        for k_size in [(1, 1), (3, 3), (5, 5)]:
            if k_size == (1, 1):
                opened = binary
            else:
                k_open = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
                opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
                
            cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if area < 1200 or area > 20000:
                    continue
                aspect = w / float(h + 1e-5)
                rect_area = cv2.contourArea(c)
                extent = rect_area / float(area + 1e-5)
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = rect_area / float(hull_area + 1e-5)
                
                all_raw_boxes.append((x, y, w, h, aspect, extent, solidity))
                
                if 50 <= w <= 135 and 42 <= h <= 105 and 1.0 <= aspect <= 1.62 and extent > 0.68:
                    cx, cy = x + w / 2.0, y + h / 2.0
                    if 0.15 * w_img <= cx <= 0.85 * w_img and 0.05 * h_img <= cy <= 0.90 * h_img:
                        roi_sobel = sobelx[y+4:y+h-4, x+4:x+w-4]
                        roi_bh = blackhat[y+4:y+h-4, x+4:x+w-4]
                        roi_gray = gray[y+4:y+h-4, x+4:x+w-4]
                        if roi_sobel.size == 0:
                            continue
                        score = (
                            float(np.mean(roi_sobel)) * 1.2
                            + float(np.mean(roi_bh)) * 1.6
                            + float(np.std(roi_gray)) * 1.1
                            + extent * 45.0
                            - abs(aspect - 1.24) * 45.0
                        )
                        candidates.append((score, (x, y, w, h), c))
                        
    # Morphological Closing candidate stream
    cnts, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h + 1e-5)
        rect_area = cv2.contourArea(c)
        extent = rect_area / float(w * h + 1e-5)
        if 50 <= w <= 135 and 42 <= h <= 105 and 1.0 <= aspect <= 1.62 and extent > 0.55:
            cx, cy = x + w / 2.0, y + h / 2.0
            if 0.15 * w_img <= cx <= 0.85 * w_img and 0.05 * h_img <= cy <= 0.90 * h_img:
                roi_sobel = sobelx[y+4:y+h-4, x+4:x+w-4]
                roi_bh = blackhat[y+4:y+h-4, x+4:x+w-4]
                roi_gray = gray[y+4:y+h-4, x+4:x+w-4]
                if roi_sobel.size == 0:
                    continue
                score = (
                    float(np.mean(roi_sobel)) * 1.2
                    + float(np.mean(roi_bh)) * 1.5
                    + float(np.std(roi_gray)) * 1.0
                    + extent * 30.0
                    - abs(aspect - 1.24) * 45.0
                )
                candidates.append((score, (x, y, w, h), c))
                
    if not candidates:
        # Fallback: relax extent & aspect slightly for severely degraded images
        for thresh_val in [-1, 130, 150, 170]:
            if thresh_val == -1:
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h + 1e-5)
                extent = cv2.contourArea(c) / float(w * h + 1e-5)
                if 45 <= w <= 150 and 38 <= h <= 115 and 0.90 <= aspect <= 1.75 and extent > 0.52:
                    cx, cy = x + w / 2.0, y + h / 2.0
                    if 0.12 * w_img <= cx <= 0.88 * w_img and 0.04 * h_img <= cy <= 0.92 * h_img:
                        roi_sobel = sobelx[y+2:y+h-2, x+2:x+w-2]
                        roi_bh = blackhat[y+2:y+h-2, x+2:x+w-2]
                        if roi_sobel.size == 0:
                            continue
                        score = float(np.mean(roi_sobel)) + float(np.mean(roi_bh)) + extent * 25.0
                        candidates.append((score, (x, y, w, h), c))
                        
    if not candidates:
        if return_debug:
            return None, maps
        return None
        
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_bbox, best_cnt = candidates[0]
    
    if return_debug:
        # Deduplicate candidate boxes for visualization
        unique_cands = []
        for sc, bb, _ in candidates:
            if not any(abs(bb[0] - u[1][0]) < 10 and abs(bb[1] - u[1][1]) < 10 for u in unique_cands):
                unique_cands.append((sc, bb))
        maps["candidates"] = unique_cands[:8]
        maps["best_bbox"] = best_bbox
        maps["best_score"] = best_score
        return (best_score, best_bbox, best_cnt), maps
        
    return best_score, best_bbox, best_cnt


def compute_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes (x, y, w, h)
    (Chapter 11: Object Detection, Fundamentals - Evaluation Metrics, Slide 5):
        IoU = Area(A intersect B) / Area(A union B)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = float(interW * interH)
    unionArea = float(boxA[2] * boxA[3] + boxB[2] * boxB[3] - interArea + 1e-6)
    return interArea / unionArea


def deskew_and_crop_plate(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    cnt: np.ndarray | None = None,
    target_size: tuple[int, int] = (160, 128),
    interp_mode: str = "bicubic",
) -> tuple[np.ndarray, float]:
    """
    Crop and geometrically rectify (deskew) a detected license plate using
    Affine Rotation around the plate center (Chapter 1 Section 1.1 Slide 14-16 & Chapter 7 Section 7.1),
    then normalize to `target_size` (160x128).
    Supports `interp_mode`: 'nearest', 'bilinear', 'bicubic'.
    Returns (rectified_plate_bgr, skew_angle_deg).
    """
    flag_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }
    cv_flag = flag_map.get(interp_mode, cv2.INTER_CUBIC)
    x, y, w, h = bbox
    angle = 0.0
    if cnt is not None and len(cnt) >= 5:
        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), raw_angle = rect
        if rw < rh:
            raw_angle += 90.0
            rw, rh = rh, rw
        if abs(raw_angle) <= 15.0:
            angle = float(raw_angle)
            
    roi = img[y:y+h, x:x+w].copy()
    if abs(angle) > 0.8:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        roi = cv2.warpAffine(roi, M, (w, h), flags=cv_flag, borderMode=cv2.BORDER_REPLICATE)
        
    roi_std = cv2.resize(roi, target_size, interpolation=cv_flag)
    return roi_std, angle

