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
3. Geometric Region Filtering & Non-Maximum Suppression (Slide 11):
   - Connected contour extraction & candidate scoring using Area, Aspect Ratio (W/H),
     Axis-Aligned Extent (A_contour / A_bbox), Rotated Minimum-Area Extent (A_contour / A_rot_rect),
     Solidity (A_contour / A_convex_hull), and internal gradient/black-hat energy
   - Multi-plate Non-Maximum Suppression (NMS via IoU) for images/frames with multiple vehicles
4. Geometric Transformations & Deskewing (Slide 7):
   - Minimum-area rotated rectangle orientation estimation
   - 2D Affine Rotation & 4-Point Perspective Homography rectification to standard (160 x 128) ROI
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


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 quadrilateral vertices into [top-left, top-right, bottom-right, bottom-left]
    for 4-Point Perspective Homography rectification (Chapter 7).
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_plate_quad(cnt: np.ndarray) -> np.ndarray | None:
    """
    Approximate a plate contour's convex hull by a 4-vertex polygon using Douglas-Peucker
    polygonal approximation (Chapter 10.1 Boundary Approximations).
    Returns ordered (4, 2) float32 array if a clean quadrilateral is found, else None.
    """
    if cnt is None or len(cnt) < 4:
        return None
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.02, 0.12, 15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return order_quad_points(approx.reshape(4, 2).astype(np.float32))
    return None


def _gather_plate_candidates(img: np.ndarray) -> tuple[list[tuple[float, tuple[int, int, int, int], np.ndarray]], dict]:
    """
    Internal candidate generator:
    1. Standard RETR_EXTERNAL stream (identical on standard rear-view vehicle dataset).
    2. Morphological closing stream.
    3. Tilted / Foreshortened / Multi-Motorbike stream using rotated minimum-area rectangle
       extent (`rot_extent`) and `RETR_LIST` for collage or angled camera views.
    """
    gray = cv2.cvtColor(img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if False else cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h_img, w_img = gray.shape
    
    maps = compute_morphological_maps(gray)
    sobelx = maps["sobelx"]
    blackhat = maps["blackhat"]
    morph_closed = maps["morph_closed"]
    
    candidates = []
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Stream 1: Standard Upright External Contours (100% identical to baseline on dataset)
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
                        
    # Stream 2: Morphological Closing candidate stream
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

    # Stream 3: Tilted / Foreshortened / Off-Center / Multi-Motorbike candidate stream
    for thresh_val in [135, 150, 155, 165, 175, 190, 200, 210]:
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 1400 or area > 24000:
                continue
            rect_area = cv2.contourArea(c)
            if rect_area < 1300:
                continue
            extent = rect_area / float(area + 1e-5)
            aspect = w / float(h + 1e-5)
            rot_rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rot_rect
            rot_area = float(rw * rh + 1e-5)
            rot_extent = rect_area / rot_area
            rot_aspect = max(rw, rh) / float(min(rw, rh) + 1e-5)
            cx, cy = x + w / 2.0, y + h / 2.0

            # Admit tilted (up to 35 deg), foreshortened, or off-center plates missed by Stream 1
            if (
                40 <= w <= 160
                and 34 <= h <= 130
                and 0.75 <= aspect <= 1.72
                and 1.02 <= rot_aspect <= 1.68
                and rot_extent > 0.66
                and extent > 0.45
                and 0.02 * w_img <= cx <= 0.98 * w_img
                and 0.05 * h_img <= cy <= 0.95 * h_img
            ):
                roi_sobel = sobelx[y+4:y+h-4, x+4:x+w-4]
                roi_bh = blackhat[y+4:y+h-4, x+4:x+w-4]
                roi_gray = gray[y+4:y+h-4, x+4:x+w-4]
                if roi_sobel.size == 0:
                    continue
                score = (
                    float(np.mean(roi_sobel)) * 1.2
                    + float(np.mean(roi_bh)) * 1.6
                    + float(np.std(roi_gray)) * 1.1
                    + rot_extent * 42.0
                    - abs(rot_aspect - 1.28) * 35.0
                    - 12.0  # lower priority than Stream 1 when a standard upright plate exists
                )
                candidates.append((score, (x, y, w, h), c))
                
    if not candidates:
        # Multi-scale & scale-invariant fallback for arbitrary video resolutions (720p/1080p/close-up)
        scale_x = w_img / 472.0
        scale_y = h_img / 303.0
        if abs(scale_x - 1.0) > 0.15 or abs(scale_y - 1.0) > 0.15:
            norm_img = cv2.resize(img, (472, 303), interpolation=cv2.INTER_AREA if scale_x > 1 else cv2.INTER_CUBIC)
            sub_cands, _ = _gather_plate_candidates(norm_img)
            for sc, (nx, ny, nw, nh), ncnt in sub_cands:
                ox = int(round(nx * scale_x))
                oy = int(round(ny * scale_y))
                ow = int(round(nw * scale_x))
                oh = int(round(nh * scale_y))
                ocnt = ncnt.astype(np.float32)
                ocnt[:, 0, 0] *= scale_x
                ocnt[:, 0, 1] *= scale_y
                ocnt = ocnt.astype(np.int32)
                candidates.append((sc, (ox, oy, ow, oh), ocnt))

    if not candidates:
        # Relative-scale fallback for close-up or wide-angle video frames
        img_area = float(w_img * h_img)
        for thresh_val in [-1, 120, 145, 170, 195]:
            if thresh_val == -1:
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                box_area = float(w * h)
                rel_area = box_area / (img_area + 1e-5)
                aspect = w / float(h + 1e-5)
                extent = cv2.contourArea(c) / (box_area + 1e-5)
                if 0.008 <= rel_area <= 0.55 and 0.80 <= aspect <= 4.5 and extent > 0.50:
                    cx, cy = x + w / 2.0, y + h / 2.0
                    if 0.02 * w_img <= cx <= 0.98 * w_img and 0.04 * h_img <= cy <= 0.96 * h_img:
                        roi_sobel = sobelx[y+2:y+h-2, x+2:x+w-2]
                        roi_bh = blackhat[y+2:y+h-2, x+2:x+w-2]
                        if roi_sobel.size == 0:
                            continue
                        score = float(np.mean(roi_sobel)) + float(np.mean(roi_bh)) + extent * 25.0 - abs(aspect - 1.3) * 15.0
                        candidates.append((score, (x, y, w, h), c))
                        
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates, maps


def detect_license_plate(img: np.ndarray, return_debug: bool = False):
    """
    Detect the primary license plate bounding box and contour in a vehicle image (Slide 9 & 11).
    """
    candidates, maps = _gather_plate_candidates(img)
    if not candidates:
        if return_debug:
            return None, maps
        return None
        
    best_score, best_bbox, best_cnt = candidates[0]
    
    if return_debug:
        unique_cands = []
        for sc, bb, _ in candidates:
            if not any(compute_iou(bb, u[1]) > 0.35 or (abs(bb[0] - u[1][0]) < 12 and abs(bb[1] - u[1][1]) < 12) for u in unique_cands):
                unique_cands.append((sc, bb))
        maps["candidates"] = unique_cands[:8]
        maps["best_bbox"] = best_bbox
        maps["best_score"] = best_score
        return (best_score, best_bbox, best_cnt), maps
        
    return best_score, best_bbox, best_cnt


def detect_all_license_plates(
    img: np.ndarray,
    max_plates: int = 6,
    iou_threshold: float = 0.25,
    min_score_ratio: float = 0.62,
) -> list[tuple[float, tuple[int, int, int, int], np.ndarray]]:
    """
    Detect all non-overlapping license plate candidates in an image using Non-Maximum Suppression
    (NMS via IoU, Chapter 11) so multi-motorbike images/frames extract every plate present.
    """
    candidates, _ = _gather_plate_candidates(img)
    if not candidates:
        return []
        
    best_score = candidates[0][0]
    selected = []
    for sc, bb, cnt in candidates:
        if sc < best_score * min_score_ratio or sc < 85.0:
            continue
        cx, cy = bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0
        overlap = False
        for _, sel_bb, _ in selected:
            scx, scy = sel_bb[0] + sel_bb[2] / 2.0, sel_bb[1] + sel_bb[3] / 2.0
            if compute_iou(bb, sel_bb) > iou_threshold or (abs(cx - scx) < 35 and abs(cy - scy) < 35):
                overlap = True
                break
        if not overlap:
            selected.append((sc, bb, cnt))
            if len(selected) >= max_plates:
                break
    return selected


def deskew_and_crop_plate(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    cnt: np.ndarray | None = None,
    target_size: tuple[int, int] = (160, 128),
    interp_mode: str = "bicubic",
) -> tuple[np.ndarray, float]:
    """
    Crop and geometrically rectify (deskew) a detected license plate (Chapter 1.1 & Chapter 7):
    - For strongly tilted or foreshortened plates (w < h or axis-aligned extent <= 0.68), uses 4-Point
      Perspective Homography (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`) from the
      detected contour quadrilateral directly to `target_size` (160x128).
    - For standard upright plates, uses 2D Affine Rotation around the plate center followed by
      interpolation (`nearest`, `bilinear`, `bicubic`) to (160x128).
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
    extent = 1.0
    if cnt is not None and len(cnt) >= 5:
        extent = cv2.contourArea(cnt) / float(w * h + 1e-5)
        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), raw_angle = rect
        if rw < rh:
            raw_angle += 90.0
            rw, rh = rh, rw
        if abs(raw_angle) <= 15.0:
            angle = float(raw_angle)

    # Use 4-Point Perspective Homography ONLY when plate is foreshortened (w < h) or strongly tilted (extent <= 0.68)
    if cnt is not None and (w < h or extent <= 0.68):
        quad = extract_plate_quad(cnt)
        if quad is not None:
            dx = float(quad[1, 0] - quad[0, 0])
            dy = float(quad[1, 1] - quad[0, 1])
            persp_angle = float(np.rad2deg(np.arctan2(dy, max(dx, 1e-5))))
            tw, th = target_size
            dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
            M_persp = cv2.getPerspectiveTransform(quad, dst)
            roi_std = cv2.warpPerspective(img, M_persp, target_size, flags=cv_flag, borderMode=cv2.BORDER_REPLICATE)
            return roi_std, persp_angle

    roi = img[y:y+h, x:x+w].copy()
    if abs(angle) > 0.8:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        roi = cv2.warpAffine(roi, M, (w, h), flags=cv_flag, borderMode=cv2.BORDER_REPLICATE)
        
    roi_std = cv2.resize(roi, target_size, interpolation=cv_flag)
    return roi_std, angle
