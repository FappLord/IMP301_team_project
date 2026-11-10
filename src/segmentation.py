"""
Module 3: Binarization & Character Segmentation (Doc Chapters 1.2, 2.2, 9.1)
=============================================================================
Implements strictly from IMP302 Course Lecture Slides (`Doc/1, 2, 9.pdf`):
1. Global Otsu Thresholding from First Principles (Ch. 1.2, Slides 21-22):
   - Histogram probability distribution p(i) = n_i / N
   - Between-class variance maximization: sigma_B^2(t) = w0(t) * w1(t) * [mu0(t) - mu1(t)]^2
   - Returns optimal threshold t*, binarized image, and full sigma_B^2(t) curve
2. K-Means Clustering Segmentation (Ch. 9.1, Slides 5-6):
   - Iterative partitioning of pixel intensities into K=2 clusters (foreground ink vs. plate background)
     by minimizing within-cluster sum of squared errors (WCSS): sum_{k=1}^K sum_{x in C_k} ||x - m_k||^2
3. Local Adaptive Gaussian Thresholding (Ch. 1.2 & Ch. 9):
   - Local threshold surface T(x,y) = G_sigma * I(x,y) - C to handle shadows and uneven lighting
4. Morphological Frame & Noise Removal (Ch. 1.2 Slides 28-30 & Ch. 2.2 Slides 21-26):
   - Horizontal/vertical projection profile border line suppression
   - Iterative morphological bridge severing for characters touching the outer plate frame
5. Connected Component Labeling (8-Connectivity Region Labeling, Ch. 1.2 Slides 25-27):
   - Two-pass / BFS 8-connected Region Labeling & color-coded label map visualization
   - Geometric bounding-box filtering, horizontal valley splitting for touching characters,
     and 2-line topological sorting (top row: 4 chars, bottom row: 4-5 chars)
"""

import cv2
import numpy as np


def otsu_threshold_manual(gray: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Compute Otsu's optimal global threshold from first principles (Ch. 1.2, Slides 21-22):
        t* = argmax_{0 <= t < 256} sigma_B^2(t)
        where sigma_B^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2
    Returns (optimal_t, binary_inv_img, sigma_b_sq_curve).
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = float(gray.size)
    prob = hist / (total + 1e-12)
    
    omega = np.cumsum(prob)
    mu_k = np.cumsum(prob * np.arange(256))
    mu_total = mu_k[-1]
    
    # Avoid division by zero at extremes
    denom = omega * (1.0 - omega)
    sigma_b_sq = np.zeros(256, dtype=np.float64)
    valid = denom > 1e-10
    sigma_b_sq[valid] = ((mu_total * omega[valid] - mu_k[valid]) ** 2) / denom[valid]
    
    opt_t = int(np.argmax(sigma_b_sq))
    binary_inv = np.where(gray <= opt_t, 255, 0).astype(np.uint8)
    return opt_t, binary_inv, sigma_b_sq


def kmeans_binarize_manual(gray: np.ndarray, max_iter: int = 30, tol: float = 1e-3) -> tuple[tuple[float, float], np.ndarray]:
    """
    K-Means Clustering Segmentation for K=2 clusters from first principles (Ch. 9.1, Slides 5-6):
    1. Initialize K=2 cluster centroids (m_0 = 25th percentile dark ink, m_1 = 75th percentile bright background).
    2. Assign each pixel x_i to the closest centroid: c(i) = argmin_{k in {0,1}} (x_i - m_k)^2.
    3. Recompute centroids m_k = (1 / |C_k|) sum_{i in C_k} x_i until convergence.
    Returns ((m_dark, m_bright), binary_inv_img).
    """
    x = gray.astype(np.float32)
    m0 = float(np.percentile(x, 25))
    m1 = float(np.percentile(x, 75))
    if abs(m1 - m0) < 1e-3:
        m0, m1 = 60.0, 190.0

    for _ in range(max_iter):
        # Decision boundary for 1D 2-means is the midpoint (m0 + m1) / 2
        mid = 0.5 * (m0 + m1)
        mask_dark = x <= mid
        new_m0 = float(np.mean(x[mask_dark])) if np.any(mask_dark) else m0
        new_m1 = float(np.mean(x[~mask_dark])) if np.any(~mask_dark) else m1
        if abs(new_m0 - m0) + abs(new_m1 - m1) < tol:
            m0, m1 = new_m0, new_m1
            break
        m0, m1 = new_m0, new_m1

    if m0 > m1:
        m0, m1 = m1, m0
    mid = 0.5 * (m0 + m1)
    binary_inv = np.where(x <= mid, 255, 0).astype(np.uint8)
    return (m0, m1), binary_inv


def binarize_plate(plate_bgr: np.ndarray, method: str = "adaptive") -> tuple[np.ndarray, dict]:
    """
    Binarize a rectified (160x128) license plate image using:
    - 'otsu'    : Global Otsu thresholding (Ch. 1.2 Slides 21-22)
    - 'kmeans'  : K-Means Clustering (K=2) segmentation (Ch. 9.1 Slides 5-6)
    - 'adaptive': Local Gaussian Adaptive Thresholding (window=27, C=8)
    Returns (binary_inv, info_dict).
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) if plate_bgr.ndim == 3 else plate_bgr.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enh = clahe.apply(gray)
    blur = cv2.GaussianBlur(enh, (3, 3), 0)
    
    opt_t, otsu_bin, sigma_curve = otsu_threshold_manual(blur)
    (km_m0, km_m1), kmeans_bin = kmeans_binarize_manual(blur)
    adaptive_bin = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 8
    )
    
    if method == "otsu":
        chosen = otsu_bin.copy()
    elif method == "kmeans":
        chosen = kmeans_bin.copy()
    else:
        chosen = adaptive_bin.copy()

    info = {
        "gray": gray,
        "enhanced": enh,
        "otsu_t": opt_t,
        "otsu_binary": otsu_bin,
        "otsu_sigma_curve": sigma_curve,
        "kmeans_centers": (km_m0, km_m1),
        "kmeans_binary": kmeans_bin,
        "adaptive_binary": adaptive_bin,
    }
    return chosen, info


def colorize_label_map(labels: np.ndarray, valid_label_ids: list[int] | None = None) -> np.ndarray:
    """
    Generate a vivid RGB color-coded map for Connected Component Labeling (Region Labeling).
    Background (0) is rendered black; valid character regions get bright distinct hues,
    and rejected noise/border regions get dim gray.
    """
    H, W = labels.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    num_labels = int(np.max(labels)) + 1
    if num_labels <= 1:
        return out
        
    palette = [
        (255, 99, 71), (50, 205, 50), (30, 144, 255), (255, 215, 0),
        (255, 105, 180), (0, 255, 255), (255, 140, 0), (186, 85, 211),
        (0, 250, 154), (255, 69, 0), (135, 206, 250), (240, 230, 140),
    ]
    valid_set = set(valid_label_ids) if valid_label_ids is not None else set(range(1, num_labels))
    color_idx = 0
    for lbl in range(1, num_labels):
        mask = labels == lbl
        if lbl in valid_set:
            c = palette[color_idx % len(palette)]
            out[mask] = (c[2], c[1], c[0]) # BGR
            color_idx += 1
        else:
            out[mask] = (75, 75, 75)
    return out


def segment_characters(plate_bgr: np.ndarray, method: str = "adaptive") -> tuple[list[dict], dict]:
    """
    Full Module 3 Character Segmentation Pipeline on a normalized (160x128) license plate:
    1. Binarization (Otsu or Adaptive Gaussian Thresholding)
    2. Projection-based border line suppression & iterative bridge severing
    3. 8-Connectivity Connected Component Labeling (Region Labeling)
    4. Geometric filtering, touching-character valley splitting, and 2-row sorting
    5. Normalization of each segmented character to 32x32 binary & grayscale patches
    
    Returns
    -------
    char_items : list of dicts
        Each item has keys: 'bbox' (x,y,w,h), 'centroid' (cx,cy), 'row' (0 or 1),
        'bin_32' (32x32 uint8), 'area_32' (32x32 uint8), 'gray_32' (32x32 uint8).
    debug_info : dict
        Intermediate images: 'raw_binary', 'clean_binary', 'raw_labels', 'clean_labels',
        'color_labels_raw', 'color_labels_clean', 'otsu_binary', 'adaptive_binary', etc.
    """
    raw_binary, bin_info = binarize_plate(plate_bgr, method=method)
    gray_plate = bin_info["gray"]
    H, W = raw_binary.shape # 128 x 160
    
    # Record raw 8-connected components before morphological frame cleaning
    _, raw_labels, _, _ = cv2.connectedComponentsWithStats(raw_binary, connectivity=8)
    
    binary = raw_binary.copy()
    binary[:7, :] = 0
    binary[-7:, :] = 0
    binary[:, :6] = 0
    binary[:, -6:] = 0
    
    for r in list(range(7, 15)) + list(range(H - 15, H - 7)):
        if np.mean(binary[r, :] > 0) > 0.40:
            binary[r, :] = 0
    for c in list(range(6, 13)) + list(range(W - 13, W - 6)):
        if np.mean(binary[:, c] > 0) > 0.40:
            binary[:, c] = 0

    clean_bin = binary.copy()
    for margin_step in [(9, 10), (11, 12), (13, 14)]:
        mx, my = margin_step
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_bin, connectivity=8)
        changed = False
        for lbl in range(1, num_labels):
            bx = stats[lbl, cv2.CC_STAT_LEFT]
            by = stats[lbl, cv2.CC_STAT_TOP]
            bw = stats[lbl, cv2.CC_STAT_WIDTH]
            bh = stats[lbl, cv2.CC_STAT_HEIGHT]
            if bw > 42 or bh > 63:
                mask = labels == lbl
                if by <= my:
                    clean_bin[:my+1, :][mask[:my+1, :]] = 0
                    changed = True
                if by + bh >= H - my:
                    clean_bin[H-my-1:, :][mask[H-my-1:, :]] = 0
                    changed = True
                if bx <= mx:
                    clean_bin[:, :mx+1][mask[:, :mx+1]] = 0
                    changed = True
                if bx + bw >= W - mx:
                    clean_bin[:, W-mx-1:][mask[:, W-mx-1:]] = 0
                    changed = True
                if by < 55 and by + bh > 72:
                    clean_bin[60:67, :][mask[60:67, :]] = 0
                    changed = True
        if not changed:
            break

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_bin, connectivity=8)
    for lbl in range(1, num_labels):
        bx = stats[lbl, cv2.CC_STAT_LEFT]
        by = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]
        if bh > 64 or bw > 68 or area < 55:
            clean_bin[labels == lbl] = 0
            continue
        if (bx <= 8 or bx + bw >= W - 8) and (area < 190 or bw <= 9 or bh > 56):
            clean_bin[labels == lbl] = 0
            continue

    num_labels, clean_labels, stats, centroids = cv2.connectedComponentsWithStats(clean_bin, connectivity=8)
    raw_boxes = []
    valid_lbl_ids = []
    for lbl in range(1, num_labels):
        bx = int(stats[lbl, cv2.CC_STAT_LEFT])
        by = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])
        aspect = bw / float(bh + 1e-5)
        
        if 25 <= bh <= 62 and 4 <= bw <= 64 and 0.09 <= aspect <= 1.45 and area >= 60:
            if 6 <= cx <= W - 6 and 10 <= cy <= H - 10:
                if aspect > 0.72 and bw >= 32:
                    roi_char = clean_bin[by:by+bh, bx:bx+bw]
                    w_start, w_end = int(bw * 0.28), int(bw * 0.72)
                    if w_end > w_start:
                        col_sums = np.sum(roi_char[:, w_start:w_end], axis=0) / 255.0
                        split_rel = w_start + int(np.argmin(col_sums))
                        if col_sums[split_rel - w_start] <= 8:
                            bw1 = split_rel
                            bw2 = bw - split_rel
                            raw_boxes.append((bx, by, bw1, bh, bx + bw1 / 2.0, cy, lbl))
                            raw_boxes.append((bx + split_rel, by, bw2, bh, bx + split_rel + bw2 / 2.0, cy, lbl))
                            valid_lbl_ids.append(lbl)
                            continue
                if aspect <= 0.82:
                    raw_boxes.append((bx, by, bw, bh, cx, cy, lbl))
                    valid_lbl_ids.append(lbl)

    top_line = sorted([b for b in raw_boxes if b[5] < H * 0.51], key=lambda b: b[0])
    bot_line = sorted([b for b in raw_boxes if b[5] >= H * 0.51], key=lambda b: b[0])
    
    def _filter_row(row, max_chars):
        if len(row) > 1:
            med_h = np.median([b[3] for b in row])
            med_y = np.median([b[1] for b in row])
            row = [b for b in row if b[3] >= 0.68 * med_h and abs(b[1] - med_y) <= 14]
        if len(row) <= max_chars:
            return row
        med_y = np.median([b[1] for b in row])
        med_h = np.median([b[3] for b in row])
        scored = []
        for b in row:
            bx, by, bw, bh, cx, cy, lbl = b
            edge_pen = 25.0 if (bx <= 9 or bx + bw >= W - 9) else 0.0
            dist = abs(by - med_y) * 1.2 + abs(bh - med_h) * 1.5 + edge_pen
            scored.append((dist, b))
        scored.sort(key=lambda x: x[0])
        kept = [x[1] for x in scored[:max_chars]]
        return sorted(kept, key=lambda b: b[0])

    top_line = _filter_row(top_line, 4)
    bot_line = _filter_row(bot_line, 5)
    ordered_boxes = [(0, b) for b in top_line] + [(1, b) for b in bot_line]
    kept_lbls = [b[6] for _, b in ordered_boxes]
    
    char_items = []
    for pos_idx, (row_idx, (bx, by, bw, bh, cx, cy, lbl)) in enumerate(ordered_boxes):
        p_bin = clean_bin[by:by+bh, bx:bx+bw]
        p_gray = gray_plate[by:by+bh, bx:bx+bw]
        pad_x = max(2, (bh - bw) // 3) if bw < bh * 0.45 else 2
        
        bin_pad = cv2.copyMakeBorder(p_bin, 2, 2, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
        gray_pad = cv2.copyMakeBorder(p_gray, 2, 2, pad_x, pad_x, cv2.BORDER_REPLICATE)
        
        bin_32 = cv2.resize(bin_pad, (32, 32), interpolation=cv2.INTER_NEAREST)
        area_32 = cv2.resize(bin_pad, (32, 32), interpolation=cv2.INTER_AREA)
        gray_32 = cv2.resize(gray_pad, (32, 32), interpolation=cv2.INTER_CUBIC)
        
        char_items.append({
            "pos_idx": pos_idx,
            "row": row_idx,
            "bbox": (bx, by, bw, bh),
            "centroid": (cx, cy),
            "label_id": lbl,
            "bin_32": bin_32,
            "area_32": area_32,
            "gray_32": gray_32,
        })
        
    debug_info = {
        **bin_info,
        "raw_binary": raw_binary,
        "clean_binary": clean_bin,
        "raw_labels": raw_labels,
        "clean_labels": clean_labels,
        "color_labels_raw": colorize_label_map(raw_labels),
        "color_labels_clean": colorize_label_map(clean_labels, kept_lbls),
    }
    return char_items, debug_info
