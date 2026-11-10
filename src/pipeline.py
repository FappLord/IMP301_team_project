"""
End-to-End Classical Image Processing Pipeline for License Plate Restoration & Recognition
==========================================================================================
Integrates:
- Dataset Splitting (Train 70% / Validation 15% / Test 15% at Plate Level, zero leakage)
- Multi-Degradation Dataset Expansion (6x expansion = 690 vehicle images across Chapter 3
  degradation models: Clean, AWGN, Salt-and-Pepper, Rayleigh/Gamma, Periodic, Motion Blur + AWGN)
- Module 1 (Primary Theme): License Plate Image Restoration & Enhancement (`src.restoration`)
- Module 2: Plate Detection & Geometric Deskewing (`src.detection`)
- Module 3: Binarization (Otsu / K-Means / Adaptive) & 8-Connected Component Segmentation (`src.segmentation`)
- Module 4: WHT/DCT/DHT + HOG + Hu/Fourier/Topology Feature Extraction & Classification (`src.recognition`)
"""

import os
import json
import glob
import cv2
import numpy as np

from .restoration import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_rayleigh_noise,
    add_periodic_noise,
    synthesize_degraded_image,
    weighted_median_filter,
    harmonic_mean_filter,
    butterworth_notch_reject_filter,
    wavelet_denoise,
    unsharp_masking_highboost,
    full_restoration_pipeline,
    evaluate_quality,
)
from .detection import detect_license_plate, deskew_and_crop_plate
from .segmentation import segment_characters
from .recognition import ClassicalCharRecognizer, format_vietnamese_plate


# Cleaned ground-truth annotations correcting 2 minor raw OCR/typing typos in dataset/label
LABEL_CORRECTIONS = {
    "0000_02187_b": "59E121500",  # Raw txt had '59E12150' (missing trailing 0 on 59-E1 215.00)
    "018": "60B801075",           # Raw txt had '60B8010175' (extra 1 on 60-B8 010.75)
}

DEGRADATION_TYPES = [
    "clean",
    "awgn",
    "salt_pepper",
    "rayleigh",
    "periodic",
    "motion_blur",
]


def load_dataset(dataset_dir: str = "dataset", use_cleaned_labels: bool = True) -> list[dict]:
    """
    Load all matched (image_path, base_name, ground_truth_label) entries from `dataset/`.
    """
    img_files = sorted(glob.glob(os.path.join(dataset_dir, "img", "*.jpg")))
    lbl_map = {}
    for lf in glob.glob(os.path.join(dataset_dir, "label", "*.txt")):
        base = os.path.splitext(os.path.basename(lf))[0]
        raw_txt = open(lf, encoding="utf-8", errors="ignore").read().strip()
        if use_cleaned_labels and base in LABEL_CORRECTIONS:
            lbl_map[base] = LABEL_CORRECTIONS[base]
        else:
            lbl_map[base] = raw_txt
            
    records = []
    for f in img_files:
        base = os.path.splitext(os.path.basename(f))[0]
        records.append({
            "base": base,
            "img_path": f,
            "gt_label": lbl_map.get(base, None),
            "raw_label": open(os.path.join(dataset_dir, "label", f"{base}.txt")).read().strip()
            if os.path.exists(os.path.join(dataset_dir, "label", f"{base}.txt"))
            else None,
        })
    return records


def create_dataset_splits(
    records: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    save_json_path: str | None = "data/splits/dataset_splits.json",
) -> dict[str, list[dict]]:
    """
    Partition the 115 labeled vehicle plates into Train (70% = 80), Validation (15% = 17),
    and Test (15% = 18) sets at the plate level (zero character leakage between splits).
    Uses stratified sampling by series letter (position 2 of the plate) so that rare letters
    are guaranteed to appear in the Training split.
    """
    labeled = [r for r in records if r.get("gt_label")]
    rng = np.random.default_rng(seed)

    # Group plates by their series letter (gt_label[2]) for stratified splitting
    by_letter: dict[str, list[dict]] = {}
    for r in labeled:
        letter = r["gt_label"][2] if len(r["gt_label"]) > 2 else "X"
        by_letter.setdefault(letter, []).append(r)

    train_list, val_list, test_list = [], [], []
    remaining_pool = []

    for letter in sorted(by_letter.keys()):
        group = list(by_letter[letter])
        rng.shuffle(group)
        # Always place at least 1 plate of every letter into Train so all classes are known
        train_list.append(group[0])
        remaining_pool.extend(group[1:])

    rng.shuffle(remaining_pool)
    n_total = len(labeled)
    n_train_target = int(round(n_total * train_ratio))  # 80
    n_val_target = int(round(n_total * val_ratio))      # 17

    need_train = max(0, n_train_target - len(train_list))
    train_list.extend(remaining_pool[:need_train])
    rest = remaining_pool[need_train:]
    val_list.extend(rest[:n_val_target])
    test_list.extend(rest[n_val_target:])

    # Sort deterministically by base name inside each split
    train_list.sort(key=lambda r: r["base"])
    val_list.sort(key=lambda r: r["base"])
    test_list.sort(key=lambda r: r["base"])

    for r in train_list:
        r["split"] = "train"
    for r in val_list:
        r["split"] = "val"
    for r in test_list:
        r["split"] = "test"

    splits = {"train": train_list, "val": val_list, "test": test_list}

    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        meta = {
            "total_labeled_plates": n_total,
            "expanded_multiplier": len(DEGRADATION_TYPES),
            "total_expanded_plates": n_total * len(DEGRADATION_TYPES),
            "degradation_types": DEGRADATION_TYPES,
            "counts": {
                "train_clean": len(train_list),
                "val_clean": len(val_list),
                "test_clean": len(test_list),
                "train_expanded": len(train_list) * len(DEGRADATION_TYPES),
                "val_expanded": len(val_list) * len(DEGRADATION_TYPES),
                "test_expanded": len(test_list) * len(DEGRADATION_TYPES),
            },
            "splits": {
                k: [{"base": r["base"], "gt_label": r["gt_label"], "img_path": r["img_path"]} for r in v]
                for k, v in splits.items()
            },
        }
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return splits


def synthesize_degradation_by_type(
    clean_img: np.ndarray,
    deg_type: str,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    Apply a specific Chapter 3 degradation model (`Doc/3. Image Degradation- Restoration and Resconstruction.pdf`)
    to a clean BGR image:
    - 'clean'       : Unmodified image
    - 'awgn'        : Additive White Gaussian Noise (sigma=10.0, Ch. 3.2 Slide 7)
    - 'salt_pepper' : Impulse Salt & Pepper Noise (pa=0.02, pb=0.02, Ch. 3.2 Slide 12)
    - 'rayleigh'    : Skewed Rayleigh Noise (a=-8.0, b=160.0, Ch. 3.2 Slide 8)
    - 'periodic'    : 2D Sinusoidal Interference (amplitude=24.0, u0=24, v0=18, Ch. 3.3 Slides 18-22)
    - 'motion_blur' : Linear Motion Blur (L=7, theta=12 deg) + AWGN (sigma=7.0, Ch. 3.4 & 3.6-3.7)
    """
    if deg_type == "clean":
        return clean_img.copy(), {"deg_type": "clean", "psf": None}
    elif deg_type == "awgn":
        deg = add_gaussian_noise(clean_img, mean=0.0, sigma=10.0, seed=seed)
        return deg, {"deg_type": "awgn", "psf": None, "sigma": 10.0}
    elif deg_type == "salt_pepper":
        deg = add_salt_pepper_noise(clean_img, pa=0.02, pb=0.02, seed=seed)
        return deg, {"deg_type": "salt_pepper", "psf": None, "pa": 0.02, "pb": 0.02}
    elif deg_type == "rayleigh":
        deg = add_rayleigh_noise(clean_img, a=-8.0, b=160.0, seed=seed)
        return deg, {"deg_type": "rayleigh", "psf": None}
    elif deg_type == "periodic":
        deg, freq_peaks = add_periodic_noise(clean_img, amplitude=24.0, u0=24, v0=18, return_peaks=True)
        return deg, {"deg_type": "periodic", "psf": None, "freq_peaks": freq_peaks}
    elif deg_type == "motion_blur":
        deg, psf = synthesize_degraded_image(
            clean_img, blur_len=7, blur_angle=12.0, noise_sigma=7.0, seed=seed
        )
        return deg, {"deg_type": "motion_blur", "psf": psf}
    raise ValueError(f"Unknown degradation type: {deg_type}")


def restore_by_degradation_type(
    degraded_img: np.ndarray,
    deg_type: str = "motion_blur",
    psf: np.ndarray | None = None,
    freq_peaks: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Chapter 3 Restoration Module (`Doc/3. Image Degradation- Restoration and Resconstruction.pdf`):
    Applies the theoretically optimal classical restoration operator for each degradation model:
    - 'awgn'        : 2D Haar Wavelet Soft-Threshold Denoising (Ch. 3.5 Slides 30-31) + Highboost
    - 'salt_pepper' : Center-Weighted Median Filter (Ch. 2.2 Slide 13 & Ch. 3.3)
    - 'rayleigh'    : Harmonic Mean Filter (Ch. 3.3 Slide 15) + Wavelet Denoising
    - 'periodic'    : 2D Frequency-Domain Butterworth Notch Reject Filter (Ch. 3.3 Slides 18-22)
    - 'motion_blur' : 2D Wavelet Denoising + Regularized Wiener / Constrained Least Squares (CLS)
                      Deconvolution (Ch. 3.6-3.7 Slides 32-38) + Unsharp Highboost
    """
    if deg_type == "clean":
        return degraded_img.copy(), {"restored": degraded_img.copy()}
    elif deg_type == "awgn":
        den = wavelet_denoise(degraded_img, sigma_est=10.0, levels=1)
        rest = unsharp_masking_highboost(den, k_boost=0.35, sigma=1.0)
        return rest, {"denoised": den, "restored": rest}
    elif deg_type == "salt_pepper":
        rest = weighted_median_filter(degraded_img, center_weight=3)
        return rest, {"restored": rest}
    elif deg_type == "rayleigh":
        harm = harmonic_mean_filter(degraded_img, ksize=3)
        rest = unsharp_masking_highboost(harm, k_boost=0.45, sigma=1.0)
        return rest, {"harmonic": harm, "restored": rest}
    elif deg_type == "periodic":
        peaks = freq_peaks if freq_peaks is not None else [(24, 18)]
        rest, spec_before, spec_after = butterworth_notch_reject_filter(
            degraded_img, notch_Step_uv=peaks, d0=6.0, order=2
        )
        return rest, {"restored": rest, "spec_before": spec_before, "spec_after": spec_after}
    else:
        # Default / 'motion_blur': Full Wavelet + Wiener/CLS Deconvolution + Highboost
        return full_restoration_pipeline(
            degraded_img, psf=psf, wiener_K=0.008, gamma_cls=0.005, use_wavelet=True, use_clahe=False
        )


def build_character_dataset(
    records: list[dict],
    save_dir: str | None = None,
    include_augmented_degradations: bool = False,
) -> tuple[list[dict], list[str]]:
    """
    Extract aligned 32x32 character patches and ground-truth labels across `records`.
    If `include_augmented_degradations=True`, expands each plate across the 6 Chapter 3
    degradation models (after applying Chapter 3 restoration), multiplying the character
    dataset by up to 6x for robust training/evaluation.
    """
    all_char_items = []
    all_labels = []
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
    deg_modes = DEGRADATION_TYPES if include_augmented_degradations else ["clean"]

    for rec_idx, rec in enumerate(records):
        gt = rec["gt_label"]
        if not gt:
            continue
        clean_img = cv2.imread(rec["img_path"])
        if clean_img is None:
            continue

        for d_idx, deg_type in enumerate(deg_modes):
            deg_img, deg_info = synthesize_degradation_by_type(
                clean_img, deg_type, seed=1000 + rec_idx * 17 + d_idx
            )
            if deg_type == "clean":
                proc_img = deg_img
            else:
                proc_img, _ = restore_by_degradation_type(
                    deg_img,
                    deg_type=deg_type,
                    psf=deg_info.get("psf"),
                    freq_peaks=deg_info.get("freq_peaks"),
                )

            det = detect_license_plate(proc_img)
            if det is None:
                continue
            _, bbox, cnt = det
            plate_bgr, _ = deskew_and_crop_plate(proc_img, bbox, cnt)
            char_items, _ = segment_characters(plate_bgr, method="adaptive")
            if len(char_items) != len(gt):
                continue
                
            for idx, item in enumerate(char_items):
                ch = gt[idx]
                item["plate_base"] = rec["base"]
                item["split"] = rec.get("split", "all")
                item["deg_type"] = deg_type
                item["gt_char"] = ch
                all_char_items.append(item)
                all_labels.append(ch)
                if save_dir is not None and deg_type == "clean":
                    ch_dir = os.path.join(save_dir, ch)
                    os.makedirs(ch_dir, exist_ok=True)
                    fname = f"{rec['base']}_pos{idx}_{ch}.png"
                    cv2.imwrite(os.path.join(ch_dir, fname), item["gray_32"])
                    
    return all_char_items, all_labels


class LicensePlatePipeline:
    """
    4-Stage Classical License Plate Restoration & Recognition Pipeline.
    """

    def __init__(self, recognizer: ClassicalCharRecognizer | None = None):
        self.recognizer = recognizer or ClassicalCharRecognizer(feature_mode="combined", classifier_type="svm")

    def fit_from_records(
        self,
        records: list[dict],
        save_char_dir: str | None = None,
        include_augmented_degradations: bool = False,
    ):
        char_items, labels = build_character_dataset(
            records,
            save_dir=save_char_dir,
            include_augmented_degradations=include_augmented_degradations,
        )
        self.recognizer.fit(char_items, labels)
        return char_items, labels

    def run(
        self,
        img: np.ndarray,
        enable_restoration: bool = False,
        deg_type: str = "motion_blur",
        psf: np.ndarray | None = None,
        freq_peaks: list[tuple[int, int]] | None = None,
        binarize_method: str = "adaptive",
        interp_mode: str = "bicubic",
        apply_syntax: bool = True,
    ) -> dict:
        """
        Execute the full 4-module pipeline on a single input image.
        """
        result = {"input_img": img.copy(), "enable_restoration": enable_restoration, "deg_type": deg_type}
        
        # Stage 1: Image Restoration & Contrast Enhancement (Chapter 3)
        if enable_restoration:
            proc_img, rest_stages = restore_by_degradation_type(
                img, deg_type=deg_type, psf=psf, freq_peaks=freq_peaks
            )
            result["restoration_stages"] = rest_stages
        else:
            proc_img = img.copy()
            result["restoration_stages"] = {"restored": proc_img}
        result["proc_img"] = proc_img
        
        # Stage 2: License Plate Detection & Geometric Deskewing (Chapters 2, 7, 11)
        det_res, det_maps = detect_license_plate(proc_img, return_debug=True)
        result["detection_maps"] = det_maps
        if det_res is None:
            result["detected"] = False
            result["plate_str"] = ""
            result["formatted_plate"] = "N/A"
            result["char_items"] = []
            return result
            
        score, bbox, cnt = det_res
        plate_bgr, skew_angle = deskew_and_crop_plate(proc_img, bbox, cnt, interp_mode=interp_mode)
        result["detected"] = True
        result["bbox"] = bbox
        result["det_score"] = score
        result["skew_angle"] = skew_angle
        result["plate_bgr"] = plate_bgr
        
        # Stage 3: Binarization (Otsu / K-Means / Adaptive) & 8-Connected Component Segmentation (Chapters 1, 9)
        char_items, seg_debug = segment_characters(plate_bgr, method=binarize_method)
        result["char_items"] = char_items
        result["segmentation_debug"] = seg_debug
        
        # Stage 4: Feature Extraction & Pattern Classification (Chapters 4, 10, 12)
        plate_str, char_preds = self.recognizer.predict_chars(char_items, apply_syntax=apply_syntax)
        result["plate_str"] = plate_str
        result["formatted_plate"] = format_vietnamese_plate(plate_str)
        result["char_preds"] = char_preds
        return result
