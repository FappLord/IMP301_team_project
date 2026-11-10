"""
Live Demo Script for IMP302 License Plate Restoration, Segmentation & Recognition Pipeline
==========================================================================================
Strictly aligned with IMP302 Course Lecture Slides (`Doc/1..12.pdf`).
Trains the recognizer strictly on the Train Split (80 plates, 6x expanded = 480 plates)
and demonstrates end-to-end restoration & recognition on Held-Out Test Split images!

Usage:
    python demo.py
    python demo.py --image dataset/img/0000_02187_b.jpg
"""

import os
import time
import argparse
import cv2
import numpy as np

from src.restoration import evaluate_quality
from src.recognition import format_vietnamese_plate
from src.pipeline import (
    load_dataset,
    create_dataset_splits,
    synthesize_degradation_by_type,
    LicensePlatePipeline,
)


def run_demo(image_path: str | None = None, test_degraded: bool = True):
    print("=" * 88)
    print("IMP302 END-TO-END LICENSE PLATE RESTORATION & RECOGNITION PIPELINE - LIVE DEMO")
    print("=" * 88)

    t0 = time.perf_counter()
    records = load_dataset("dataset", use_cleaned_labels=True)
    splits = create_dataset_splits(records, train_ratio=0.70, val_ratio=0.15, seed=42)
    train_recs, val_recs, test_recs = splits["train"], splits["val"], splits["test"]

    pipeline = LicensePlatePipeline()
    char_items, labels = pipeline.fit_from_records(train_recs, include_augmented_degradations=True)
    train_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Dataset Split] Train={len(train_recs)} plates | Val={len(val_recs)} plates | Test={len(test_recs)} plates (Zero Leakage)")
    print(f"[6x Expansion ] Trained on {len(train_recs)*6} degraded/restored plates ({len(char_items)} char patches, {len(set(labels))} classes) in {train_ms:.1f} ms\n")

    if image_path is not None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        rec_match = [r for r in records if r["base"] == base]
        gt = rec_match[0]["gt_label"] if rec_match else None
        sp = rec_match[0].get("split", "custom") if rec_match else "custom"
        demo_list = [{"base": base, "img_path": image_path, "gt_label": gt, "split": sp}]
    else:
        # Pick 5 Held-Out Test Split images
        demo_list = test_recs[:5]

    for rec in demo_list:
        img = cv2.imread(rec["img_path"])
        gt = rec["gt_label"]
        gt_fmt = format_vietnamese_plate(gt) if gt else "Unknown"

        t_start = time.perf_counter()
        res_clean = pipeline.run(img, enable_restoration=False, binarize_method="adaptive", apply_syntax=True)
        dt_ms = (time.perf_counter() - t_start) * 1000.0

        status = "EXACT MATCH" if (gt and res_clean["plate_str"] == gt) else "PREDICTED"
        print(f"► Image: {rec['base']}.jpg (Split: {rec.get('split', 'test').upper()}) | Ground Truth: {gt_fmt}")
        print(f"  [Clean Input]        BBox={res_clean.get('bbox')} | Skew={res_clean.get('skew_angle', 0.0):+.1f}° "
              f"| SegChars={len(res_clean['char_items'])} | Predicted: {res_clean['formatted_plate']} [{status}] ({dt_ms:.1f} ms)")

        if test_degraded:
            for deg_type, deg_title in [
                ("motion_blur", "MotionBlur+AWGN (Ch.3.6-7)"),
                ("salt_pepper", "Salt&Pepper 4%  (Ch.3.2-3)"),
                ("periodic", "Periodic Noise  (Ch.3.3)  "),
            ]:
                deg_img, deg_info = synthesize_degradation_by_type(img, deg_type, seed=42)
                res_no_m1 = pipeline.run(deg_img, enable_restoration=False)
                res_with_m1 = pipeline.run(
                    deg_img,
                    enable_restoration=True,
                    deg_type=deg_type,
                    psf=deg_info.get("psf"),
                    freq_peaks=deg_info.get("freq_peaks"),
                )
                q_deg = evaluate_quality(img, deg_img)
                q_rest = evaluate_quality(img, res_with_m1["proc_img"])
                print(f"  [{deg_title}] No-Rest : PSNR={q_deg['psnr']:.2f} dB | Seg={len(res_no_m1['char_items'])} chars | Pred: {res_no_m1['formatted_plate']}")
                print(f"  [{deg_title}] Restored: PSNR={q_rest['psnr']:.2f} dB (+{q_rest['psnr']-q_deg['psnr']:.2f} dB) | Seg={len(res_with_m1['char_items'])} chars | Pred: {res_with_m1['formatted_plate']}")
        print("-" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMP302 Classical License Plate Restoration & Recognition Live Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to a specific vehicle image")
    parser.add_argument("--degrade", action="store_true", default=True, help="Test Chapter 3 synthetic degradations & restoration")
    args = parser.parse_args()
    run_demo(image_path=args.image, test_degraded=args.degrade)
