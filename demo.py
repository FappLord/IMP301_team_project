"""
IMP302 Interactive Video & Image License Plate Restoration & Extraction GUI Studio
==================================================================================
Provides an interactive graphical user interface (Tkinter + OpenCV + PIL) to:
1. Input any Video file (.mp4, .avi, .mov, .mkv, .webm) or Image file (.jpg, .png)
2. Optionally load built-in Sample Surveillance Videos (Clean or Motion-Blurred + AWGN)
3. Apply Chapter 3 Image Restoration (Wavelet + Wiener/CLS Deconvolution, Weighted Median,
   Butterworth Notch Reject) and Chapter 1 Multi-Frame Temporal Averaging / Voting
4. Visualize all 4 Classical Modules in real time:
   - Main Video Stream with Bounding Box, Skew Angle & Recognized Plate Overlay
   - Module 2: Bicubic Deskewed Plate ROI (160x128)
   - Module 3: Binarized Plate (Adaptive / Otsu Ch.1.2 / K-Means Ch.9.1) & 8-Connected Color Label Map
   - Module 4: Segmented 32x32 Character Strip with Per-Character Predictions
   - Automatic Video Plate Extraction Log Table (with Snapshot Inspector & CSV Export)

Usage:
    python demo.py                 # Opens the Interactive Video & Image GUI Studio
    python demo.py --video path.mp4
    python demo.py --cli           # Runs CLI batch benchmark on Held-Out Test Split
"""

import os
import csv
import time
import argparse
from collections import Counter, deque
import cv2
import numpy as np

from src.restoration import evaluate_quality
from src.recognition import ClassicalCharRecognizer, format_vietnamese_plate
from src.pipeline import (
    load_dataset,
    create_dataset_splits,
    synthesize_degradation_by_type,
    build_character_dataset,
    LicensePlatePipeline,
)


def ensure_sample_videos(test_recs: list[dict], out_dir: str = "data/sample_videos") -> tuple[str, str]:
    """
    Generate two realistic surveillance video streams (.mp4) from Held-Out Test Split plates
    if they do not already exist:
    1. `motorbike_surveillance_clean.mp4` (Clean camera stream with slight camera pan/zoom)
    2. `motorbike_surveillance_degraded.mp4` (Degraded by Chapter 3 Linear Motion Blur L=7 + AWGN)
    """
    os.makedirs(out_dir, exist_ok=True)
    clean_mp4 = os.path.join(out_dir, "motorbike_surveillance_clean.mp4")
    deg_mp4 = os.path.join(out_dir, "motorbike_surveillance_degraded.mp4")

    if os.path.exists(clean_mp4) and os.path.exists(deg_mp4) and os.path.getsize(clean_mp4) > 50000:
        return clean_mp4, deg_mp4

    print("[Video Init] Generating sample surveillance videos in `data/sample_videos/`...")
    W, H = 472, 303
    fps = 12
    frames_per_vehicle = 18  # 1.5 seconds per motorbike passing the gate

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_clean = cv2.VideoWriter(clean_mp4, fourcc, fps, (W, H))
    writer_deg = cv2.VideoWriter(deg_mp4, fourcc, fps, (W, H))

    selected_recs = test_recs[:8]
    for v_idx, rec in enumerate(selected_recs):
        base_img = cv2.imread(rec["img_path"])
        if base_img is None:
            continue
        base_img = cv2.resize(base_img, (W, H))
        for f_i in range(frames_per_vehicle):
            # Simulate subtle camera motion (panning dx, dy) across consecutive video frames
            progress = (f_i - frames_per_vehicle / 2.0) / float(frames_per_vehicle)
            dx = progress * 10.0
            dy = np.sin(progress * np.pi) * 4.0
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame_clean = cv2.warpAffine(base_img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            writer_clean.write(frame_clean)

            # Apply Chapter 3 Motion Blur + AWGN with frame-varying noise seed
            frame_deg, _ = synthesize_degradation_by_type(
                frame_clean, "motion_blur", seed=1000 + v_idx * 100 + f_i
            )
            writer_deg.write(frame_deg)

    writer_clean.release()
    writer_deg.release()
    print(f"[Video Init] Saved `{clean_mp4}` & `{deg_mp4}`")
    return clean_mp4, deg_mp4


def render_char_strip_canvas(char_items: list[dict], char_preds: list[tuple[str, float]], width: int = 440, height: int = 86) -> np.ndarray:
    """Render segmented 32x32 characters side-by-side with their predicted class and confidence."""
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    if not char_items:
        cv2.putText(canvas, "Chua phat hien ky tu (No characters segmented)", (14, height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
        return canvas

    n = len(char_items)
    slot_w = min(46, max(36, (width - 12) // max(1, n)))
    start_x = max(6, (width - n * slot_w) // 2)

    for idx, it in enumerate(char_items):
        x0 = start_x + idx * slot_w
        patch = it["bin_32"]
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        border_col = (80, 220, 100) if it.get("row", 0) == 0 else (60, 180, 255)
        patch_bordered = cv2.copyMakeBorder(patch_bgr, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=border_col)
        ph, pw = patch_bordered.shape[:2]
        y0 = 6
        if x0 + pw <= width and y0 + ph <= height:
            canvas[y0:y0 + ph, x0:x0 + pw] = patch_bordered

        if idx < len(char_preds):
            ch, conf = char_preds[idx]
            cv2.putText(canvas, ch, (x0 + 8, y0 + ph + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 140), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{int(round(conf * 100))}%", (x0 + 2, y0 + ph + 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


class LicensePlateVideoStudioGUI:
    """
    Interactive Desktop GUI for Video & Image License Plate Restoration, Segmentation & Recognition.
    """

    def __init__(self, initial_video_path: str | None = None):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        from PIL import Image, ImageTk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.Image = Image
        self.ImageTk = ImageTk

        self.root = tk.Tk()
        self.root.title("IMP302 Studio — Khôi Phục & Trích Xuất Biển Số Xe Từ Video / Ảnh (Classical Image Processing)")
        self.root.geometry("1420x860")
        self.root.configure(bg="#181a1f")

        # Load dataset & pre-train models on Train Split
        print("[GUI Init] Loading dataset and training recognizer...")
        self.records = load_dataset("dataset", use_cleaned_labels=True)
        self.splits = create_dataset_splits(self.records, train_ratio=0.70, val_ratio=0.15, seed=42)
        self.clean_sample_mp4, self.deg_sample_mp4 = ensure_sample_videos(self.splits["test"])

        # Train default SVM recognizer at startup for fast ~2s launch; lazy-train others on selection
        self.train_chars, self.train_labels = build_character_dataset(
            self.splits["train"], include_augmented_degradations=False
        )
        self.recognizers = {}
        svm_rec = ClassicalCharRecognizer(feature_mode="combined", classifier_type="svm")
        svm_rec.fit(self.train_chars, self.train_labels)
        self.recognizers["svm"] = svm_rec

        self.pipeline = LicensePlatePipeline(self.recognizers["svm"])

        # Video playback & temporal state
        self.cap = None
        self.video_path = None
        self.is_playing = False
        self.total_frames = 0
        self.current_frame_idx = 0
        self.video_fps = 15.0
        self.current_raw_frame = None

        # Multi-frame temporal buffer (Chapter 1.1 Slide 12 Image Averaging & Temporal Majority Voting)
        self.frame_buffer = deque(maxlen=3)
        self.plate_vote_window = deque(maxlen=7)
        self.extracted_plates_log = {}  # plate_str -> dict record
        self.log_snapshots = {}         # item_id -> dict of images

        self._build_ui()

        # Load initial video
        start_vid = initial_video_path if (initial_video_path and os.path.exists(initial_video_path)) else self.clean_sample_mp4
        self.load_video_file(start_vid, auto_play=False)

    def _build_ui(self):
        tk = self.tk
        ttk = self.ttk

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#21252b", foreground="#e6edf3", fieldbackground="#21252b", rowheight=26, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background="#2c313c", foreground="#79c0ff", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#1f6feb")])

        # Top Header Bar
        header = tk.Frame(self.root, bg="#111318", height=54, pady=6, padx=12)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="🎥 IMP302 STUDIO: KHÔI PHỤC ẢNH (CH. 3) & TRÍCH XUẤT BIỂN SỐ XE TỪ VIDEO / CAMERA",
            font=("Segoe UI", 13, "bold"),
            fg="#58a6ff",
            bg="#111318",
        ).pack(side=tk.LEFT)
        self.lbl_status = tk.Label(
            header,
            text="Sẵn sàng | Đã huấn luyện bộ phân loại cổ điển (Ch. 12)",
            font=("Segoe UI", 10),
            fg="#8b949e",
            bg="#111318",
        )
        self.lbl_status.pack(side=tk.RIGHT)

        # Control Toolbar Row 1: File & Playback Controls
        toolbar = tk.Frame(self.root, bg="#21252b", pady=7, padx=10)
        toolbar.pack(fill=tk.X, padx=8, pady=(6, 3))

        btn_cfg = {"font": ("Segoe UI", 9, "bold"), "bd": 0, "padx": 10, "pady": 5, "cursor": "hand2"}
        tk.Button(toolbar, text="📂 Chọn Video (.mp4/.avi)", bg="#238636", fg="white",
                  command=self.browse_video, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="🎬 Mở Video Mẫu (Sạch)", bg="#1f6feb", fg="white",
                  command=lambda: self.load_video_file(self.clean_sample_mp4, deg_override="none"), **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="🌧️ Mở Video Mẫu (Nhòe Chuyển Động Ch.3)", bg="#9e6a03", fg="white",
                  command=lambda: self.load_video_file(self.deg_sample_mp4, deg_override="motion_blur", auto_restore=True), **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="🖼️ Chọn Ảnh Lẻ (.jpg/.png)", bg="#388bfd", fg="white",
                  command=self.browse_image, **btn_cfg).pack(side=tk.LEFT, padx=4)

        tk.Frame(toolbar, width=16, bg="#21252b").pack(side=tk.LEFT)
        self.btn_play = tk.Button(toolbar, text="▶️ Phát Video", bg="#2ea043", fg="white",
                                  command=self.toggle_play, **btn_cfg)
        self.btn_play.pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="⏭️ +1 Frame", bg="#30363d", fg="#e6edf3",
                  command=self.step_one_frame, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="🔄 Đặt Lại Lịch Sử", bg="#30363d", fg="#e6edf3",
                  command=self.clear_history, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="💾 Xuất CSV Biển Số", bg="#8957e5", fg="white",
                  command=self.export_csv, **btn_cfg).pack(side=tk.RIGHT, padx=4)

        # Control Toolbar Row 2: Classical Image Processing Parameters (Chapters 1, 3, 9, 12)
        param_bar = tk.Frame(self.root, bg="#1c2026", pady=6, padx=10)
        param_bar.pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Label(param_bar, text="Giả lập Suy giảm (Ch. 3):", bg="#1c2026", fg="#c9d1d9", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(2, 4))
        self.var_sim_deg = tk.StringVar(value="none")
        cb_deg = ttk.Combobox(
            param_bar,
            textvariable=self.var_sim_deg,
            values=["none", "motion_blur", "salt_pepper", "periodic", "awgn", "rayleigh"],
            width=13,
            state="readonly",
        )
        cb_deg.pack(side=tk.LEFT, padx=(0, 12))
        cb_deg.bind("<<ComboboxSelected>>", lambda e: self.reprocess_current_frame())

        self.var_restore = tk.BooleanVar(value=True)
        tk.Checkbutton(
            param_bar,
            text="✨ Bật Khôi Phục Ảnh (Ch. 3: Wavelet + Wiener/CLS / Weighted Median / Notch)",
            variable=self.var_restore,
            bg="#1c2026",
            fg="#58a6ff",
            selectcolor="#111318",
            activebackground="#1c2026",
            font=("Segoe UI", 9, "bold"),
            command=self.reprocess_current_frame,
        ).pack(side=tk.LEFT, padx=6)

        self.var_temporal = tk.BooleanVar(value=True)
        tk.Checkbutton(
            param_bar,
            text="🎞️ Bỏ Phiếu Đa Khung Hình (Temporal Voting)",
            variable=self.var_temporal,
            bg="#1c2026",
            fg="#3fb950",
            selectcolor="#111318",
            activebackground="#1c2026",
            font=("Segoe UI", 9, "bold"),
            command=self.reprocess_current_frame,
        ).pack(side=tk.LEFT, padx=6)

        tk.Label(param_bar, text="| Nhị phân hóa (Ch.1 & 9):", bg="#1c2026", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)
        self.var_binarize = tk.StringVar(value="adaptive")
        cb_bin = ttk.Combobox(param_bar, textvariable=self.var_binarize, values=["adaptive", "otsu", "kmeans"], width=10, state="readonly")
        cb_bin.pack(side=tk.LEFT, padx=4)
        cb_bin.bind("<<ComboboxSelected>>", lambda e: self.reprocess_current_frame())

        tk.Label(param_bar, text="| Phân loại (Ch.12):", bg="#1c2026", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)
        self.var_clf = tk.StringVar(value="svm")
        cb_clf = ttk.Combobox(param_bar, textvariable=self.var_clf, values=["svm", "mlp", "bayes", "knn", "min_dist", "corr"], width=9, state="readonly")
        cb_clf.pack(side=tk.LEFT, padx=4)
        cb_clf.bind("<<ComboboxSelected>>", lambda e: self.on_change_classifier())

        # Main Split Body: Left (Video Player + Timeline) | Right (4-Module Inspector + Plate Log Table)
        body = tk.Frame(self.root, bg="#181a1f")
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)

        # --- LEFT COLUMN: MAIN VIDEO VIEWPORT ---
        left_col = tk.Frame(body, bg="#21252b", bd=1, relief=tk.SOLID)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self.lbl_video_title = tk.Label(
            left_col,
            text="KHUNG HÌNH VIDEO & ĐỊNH VỊ BIỂN SỐ (MODULE 1 & MODULE 2)",
            font=("Segoe UI", 10, "bold"),
            bg="#2c313c",
            fg="#e6edf3",
            pady=5,
        )
        self.lbl_video_title.pack(fill=tk.X)

        self.lbl_video_canvas = tk.Label(left_col, bg="#0d1117")
        self.lbl_video_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Timeline Slider Bar
        seek_frame = tk.Frame(left_col, bg="#21252b", padx=10, pady=6)
        seek_frame.pack(fill=tk.X)
        self.lbl_frame_counter = tk.Label(seek_frame, text="Frame: 0 / 0", bg="#21252b", fg="#8b949e", font=("Consolas", 10))
        self.lbl_frame_counter.pack(side=tk.LEFT, padx=(0, 8))

        self.var_seek = tk.IntVar(value=0)
        self.scale_seek = tk.Scale(
            seek_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.var_seek,
            showvalue=False,
            bg="#21252b",
            fg="#58a6ff",
            troughcolor="#0d1117",
            highlightthickness=0,
            command=self.on_seek_drag,
        )
        self.scale_seek.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- RIGHT COLUMN: MODULE 2, 3, 4 INSPECTOR & EXTRACTED PLATES LOG ---
        right_col = tk.Frame(body, bg="#21252b", width=620, bd=1, relief=tk.SOLID)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(4, 0))
        right_col.pack_propagate(False)

        tk.Label(
            right_col,
            text="KẾT QUẢ TRÍCH XUẤT & PHÂN TÍCH TỪNG MODULE (CH. 1, 2, 3, 4, 9, 10, 12)",
            font=("Segoe UI", 10, "bold"),
            bg="#2c313c",
            fg="#e6edf3",
            pady=5,
        ).pack(fill=tk.X)

        # Big Recognized Plate Readout Box
        readout_box = tk.Frame(right_col, bg="#111318", pady=8, padx=12)
        readout_box.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(readout_box, text="BIỂN SỐ NHẬN DẠNG:", bg="#111318", fg="#8b949e", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.lbl_big_plate = tk.Label(
            readout_box,
            text="---",
            bg="#111318",
            fg="#3fb950",
            font=("Consolas", 22, "bold"),
            padx=12,
        )
        self.lbl_big_plate.pack(side=tk.LEFT)

        self.lbl_conf_info = tk.Label(
            readout_box,
            text="Độ tự tin: --%\nTrạng thái: Đang chờ",
            bg="#111318",
            fg="#79c0ff",
            font=("Segoe UI", 9),
            justify=tk.LEFT,
        )
        self.lbl_conf_info.pack(side=tk.RIGHT)

        # Row of 3 Intermediate Sub-Images (Deskewed ROI | Binary | 8-Connected Regions)
        sub_imgs_frame = tk.Frame(right_col, bg="#21252b")
        sub_imgs_frame.pack(fill=tk.X, padx=8, pady=4)

        self.lbl_sub1 = self._create_sub_panel(sub_imgs_frame, "1. Biển Số Cắt & Xoay Bicubic\n(160x128 - Ch. 1.1 & Ch. 7)")
        self.lbl_sub2 = self._create_sub_panel(sub_imgs_frame, "2. Nhị Phân Hóa & Lọc Viền\n(Otsu / K-Means / Adaptive)")
        self.lbl_sub3 = self._create_sub_panel(sub_imgs_frame, "3. Gán Nhãn Vùng 8 Hướng\n(8-Connected CCL - Ch. 1.2)")

        # Segmented 32x32 Characters Strip
        char_frame = tk.Frame(right_col, bg="#161b22", bd=1, relief=tk.SOLID)
        char_frame.pack(fill=tk.X, padx=8, pady=5)
        tk.Label(
            char_frame,
            text="4. Các Ký Tự 32x32 Đã Tách & Dự Đoán Từng Vị Trí (Module 4: WHT + HOG + Hu/Fourier/Euler):",
            bg="#161b22",
            fg="#c9d1d9",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
            padx=6,
            pady=2,
        ).pack(fill=tk.X)
        self.lbl_char_strip = tk.Label(char_frame, bg="#1c1f26", height=86)
        self.lbl_char_strip.pack(fill=tk.X, padx=4, pady=4)

        # Extracted License Plates History Table (Treeview)
        log_frame = tk.Frame(right_col, bg="#21252b")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        tk.Label(
            log_frame,
            text="📋 DANH SÁCH BIỂN SỐ XE ĐÃ TRÍCH XUẤT TRONG VIDEO (Click vào dòng để xem lại ảnh biển số):",
            bg="#21252b",
            fg="#79c0ff",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 4))

        cols = ("stt", "frame_time", "plate", "chars", "conf", "hits", "mode")
        self.tree = ttk.Treeview(log_frame, columns=cols, show="headings", height=9)
        self.tree.heading("stt", text="#")
        self.tree.heading("frame_time", text="Frame / Giây")
        self.tree.heading("plate", text="Biển Số Xe")
        self.tree.heading("chars", text="Số Ký Tự")
        self.tree.heading("conf", text="Độ Tự Tin")
        self.tree.heading("hits", text="Số Frame")
        self.tree.heading("mode", text="Chế Độ Xử Lý")

        self.tree.column("stt", width=36, anchor=tk.CENTER)
        self.tree.column("frame_time", width=95, anchor=tk.CENTER)
        self.tree.column("plate", width=130, anchor=tk.CENTER)
        self.tree.column("chars", width=65, anchor=tk.CENTER)
        self.tree.column("conf", width=75, anchor=tk.CENTER)
        self.tree.column("hits", width=65, anchor=tk.CENTER)
        self.tree.column("mode", width=115, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_log_row)

    def _create_sub_panel(self, parent, title: str):
        tk = self.tk
        frame = tk.Frame(parent, bg="#161b22", bd=1, relief=tk.SOLID)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        tk.Label(frame, text=title, bg="#161b22", fg="#8b949e", font=("Segoe UI", 8, "bold"), pady=3).pack(fill=tk.X)
        lbl = tk.Label(frame, bg="#0d1117")
        lbl.pack( padx=4, pady=4)
        return lbl

    def on_change_classifier(self):
        c_type = self.var_clf.get()
        if c_type not in self.recognizers:
            rec_obj = ClassicalCharRecognizer(feature_mode="combined", classifier_type=c_type)
            rec_obj.fit(self.train_chars, self.train_labels)
            self.recognizers[c_type] = rec_obj
        self.pipeline.recognizer = self.recognizers[c_type]
        self.reprocess_current_frame()

    def browse_video(self):
        path = self.filedialog.askopenfilename(
            title="Chọn File Video Đầu Vào",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.load_video_file(path, deg_override="none", auto_play=True)

    def browse_image(self):
        path = self.filedialog.askopenfilename(
            title="Chọn Ảnh Phương Tiện Đầu Vào",
            initialdir="dataset/img",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.is_playing = False
            self.btn_play.configure(text="▶️ Phát Video", bg="#2ea043")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            img = cv2.imread(path)
            if img is not None:
                self.video_path = path
                self.total_frames = 1
                self.current_frame_idx = 0
                self.current_raw_frame = img
                self.frame_buffer.clear()
                self.plate_vote_window.clear()
                self.process_and_display_frame(img)
                self.lbl_status.configure(text=f"Đang xem ảnh tĩnh: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

    def load_video_file(self, path: str, deg_override: str | None = None, auto_restore: bool | None = None, auto_play: bool = True):
        self.is_playing = False
        if self.cap is not None:
            self.cap.release()

        if deg_override is not None:
            self.var_sim_deg.set(deg_override)
        if auto_restore is not None:
            self.var_restore.set(auto_restore)

        self.cap = cv2.VideoCapture(path)
        self.video_path = path
        self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.video_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 15.0)
        if self.video_fps <= 1.0 or self.video_fps > 120.0:
            self.video_fps = 15.0

        self.current_frame_idx = 0
        self.scale_seek.configure(to=max(1, self.total_frames - 1))
        self.frame_buffer.clear()
        self.plate_vote_window.clear()

        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.current_raw_frame = frame
            self.process_and_display_frame(frame)

        self.lbl_status.configure(
            text=f"Đã mở video: {os.path.basename(path)} | {self.total_frames} frames @ {self.video_fps:.1f} FPS"
        )
        if auto_play:
            self.is_playing = True
            self.btn_play.configure(text="⏸️ Tạm Dừng", bg="#d29922")
            self._play_loop()
        else:
            self.btn_play.configure(text="▶️ Phát Video", bg="#2ea043")

    def toggle_play(self):
        if self.cap is None:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.configure(text="⏸️ Tạm Dừng", bg="#d29922")
            self._play_loop()
        else:
            self.btn_play.configure(text="▶️ Phát Video", bg="#2ea043")

    def step_one_frame(self):
        self.is_playing = False
        self.btn_play.configure(text="▶️ Phát Video", bg="#2ea043")
        self._read_next_frame()

    def on_seek_drag(self, val):
        if self.cap is None or self.is_playing:
            return
        target_idx = int(float(val))
        if abs(target_idx - self.current_frame_idx) >= 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_frame_idx = target_idx
                self.current_raw_frame = frame
                self.plate_vote_window.clear()
                self.process_and_display_frame(frame)

    def _read_next_frame(self) -> bool:
        if self.cap is None:
            return False
        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Loop back to start of video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self.plate_vote_window.clear()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False
        else:
            self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        self.var_seek.set(self.current_frame_idx)
        self.current_raw_frame = frame
        self.process_and_display_frame(frame)
        return True

    def _play_loop(self):
        if not self.is_playing:
            return
        t0 = time.perf_counter()
        ok = self._read_next_frame()
        elapsed_ms = int((time.perf_counter() - t0) * 1000.0)
        target_delay = max(15, int(1000.0 / self.video_fps) - elapsed_ms)
        if ok and self.is_playing:
            self.root.after(target_delay, self._play_loop)

    def reprocess_current_frame(self):
        if self.current_raw_frame is not None:
            self.process_and_display_frame(self.current_raw_frame)

    def process_and_display_frame(self, raw_frame: np.ndarray):
        t0 = time.perf_counter()
        sim_deg = self.var_sim_deg.get()
        enable_rest = bool(self.var_restore.get())
        use_temporal = bool(self.var_temporal.get())
        bin_method = self.var_binarize.get()

        # 1. Optional live degradation simulation (if user selected a degradation on a clean video)
        is_already_deg_video = self.video_path and "degraded" in os.path.basename(self.video_path).lower()
        if sim_deg != "none" and not is_already_deg_video:
            working_frame, deg_info = synthesize_degradation_by_type(
                raw_frame, sim_deg, seed=42 + self.current_frame_idx
            )
            active_deg = sim_deg
        elif is_already_deg_video:
            working_frame = raw_frame.copy()
            _, deg_info = synthesize_degradation_by_type(raw_frame, "motion_blur", seed=42)
            active_deg = "motion_blur" if sim_deg == "none" else sim_deg
        else:
            working_frame = raw_frame.copy()
            deg_info = {"psf": None, "freq_peaks": None}
            active_deg = "motion_blur"

        # 2. Run 4-Module Classical Pipeline
        res = self.pipeline.run(
            working_frame,
            enable_restoration=enable_rest and (sim_deg != "none" or is_already_deg_video),
            deg_type=active_deg,
            psf=deg_info.get("psf"),
            freq_peaks=deg_info.get("freq_peaks"),
            binarize_method=bin_method,
            apply_syntax=True,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        fps_inst = 1000.0 / max(1.0, dt_ms)

        raw_plate_str = res["plate_str"]
        char_preds = res.get("char_preds", [])
        mean_conf = float(np.mean([c for _, c in char_preds]) * 100.0) if char_preds else 0.0
        all_plates = res.get("all_plates", [])

        # 3. Temporal Majority Voting across consecutive video frames (valid 8 or 9 character plates)
        if len(raw_plate_str) in (8, 9) and len(all_plates) <= 1:
            self.plate_vote_window.append(raw_plate_str)

        if use_temporal and len(self.plate_vote_window) > 0 and len(all_plates) <= 1:
            most_common_plate, vote_count = Counter(self.plate_vote_window).most_common(1)[0]
            display_plate_fmt = format_vietnamese_plate(most_common_plate)
            active_plate_raw = most_common_plate
        else:
            display_plate_fmt = res["formatted_plate"]
            active_plate_raw = raw_plate_str
            vote_count = 1

        # 4. Draw annotations on Main Video Viewport (supports 1 or multiple plates per frame/image)
        vis_main = res["proc_img"].copy()
        if res["detected"]:
            plates_to_draw = all_plates if all_plates else [res]
            for p_idx, p_info in enumerate(plates_to_draw):
                bx, by, bw, bh = p_info["bbox"]
                p_preds = p_info.get("char_preds", [])
                p_conf = float(np.mean([c for _, c in p_preds]) * 100.0) if p_preds else mean_conf
                p_fmt = p_info["formatted_plate"] if len(plates_to_draw) > 1 else display_plate_fmt
                box_color = (0, 255, 80) if p_idx == 0 else (255, 200, 0)
                bg_color = (15, 110, 35) if p_idx == 0 else (140, 85, 10)
                cv2.rectangle(vis_main, (bx, by), (bx + bw, by + bh), box_color, 2)
                banner_txt = f"{p_fmt} ({p_conf:.0f}%)"
                (tw, th), _ = cv2.getTextSize(banner_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
                by_top = max(0, by - th - 10)
                bx_clamped = max(2, min(bx, vis_main.shape[1] - tw - 14))
                cv2.rectangle(vis_main, (bx_clamped, by_top), (bx_clamped + tw + 12, by), bg_color, -1)
                cv2.putText(vis_main, banner_txt, (bx_clamped + 6, by - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

        # HUD overlay on top-left of video
        mode_tag = f"RESTORED ({active_deg.upper()})" if (enable_rest and (sim_deg != "none" or is_already_deg_video)) else "DIRECT INPUT"
        hud_txt = f"Frame {self.current_frame_idx + 1}/{self.total_frames} | {mode_tag} | {dt_ms:.1f} ms ({fps_inst:.1f} FPS)"
        cv2.rectangle(vis_main, (6, 6), (455, 30), (15, 18, 24), -1)
        cv2.putText(vis_main, hud_txt, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 200, 255), 1, cv2.LINE_AA)

        self._set_tk_image(self.lbl_video_canvas, vis_main, target_size=(730, 470))
        self.lbl_frame_counter.configure(
            text=f"Frame: {self.current_frame_idx + 1}/{self.total_frames} ({self.current_frame_idx / max(1.0, self.video_fps):.1f}s)"
        )

        # 5. Update Module 2, 3, 4 Sub-Panels
        if res["detected"]:
            plate_bgr = res["plate_bgr"]
            seg_dbg = res["segmentation_debug"]
            clean_bin = cv2.cvtColor(seg_dbg["clean_binary"], cv2.COLOR_GRAY2BGR)
            color_ccl = seg_dbg["color_labels_clean"]
            char_strip = render_char_strip_canvas(res["char_items"], char_preds)

            self._set_tk_image(self.lbl_sub1, plate_bgr, target_size=(180, 144))
            self._set_tk_image(self.lbl_sub2, clean_bin, target_size=(180, 144))
            self._set_tk_image(self.lbl_sub3, color_ccl, target_size=(180, 144))
            self._set_tk_image(self.lbl_char_strip, char_strip, target_size=(580, 86))

            is_valid_len = len(active_plate_raw) in (8, 9)
            if len(all_plates) > 1:
                multi_txt = "  |  ".join(p["formatted_plate"] for p in all_plates)
                self.lbl_big_plate.configure(text=multi_txt, fg="#3fb950", font=("Consolas", 16, "bold"))
            else:
                self.lbl_big_plate.configure(
                    text=display_plate_fmt if display_plate_fmt else "N/A",
                    fg="#3fb950" if is_valid_len else "#f85149",
                    font=("Consolas", 22, "bold"),
                )
            self.lbl_conf_info.configure(
                text=f"Độ tự tin TB: {mean_conf:.1f}% | Ký tự: {len(res['char_items'])} ({len(all_plates)} biển)\n"
                     f"Góc nghiêng: {res['skew_angle']:+.1f}° | Ổn định: {vote_count} frame"
            )

            # Log all valid extracted plates into the Video Extraction Table (in left-to-right order)
            plates_to_log = list(reversed(all_plates)) if len(all_plates) > 1 else [res]
            for p_idx, p_info in enumerate(plates_to_log):
                p_raw = p_info["plate_str"] if len(plates_to_log) > 1 else active_plate_raw
                p_fmt = p_info["formatted_plate"] if len(plates_to_log) > 1 else display_plate_fmt
                p_items = p_info["char_items"]
                p_preds = p_info.get("char_preds", [])
                p_conf = float(np.mean([c for _, c in p_preds]) * 100.0) if p_preds else mean_conf
                if len(p_raw) in (8, 9) and p_conf >= 45.0:
                    p_bgr = p_info["plate_bgr"]
                    p_sdbg = p_info["segmentation_debug"]
                    p_cbin = cv2.cvtColor(p_sdbg["clean_binary"], cv2.COLOR_GRAY2BGR)
                    p_cccl = p_sdbg["color_labels_clean"]
                    p_strip = render_char_strip_canvas(p_items, p_preds)
                    self._update_plate_log(
                        plate_raw=p_raw,
                        plate_fmt=p_fmt,
                        num_chars=len(p_items),
                        conf=p_conf,
                        mode_tag=mode_tag,
                        snapshots={
                            "vis_main": vis_main,
                            "plate_bgr": p_bgr,
                            "clean_bin": p_cbin,
                            "color_ccl": p_cccl,
                            "char_strip": p_strip,
                        },
                    )
        else:
            # Clear right-hand panels when no plate is detected so stale results do not remain
            blank_sub = np.full((144, 180, 3), 24, dtype=np.uint8)
            cv2.putText(blank_sub, "No Plate", (42, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 130, 145), 1, cv2.LINE_AA)
            blank_strip = render_char_strip_canvas([], [])
            self._set_tk_image(self.lbl_sub1, blank_sub, target_size=(180, 144))
            self._set_tk_image(self.lbl_sub2, blank_sub, target_size=(180, 144))
            self._set_tk_image(self.lbl_sub3, blank_sub, target_size=(180, 144))
            self._set_tk_image(self.lbl_char_strip, blank_strip, target_size=(580, 86))
            self.lbl_big_plate.configure(text="KHÔNG TÌM THẤY", fg="#f85149", font=("Consolas", 20, "bold"))
            self.lbl_conf_info.configure(text="Độ tự tin TB: 0.0% | Ký tự: 0\nGóc nghiêng: 0.0° | Ổn định: 0 frame")

    def _update_plate_log(self, plate_raw: str, plate_fmt: str, num_chars: int, conf: float, mode_tag: str, snapshots: dict):
        sec = self.current_frame_idx / max(1.0, self.video_fps)
        frame_time_str = f"F{self.current_frame_idx + 1} ({sec:.1f}s)"

        if plate_raw in self.extracted_plates_log:
            entry = self.extracted_plates_log[plate_raw]
            entry["hits"] += 1
            if conf >= entry["conf"]:
                entry["conf"] = conf
                entry["snapshots"] = snapshots
            self.tree.item(
                entry["tree_id"],
                values=(
                    entry["stt"],
                    entry["first_seen"],
                    plate_fmt,
                    f"{num_chars} ký tự",
                    f"{entry['conf']:.1f}%",
                    f"{entry['hits']} frames",
                    mode_tag,
                ),
            )
            self.log_snapshots[entry["tree_id"]] = entry["snapshots"]
        else:
            stt = len(self.extracted_plates_log) + 1
            tree_id = self.tree.insert(
                "",
                0,  # Insert newest at top
                values=(
                    stt,
                    frame_time_str,
                    plate_fmt,
                    f"{num_chars} ký tự",
                    f"{conf:.1f}%",
                    "1 frame",
                    mode_tag,
                ),
            )
            self.extracted_plates_log[plate_raw] = {
                "stt": stt,
                "tree_id": tree_id,
                "first_seen": frame_time_str,
                "plate_raw": plate_raw,
                "plate_fmt": plate_fmt,
                "num_chars": num_chars,
                "conf": conf,
                "hits": 1,
                "mode": mode_tag,
                "snapshots": snapshots,
            }
            self.log_snapshots[tree_id] = snapshots

    def on_select_log_row(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        item_id = sel[0]
        snaps = self.log_snapshots.get(item_id)
        if not snaps:
            return
        # Pause video so user can inspect the selected plate snapshot
        self.is_playing = False
        self.btn_play.configure(text="▶️ Phát Video", bg="#2ea043")

        vals = self.tree.item(item_id, "values")
        self.lbl_big_plate.configure(text=vals[2], fg="#58a6ff")
        self.lbl_conf_info.configure(text=f"Đang xem lại bản ghi #{vals[0]} tại {vals[1]}\nĐộ tự tin: {vals[4]} ({vals[5]})")
        self._set_tk_image(self.lbl_video_canvas, snaps["vis_main"], target_size=(730, 470))
        self._set_tk_image(self.lbl_sub1, snaps["plate_bgr"], target_size=(180, 144))
        self._set_tk_image(self.lbl_sub2, snaps["clean_bin"], target_size=(180, 144))
        self._set_tk_image(self.lbl_sub3, snaps["color_ccl"], target_size=(180, 144))
        self._set_tk_image(self.lbl_char_strip, snaps["char_strip"], target_size=(580, 86))

    def clear_history(self):
        self.extracted_plates_log.clear()
        self.log_snapshots.clear()
        self.plate_vote_window.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.lbl_status.configure(text="Đã xóa lịch sử bảng trích xuất biển số.")

    def export_csv(self):
        if not self.extracted_plates_log:
            self.messagebox.showinfo("Thông báo", "Chưa có biển số nào trong danh sách để xuất CSV!")
            return
        os.makedirs("outputs", exist_ok=True)
        out_csv = "outputs/extracted_video_plates.csv"
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["STT", "Thoi_Gian_Frame", "Bien_So_Chuan_Hoa", "Bien_So_Tho", "So_Ky_Tu", "Do_Tu_Tin_Pct", "So_Frame_Xac_Nhan", "Che_Do_Xu_Ly"])
            for entry in sorted(self.extracted_plates_log.values(), key=lambda x: x["stt"]):
                writer.writerow([
                    entry["stt"],
                    entry["first_seen"],
                    entry["plate_fmt"],
                    entry["plate_raw"],
                    entry["num_chars"],
                    f"{entry['conf']:.2f}",
                    entry["hits"],
                    entry["mode"],
                ])
        self.messagebox.showinfo("Xuất CSV Thành Công", f"Đã lưu {len(self.extracted_plates_log)} biển số trích xuất vào:\n{os.path.abspath(out_csv)}")

    def _set_tk_image(self, label_widget, bgr_img: np.ndarray, target_size: tuple[int, int]):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
        pil_img = self.Image.fromarray(resized)
        tk_img = self.ImageTk.PhotoImage(image=pil_img)
        label_widget.configure(image=tk_img)
        label_widget.image = tk_img

    def run(self):
        self.root.mainloop()


def run_cli_demo(image_path: str | None = None):
    print("=" * 88)
    print("IMP302 END-TO-END LICENSE PLATE RESTORATION & RECOGNITION PIPELINE - CLI DEMO")
    print("=" * 88)
    records = load_dataset("dataset", use_cleaned_labels=True)
    splits = create_dataset_splits(records, train_ratio=0.70, val_ratio=0.15, seed=42)
    pipeline = LicensePlatePipeline()
    pipeline.fit_from_records(splits["train"], include_augmented_degradations=False)

    demo_list = splits["test"][:5] if image_path is None else [{"base": os.path.basename(image_path), "img_path": image_path, "gt_label": None}]
    for rec in demo_list:
        img = cv2.imread(rec["img_path"])
        gt = rec.get("gt_label")
        res = pipeline.run(img, enable_restoration=False)
        print(f"► {rec['base']} | GT: {format_vietnamese_plate(gt) if gt else 'N/A'} -> Pred: {res['formatted_plate']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMP302 Video & Image License Plate Restoration & Extraction Studio")
    parser.add_argument("--video", type=str, default=None, help="Path to an input video file (.mp4, .avi, .mov)")
    parser.add_argument("--image", type=str, default=None, help="Path to an input image file")
    parser.add_argument("--cli", action="store_true", help="Run non-GUI command-line benchmark")
    args = parser.parse_args()

    if args.cli:
        run_cli_demo(image_path=args.image)
    else:
        app = LicensePlateVideoStudioGUI(initial_video_path=args.video)
        app.run()
