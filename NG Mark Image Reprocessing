import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import sys
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import math
import platform
import shutil
import logging
import cProfile, pstats, io
from PIL import Image, ImageTk
from turbojpeg import TurboJPEG, TJPF_BGR
import json
from tkinter import simpledialog
from pathlib import Path

SETTINGS_FILE = "ngmark_profiles.json"      



def _load_profiles() -> dict:
    """Return {'profile name': {param: value, …}, …} – empty dict if none."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_profiles(profiles: dict) -> None:
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profiles, f, indent=2)
    os.replace(tmp, SETTINGS_FILE)


GLOBAL_TURBOJPEG = TurboJPEG(lib_path=r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")
logging.basicConfig(filename='image_analysis.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_ALPHA = 1.9
DEFAULT_BETA = 90
DEFAULT_DOT_THRESH = 15
DEFAULT_MIN_AREA = 2
DEFAULT_MAX_AREA = 13
DEFAULT_MIN_CIRC = 0.6
DEFAULT_MAX_DIM = 800  
DEFAULT_BLACKHAT_THRESH = 60

DEFAULT_BATCH_SIZE = 10000  

GLOBAL_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
GLOBAL_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))


def custom_select_roi(original_image, window_name="ROI Selector", display_size=(1000, 800)):
    orig_h, orig_w = original_image.shape[:2]
    disp_w, disp_h = display_size
    initial_zoom = min(disp_w / orig_w, disp_h / orig_h, 1.0)
    effective_zoom = initial_zoom

    roi_start = None
    roi_end = None
    drawing = False
    roi_rect = None
    offset_x, offset_y = 0, 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_start, roi_end, drawing, roi_rect, offset_x, offset_y, effective_zoom
        img_width = int(orig_w * effective_zoom)
        img_height = int(orig_h * effective_zoom)
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < offset_x or x > offset_x + img_width or y < offset_y or y > offset_y + img_height:
                return
            roi_start = (x, y)
            roi_end = (x, y)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            roi_end = (x, y)
            drawing = False
            x0, y0 = roi_start
            x1, y1 = roi_end
            x0 = max(x0, offset_x)
            y0 = max(y0, offset_y)
            x1 = min(x1, offset_x + img_width)
            y1 = min(y1, offset_y + img_height)
            x_min = min(x0, x1)
            y_min = min(y0, y1)
            roi_rect = (x_min, y_min, abs(x1 - x0), abs(y1 - y0))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, mouse_callback)
    while True:
        new_width = int(orig_w * effective_zoom)
        new_height = int(orig_h * effective_zoom)
        resized_image = cv2.resize(original_image, (new_width, new_height))
        canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        offset_x = (disp_w - new_width) // 2
        offset_y = (disp_h - new_height) // 2
        canvas[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = resized_image
        if roi_start and roi_end:
            cv2.rectangle(canvas, roi_start, roi_end, (0, 255, 0), 2)
        overlay_text = "Highlight then press C to confirm ROI"
        cv2.putText(canvas, overlay_text, (10, disp_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, overlay_text, (10, disp_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            roi_rect = None
            break
        elif key == ord('c'):
            if roi_rect is None and roi_start and roi_end:
                x0, y0 = roi_start
                x1, y1 = roi_end
                x_min = min(x0, x1)
                y_min = min(y0, y1)
                roi_rect = (x_min, y_min, abs(x1 - x0), abs(y1 - y0))
            if roi_rect:
                break
    cv2.destroyWindow(window_name)
    if roi_rect:
        x_disp, y_disp, w_disp, h_disp = roi_rect
        x_in_resized = x_disp - offset_x
        y_in_resized = y_disp - offset_y
        x_orig = int(x_in_resized / effective_zoom)
        y_orig = int(y_in_resized / effective_zoom)
        w_orig = int(w_disp / effective_zoom)
        h_orig = int(h_disp / effective_zoom)
        print("Selected ROI in original image coordinates:", (x_orig, y_orig, w_orig, h_orig))
        return (x_orig, y_orig, w_orig, h_orig)
    else:
        print("ROI selection cancelled.")
        return None


def get_first_image_roi(image_folder, roi):
    image_files = list(get_all_images(image_folder))
    if not image_files:
        return None
    first_img = fast_imread(image_files[0])
    if first_img is None:
        return None
    x, y, w, h = roi
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, first_img.shape[1] - x)
    h = min(h, first_img.shape[0] - y)
    return first_img[y:y+h, x:x+w].copy()


def batch_iterator(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def enhance_brightness_contrast(image, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, use_cuda=False, apply_clahe=True):
    if use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)
        gpu_lab = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2Lab)
        lab_cpu = gpu_lab.download()
        l, a, b = cv2.split(lab_cpu)
        if apply_clahe:
            l = GLOBAL_CLAHE.apply(l)
        lab_cpu = cv2.merge((l, a, b))
        gpu_lab.upload(lab_cpu)
        gpu_bgr = cv2.cuda.cvtColor(gpu_lab, cv2.COLOR_Lab2BGR)
        gpu_scaled = cv2.cuda.multiply(gpu_bgr, np.array([alpha], dtype=np.float32))
        gpu_scaled = cv2.cuda.add(gpu_scaled, np.array([beta], dtype=np.float32))
        return gpu_scaled.download()
    else:
        if apply_clahe:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = GLOBAL_CLAHE.apply(lab[:, :, 0])
            enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced_img = image.copy()
        return cv2.convertScaleAbs(enhanced_img, alpha=alpha, beta=beta)

def fast_imread(image_path):
    return cv2.imread(image_path)

def get_all_images(root_folder):
    for entry in os.scandir(root_folder):
        if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            yield entry.path
        elif entry.is_dir():
            yield from get_all_images(entry.path)

def save_image(image, path, description):
    try:
        success = cv2.imwrite(path, image)
        if not success:
            logging.error(f"Failed to save {description} image: {path}")
    except Exception as e:
        logging.error(f"Error saving {description} image {path}: {str(e)}")

def get_first_image_roi(image_folder, roi):
    image_files = list(get_all_images(image_folder))
    if not image_files:
        return None
    first_img = fast_imread(image_files[0])
    if first_img is None:
        return None
    x, y, w, h = roi
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, first_img.shape[1] - x)
    h = min(h, first_img.shape[0] - y)
    return first_img[y:y+h, x:x+w].copy()

def select_crop_area_from_first_image(image_folder):
    image_files = list(get_all_images(image_folder))
    if not image_files:
        print("No image files found in", image_folder)
        return None
    first_image_path = image_files[0]
    first_image = fast_imread(first_image_path)
    if first_image is None:
        print("Error loading image:", first_image_path)
        return None
    return custom_select_roi(first_image)

def detect_blobs(image_path, output_folder, roi=None, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
                 dot_threshold=DEFAULT_DOT_THRESH, use_cuda=False, max_dim=DEFAULT_MAX_DIM,
                 blackhat_enabled=True, apply_clahe=True, blackhat_thresh=DEFAULT_BLACKHAT_THRESH,
                 min_area=DEFAULT_MIN_AREA, max_area=DEFAULT_MAX_AREA, min_circularity=DEFAULT_MIN_CIRC):
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            with open(image_path, 'rb') as f:
                jpeg_bytes = f.read()
            img = GLOBAL_TURBOJPEG.decode(jpeg_bytes, pixel_format=TJPF_BGR)
        else:
            img = fast_imread(image_path)
        if img is None:
            logging.error(f"Could not read image: {image_path}")
            return 0
        if roi is not None:
            x, y, w, h = roi
            img = img[y:y+h, x:x+w]
        h_img, w_img = img.shape[:2]
        scale_factor = min(max_dim / w_img, max_dim / h_img, 1.0)
        if scale_factor < 1.0:
            img = cv2.resize(img, (int(w_img * scale_factor), int(h_img * scale_factor)))
        enhanced_img = enhance_brightness_contrast(img, alpha=alpha, beta=beta,
                                                   use_cuda=use_cuda, apply_clahe=apply_clahe)
        gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        if blackhat_enabled:
            processed = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, GLOBAL_KERNEL)
        else:
            processed = gray
        _, binary = cv2.threshold(processed, blackhat_thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        modified_img = enhanced_img.copy()
        dot_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
            if circularity >= min_circularity:
                dot_count += 1
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
                modified_img[mask == 255] = (0, 0, 255)
        if dot_count > dot_threshold:
            base_name = os.path.basename(image_path)
            os.makedirs(output_folder, exist_ok=True)
            name_no_ext, ext = os.path.splitext(base_name)
            modified_path = os.path.join(output_folder, f"{dot_count}_modified_{name_no_ext}{ext}")
            save_image(modified_img, modified_path, "modified")
        return dot_count
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return 0


class InputPanel(tk.Frame):
    instances = []
    
    def __init__(self, master, panel_title, turbojpeg_instance, global_settings, enable_auto_start=False):
        super().__init__(master, bd=2, relief="groove")
        InputPanel.instances.append(self)
        self.panel_title = panel_title
        self.turbojpeg = turbojpeg_instance
        self.global_settings = global_settings  
        self.enable_auto_start = enable_auto_start  

        
        self.input_folder = None
        self.output_folder = None
        self.roi = None
        self.roi_crop = None

        
        self.alpha_var = tk.StringVar(value=str(DEFAULT_ALPHA))
        self.beta_var = tk.StringVar(value=str(DEFAULT_BETA))
        self.dot_threshold_var = tk.StringVar(value=str(DEFAULT_DOT_THRESH))
        self.min_area_var = tk.StringVar(value=str(DEFAULT_MIN_AREA))
        self.max_area_var = tk.StringVar(value=str(DEFAULT_MAX_AREA))
        self.min_circ_var = tk.StringVar(value=str(DEFAULT_MIN_CIRC))
        self.blackhat_thresh_var = tk.StringVar(value=str(DEFAULT_BLACKHAT_THRESH))
        self.clahe_var = tk.BooleanVar(value=True)
        self.blackhat_var = tk.BooleanVar(value=True)

        
        if self.enable_auto_start:
            self.auto_start_var = tk.BooleanVar(value=False)

        
        self.executor = None
        self.futures = {}
        self.completed = 0
        self.total_images = 0
        self.total_circles = 0
        self.start_time = 0
        self.profiler = None
        self.profile_results = ""
        self.terminated = False
        self.image_batches = None
        self.processing_params = {}
        self.on_complete = None  

        
        self.current_original_img = None
        self.current_enh_img = None  
        self.current_highlight_img = None  

        
        self.magnifier_enabled = False
        self.magnifier_window = None
        self.magnifier_label = None

        self.create_widgets()

    def create_widgets(self):
        
        tk.Label(self, text=self.panel_title, font=("Consolas", 12, "bold")).pack(pady=5)
        top_row = tk.Frame(self)
        top_row.pack(pady=5, fill="x")          
        
        frame_folders = tk.Frame(top_row)
        frame_folders.pack(side="left", padx=5)  
        tk.Label(frame_folders, text="Input Folder:").grid(row=0, column=0, sticky="w", padx=5)
        self.input_folder_var = tk.StringVar()
        tk.Entry(frame_folders, textvariable=self.input_folder_var, width=40).grid(row=0, column=1, padx=5)
        tk.Button(frame_folders, text="Browse", command=self.select_input_folder).grid(row=0, column=2, padx=5)
        tk.Label(frame_folders, text="Output Folder:").grid(row=1, column=0, sticky="w", padx=5)
        self.output_folder_var = tk.StringVar()
        tk.Entry(frame_folders, textvariable=self.output_folder_var, width=40).grid(row=1, column=1, padx=5)
        tk.Button(frame_folders, text="Browse", command=self.select_output_folder).grid(row=1, column=2, padx=5)
      
        frame_params = tk.Frame(top_row)
        frame_params.pack(side="left", padx=15)
        tk.Label(frame_params, text="Dot Thresh:").grid(row=0, column=0, padx=0, sticky="e")
        tk.Entry(frame_params, textvariable=self.dot_threshold_var, width=6).grid(row=0, column=1, padx=0)
        tk.Label(frame_params, text="min_area:").grid(row=0, column=2, padx=0, sticky="e")
        tk.Entry(frame_params, textvariable=self.min_area_var, width=6).grid(row=0, column=3, padx=0)
        tk.Label(frame_params, text="max_area:").grid(row=0, column=4, padx=0, sticky="e")
        tk.Entry(frame_params, textvariable=self.max_area_var, width=6).grid(row=0, column=5, padx=0)
        tk.Label(frame_params, text="min_circ:").grid(row=0, column=6, padx=0, sticky="e")
        tk.Entry(frame_params, textvariable=self.min_circ_var, width=6).grid(row=0, column=7, padx=0)

        tk.Label(frame_params, text="Alpha:").grid(row=1, column=0, padx=0, sticky="e")
        tk.Entry(frame_params, textvariable=self.alpha_var, width=6).grid(row=1, column=1, padx=0)
        tk.Label(frame_params, text="Beta:").grid(row=1, column=2, padx=0, sticky="e")
        tk.Entry(frame_params, textvariable=self.beta_var, width=6).grid(row=1, column=3, padx=0)
        tk.Label(frame_params, text="Blackhat Thresh:").grid(row=1, column=4, padx=0, sticky="e")
        tk.Entry(frame_params, textvariable=self.blackhat_thresh_var, width=6).grid(row=1, column=5, padx=0)
        tk.Checkbutton(frame_params, text="Blackhat", variable=self.blackhat_var, command=self.update_preview).grid(row=1, column=6, padx=0)
        tk.Checkbutton(frame_params, text="CLAHE", variable=self.clahe_var, command=self.update_preview).grid(row=1, column=7, padx=0)

        frame_controls = tk.Frame(self)
        frame_controls.pack(pady=5)
        tk.Button(frame_controls, text="Update", command=self.update_preview).grid(row=0, column=0, padx=5)
        tk.Button(frame_controls, text="Reselect ROI", command=self.select_roi).grid(row=0, column=1, padx=5)
        self.run_button = tk.Button(frame_controls, text="Run", command=self.start_processing)
        self.run_button.grid(row=0, column=2, padx=5)
        self.terminate_button = tk.Button(frame_controls, text="Terminate", command=self.terminate_processing, state="disabled")
        self.terminate_button.grid(row=0, column=3, padx=5)
        if self.enable_auto_start:
            tk.Checkbutton(frame_controls, text="Auto Start", variable=self.auto_start_var).grid(row=0, column=4, padx=5)
        self.magnifier_button = tk.Button(frame_controls, text="Magnifier Off", command=self.toggle_magnifier)
        self.magnifier_button.grid(row=0, column=5, padx=5)

        
        profile_bar = tk.Frame(self)
        profile_bar.pack(pady=2)

        self.profiles = _load_profiles()
        
        opt_list = ["[select]"] + list(self.profiles.keys())
        self.selected_profile = tk.StringVar(value=opt_list[0])

        self.profile_menu = tk.OptionMenu(
        profile_bar,
        self.selected_profile,
        *opt_list,                           
        command=lambda p: self.load_profile(p) if p != "[select]" else None
        )
        self.profile_menu.config(width=20)
        self.profile_menu.pack(side="left", padx=2)
        
        tk.Button(profile_bar, text="Save",
                  command=self.save_profile).pack(side="left", padx=2)
        tk.Button(profile_bar, text="Delete",
                  command=self.delete_profile).pack(side="left", padx=2)

        self.counter_label = tk.Label(self, text="Processed: 0/0 images")
        self.counter_label.pack(pady=5)
        self.log_text = scrolledtext.ScrolledText(self, width=100, height=1)
        self.log_text.pack(pady=1)
        self.preview_frame = tk.Frame(self)
        self.preview_frame.pack(pady=5)
        self.original_label = tk.Label(self.preview_frame, text="Original ROI Preview")
        self.original_label.pack()
        self.orig_rgb_label = tk.Label(self.preview_frame, text="RGB: N/A")
        self.orig_rgb_label.pack()
        self.enhanced_label = tk.Label(self.preview_frame, text="Enhanced ROI Preview")
        self.enhanced_label.pack()
        self.enh_rgb_label = tk.Label(self.preview_frame, text="RGB: N/A")
        self.enh_rgb_label.pack()
        self.original_label.bind("<Motion>", self.show_orig_pixel_value)
        self.original_label.bind("<Leave>", self.hide_magnifier)
        self.enhanced_label.bind("<Motion>", self.show_enh_pixel_value)
        self.enhanced_label.bind("<Leave>", self.hide_magnifier)

    def select_input_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder = folder
            self.input_folder_var.set(folder)
            self.log(f"Input folder set: {folder}")
            roi = select_crop_area_from_first_image(folder)
            if roi:
                self.roi = roi
                self.log(f"ROI selected: {roi}")
                cropped = get_first_image_roi(folder, roi)
                if cropped is not None:
                    self.roi_crop = cropped
                    self.show_original_roi_preview()
                else:
                    self.log("Error: Cannot load first image for preview.")
            else:
                self.log("ROI selection cancelled.")

    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder = folder
            self.output_folder_var.set(folder)
            self.log(f"Output folder set: {folder}")

    def select_roi(self):
        if self.input_folder:
            image_files = list(get_all_images(self.input_folder))
            if image_files:
                first_image = fast_imread(image_files[0])
                roi = custom_select_roi(first_image)
                if roi:
                    self.roi = roi
                    self.log(f"ROI updated: {roi}")
                    self.roi_crop = first_image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]].copy()
                    self.show_original_roi_preview()
        else:
            self.log("Select input folder first.")

    def show_original_roi_preview(self):
        if self.roi_crop is None:
            return
        self.current_original_img = self.roi_crop.copy()
        roi_rgb = cv2.cvtColor(self.roi_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        self.original_img_tk = ImageTk.PhotoImage(pil_img)
        self.original_label.config(image=self.original_img_tk)
        self.preview_frame.config(width=self.roi_crop.shape[1], height=self.roi_crop.shape[0])
        self.update_preview()

    def update_preview(self):
        if self.roi_crop is None:
            return
        try:
            alpha = float(self.alpha_var.get())
            beta = float(self.beta_var.get())
            dot_threshold = int(self.dot_threshold_var.get())
            min_area = float(self.min_area_var.get())
            max_area = float(self.max_area_var.get())
            min_circ = float(self.min_circ_var.get())
            blackhat_thresh = int(self.blackhat_thresh_var.get())
        except ValueError:
            self.log("Invalid parameter value(s) for preview.")
            return
        roi_img = self.roi_crop.copy()
        h_img, w_img = roi_img.shape[:2]
        scale_factor = min(DEFAULT_MAX_DIM / w_img, DEFAULT_MAX_DIM / h_img, 1.0)
        if scale_factor < 1.0:
            roi_img = cv2.resize(roi_img, (int(w_img * scale_factor), int(h_img * scale_factor)))
        self.current_enh_img = roi_img.copy()

        # Img enhancement
        enhanced_img = enhance_brightness_contrast(roi_img, alpha=alpha, beta=beta,
                                                  use_cuda=False, apply_clahe=self.clahe_var.get())
        gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        if self.blackhat_var.get():
            processed = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, GLOBAL_KERNEL)
        else:
            processed = gray
        _, binary = cv2.threshold(processed, blackhat_thresh, 255, cv2.THRESH_BINARY)
        highlight_img = enhanced_img.copy()
        dot_count = 0
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
            if circularity >= min_circ:
                dot_count += 1
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
                highlight_img[mask == 255] = (0, 0, 255)
        cv2.putText(highlight_img, f"Dots: {dot_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        self.current_highlight_img = highlight_img.copy()
        preview_rgb = cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(preview_rgb)
        self.enhanced_img_tk = ImageTk.PhotoImage(pil_img)
        self.enhanced_label.config(image=self.enhanced_img_tk)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    # RGB value & Magnifier 

    def show_orig_pixel_value(self, event):
        if self.current_original_img is not None:
            x, y = event.x, event.y
            h, w, _ = self.current_original_img.shape
            if 0 <= x < w and 0 <= y < h:
                pixel = self.current_original_img[y, x]
                r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
                self.orig_rgb_label.config(text=f"RGB: ({r}, {g}, {b})")
                if self.magnifier_enabled:
                    self.update_magnifier(event, self.current_original_img)
            else:
                self.orig_rgb_label.config(text="RGB: N/A")
        else:
            self.orig_rgb_label.config(text="RGB: N/A")

    def show_enh_pixel_value(self, event):
        if self.current_highlight_img is not None:
            x, y = event.x, event.y
            h, w, _ = self.current_highlight_img.shape
            if 0 <= x < w and 0 <= y < h:
                pixel = self.current_highlight_img[y, x]
                r, g, b = int(pixel[2]), int(pixel[1]), int(pixel[0])
                self.enh_rgb_label.config(text=f"RGB: ({r}, {g}, {b})")
                if self.magnifier_enabled:
                    self.update_magnifier(event, self.current_highlight_img)
            else:
                self.enh_rgb_label.config(text="RGB: N/A")
        else:
            self.enh_rgb_label.config(text="RGB: N/A")

    def update_magnifier(self, event, source_img):
        region_size = 20  
        zoom_factor = 5
        x, y = event.x, event.y
        h, w, _ = source_img.shape
        x0 = max(x - region_size, 0)
        y0 = max(y - region_size, 0)
        x1 = min(x + region_size, w)
        y1 = min(y + region_size, h)
        region = source_img[y0:y1, x0:x1]
        if region.size == 0:
            return
        magnified = cv2.resize(region, (region.shape[1]*zoom_factor, region.shape[0]*zoom_factor), interpolation=cv2.INTER_NEAREST)
        magnified_rgb = cv2.cvtColor(magnified, cv2.COLOR_BGR2RGB)
        pil_mag = Image.fromarray(magnified_rgb)
        mag_photo = ImageTk.PhotoImage(pil_mag)
        if self.magnifier_window is None:
            self.magnifier_window = tk.Toplevel(self)
            self.magnifier_window.overrideredirect(True)
            self.magnifier_label = tk.Label(self.magnifier_window)
            self.magnifier_label.pack()
        self.magnifier_label.config(image=mag_photo)
        self.magnifier_label.image = mag_photo  
        offset = 20
        self.magnifier_window.geometry(f"+{event.x_root + offset}+{event.y_root + offset}")
        self.magnifier_window.deiconify()

    def hide_magnifier(self, event):
        if self.magnifier_window is not None:
            self.magnifier_window.withdraw()

    def toggle_magnifier(self):
        self.magnifier_enabled = not self.magnifier_enabled
        state_text = "Magnifier On" if self.magnifier_enabled else "Magnifier Off"
        self.magnifier_button.config(text=state_text)
        if not self.magnifier_enabled and self.magnifier_window is not None:
            self.magnifier_window.withdraw()
        self.log(f"Magnifier toggled: {state_text}")

    def start_processing(self):
        if not self.input_folder:
            self.log("Please select an input folder.")
            return
        if not self.output_folder:
            self.log("Please select an output folder.")
            return
        try:
            alpha = float(self.alpha_var.get())
            beta = float(self.beta_var.get())
            dot_threshold = int(self.dot_threshold_var.get())
            blackhat_thresh = int(self.blackhat_thresh_var.get())
            min_area = float(self.min_area_var.get())
            max_area = float(self.max_area_var.get())
            min_circ = float(self.min_circ_var.get())
        except ValueError:
            self.log("Invalid parameter value(s) for processing.")
            return

        self.completed = 0
        self.total_circles = 0
        self.start_time = time.time()
        self.terminated = False
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.run_button.config(state="disabled")
        self.terminate_button.config(state="normal")
        self.total_images = sum(1 for _ in get_all_images(self.input_folder))
        self.log(f"Processing {self.total_images} images...")
        batch_size = self.global_settings.get("batch_size", DEFAULT_BATCH_SIZE)
        image_gen = get_all_images(self.input_folder)
        self.image_batches = batch_iterator(image_gen, batch_size)
        os.makedirs(self.output_folder, exist_ok=True)
        self.futures.clear()
        self.processing_params = {
            "alpha": alpha,
            "beta": beta,
            "dot_threshold": dot_threshold,
            "max_dim": DEFAULT_MAX_DIM,
            "blackhat_thresh": blackhat_thresh,
            "min_area": min_area,
            "max_area": max_area,
            "min_circ": min_circ
        }
        num_workers = self.global_settings.get("num_workers", os.cpu_count())
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        self.submit_next_batch()
        self.check_futures()

    def submit_next_batch(self):
        try:
            batch = next(self.image_batches)
        except StopIteration:
            return
        for file_path in batch:
            future = self.executor.submit(
                detect_blobs,
                file_path,
                self.output_folder,
                self.roi,
                self.processing_params["alpha"],
                self.processing_params["beta"],
                self.processing_params["dot_threshold"],
                self.global_settings.get("cuda_enabled", False),
                self.processing_params["max_dim"],
                self.blackhat_var.get(),
                self.clahe_var.get(),
                self.processing_params["blackhat_thresh"],
                self.processing_params["min_area"],
                self.processing_params["max_area"],
                self.processing_params["min_circ"]
            )
            self.futures[future] = file_path

    def check_futures(self):
        if self.terminated:
            self.log("Processing terminated by user.")
            return
        done_futures = [f for f in list(self.futures.keys()) if f.done()]
        for f in done_futures:
            file_path = self.futures.pop(f)
            try:
                num_circles = f.result()
                self.completed += 1
                self.total_circles += num_circles
            except Exception as e:
                self.log(f"Error processing {file_path}: {e}")
                self.completed += 1
        self.update_progress()
        if not self.futures and not self.terminated:
            self.submit_next_batch()
        if self.futures:
            self.after(50, self.check_futures)
        else:
            if self.completed == self.total_images:
                self.profiler.disable()
                s = io.StringIO()
                ps = pstats.Stats(self.profiler, stream=s).sort_stats(pstats.SortKey.TIME)
                ps.print_stats()
                self.profile_results = s.getvalue()
                self.log(f"Analysis complete. Processed {self.total_images} images, found {self.total_circles} circles.")
                self.run_button.config(state="normal")
                self.terminate_button.config(state="disabled")
                if self.on_complete:
                    self.on_complete()
            else:
                self.after(50, self.check_futures)

    def update_progress(self):
        elapsed_time = time.time() - self.start_time
        speed = self.completed / elapsed_time if elapsed_time > 0 else 0
        self.counter_label.config(text=f"Processed: {self.completed}/{self.total_images} images, {speed:.2f} imgs/s")

    def terminate_processing(self):
        self.terminated = True
        self.log("Processing terminated by user.")
        self.run_button.config(state="normal")
        self.terminate_button.config(state="disabled")
    
    def _current_params(self) -> dict:
        return {
            "alpha":      float(self.alpha_var.get()),
            "beta":       float(self.beta_var.get()),
            "dot":        int(self.dot_threshold_var.get()),
            "minA":       float(self.min_area_var.get()),
            "maxA":       float(self.max_area_var.get()),
            "minC":       float(self.min_circ_var.get()),
            "maxDim":     DEFAULT_MAX_DIM,
            "bhThresh":   int(self.blackhat_thresh_var.get()),
            "clahe":      bool(self.clahe_var.get()),
            "blackhat":   bool(self.blackhat_var.get()),
        }
    
    def _apply_params(self, p: dict):
        self.alpha_var.set(p["alpha"])
        self.beta_var.set(p["beta"])
        self.dot_threshold_var.set(p["dot"])
        self.min_area_var.set(p["minA"])
        self.max_area_var.set(p["maxA"])
        self.min_circ_var.set(p["minC"])
        self.blackhat_thresh_var.set(p["bhThresh"])
        self.clahe_var.set(p["clahe"])
        self.blackhat_var.set(p["blackhat"])
        self.update_preview()

    
    def save_profile(self):
        name = simpledialog.askstring("Save Profile", "Profile name:",
                                      parent=self)
        if not name:
            return
        self.profiles[name] = self._current_params()
        _save_profiles(self.profiles)
        for p in InputPanel.instances:
            p.refresh_profile_menu(keep_choice=False)
        self.log(f"Profile '{name}' saved.")

    def load_profile(self, name):
        if name not in self.profiles:
            return
        self._apply_params(self.profiles[name])
        self.log(f"Profile '{name}' loaded.")
    
    def refresh_profile_menu(self, keep_choice=True):
        """Reload the JSON file and rebuild the dropdown."""
        current = self.selected_profile.get()
        self.profiles = _load_profiles()
        opt_list = ["[select]"] + list(self.profiles.keys())

        menu = self.profile_menu["menu"]
        menu.delete(0, "end")
        for name in opt_list:
            menu.add_command(
                label=name,
                command=lambda v=name: (
                    self.selected_profile.set(v),
                    self.load_profile(v) if v != "[select]" else None
                )
            )
        if keep_choice and current in opt_list:
            self.selected_profile.set(current)
        else:
            self.selected_profile.set(opt_list[0])


    def delete_profile(self):
        name = self.selected_profile.get()
        if name in self.profiles and messagebox.askyesno(
                "Delete Profile", f"Delete profile '{name}'?"):
            del self.profiles[name]
            _save_profiles(self.profiles)
            for p in InputPanel.instances:
                p.refresh_profile_menu(keep_choice=False)
            self.log(f"Profile '{name}' deleted.")

class DualInputApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NG Mark Detection v1.7")
        self.global_settings = {
            "num_workers": os.cpu_count(),
            "batch_size": DEFAULT_BATCH_SIZE,
            "cuda_enabled": False
        }
        self.num_gpus = cv2.cuda.getCudaEnabledDeviceCount()
        if self.num_gpus > 0:
            try:
                dev_info = cv2.cuda.DeviceInfo(0)
                self.gpu_name = dev_info.name()
            except Exception:
                self.gpu_name = "Unknown CUDA Device"
        else:
            self.gpu_name = None

        top_frame = tk.Frame(self)
        top_frame.pack(side="top", fill="x", pady=5)
        self.add_input_button = tk.Button(top_frame, text="+Add Input", command=self.add_second_input)
        self.add_input_button.pack(side="left", padx=5)
        tk.Button(top_frame, text="Settings", command=self.open_settings).pack(side="left", padx=5)

        self.input_panels_frame = tk.Frame(self)
        self.input_panels_frame.pack(side="top", fill="both", expand=True)
        self.input_panel1 = InputPanel(self.input_panels_frame, "Input 1", GLOBAL_TURBOJPEG, self.global_settings)
        self.input_panel1.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.input_panel1.on_complete = self.input1_complete_callback
        self.input_panel2 = None

        self.input_panel1.run_button.config(command=self.start_input1_processing)

    def start_input1_processing(self):
        
        if self.input_panel2:
            
            if not (self.input_panel2.input_folder and self.input_panel2.output_folder and self.input_panel2.roi):
                messagebox.showerror("Input 2 Incomplete", "Input 2 is not fully configured. Please complete its setup before starting Input 1 processing.")
                return
            
            if not self.input_panel2.auto_start_var.get():
                result = messagebox.askyesno("Enable Auto Start", "Do you want to enable auto start for Input 2?")
                if result:
                    self.input_panel2.auto_start_var.set(True)
        self.input_panel1.start_processing()

    def add_second_input(self):
        if not self.input_panel2:
            self.input_panel2 = InputPanel(self.input_panels_frame, "Input 2", GLOBAL_TURBOJPEG, self.global_settings, enable_auto_start=True)
            self.input_panel2.pack(side="left", fill="both", expand=True, padx=5, pady=5)
            self.add_input_button.config(state="disabled")

    def input1_complete_callback(self):
        if self.input_panel2 and self.input_panel2.auto_start_var.get():
            self.input_panel2.log("Auto-start triggered: Starting processing for Input 2.")
            self.input_panel2.start_processing()

    def open_settings(self):
        settings_window = tk.Toplevel(self)
        settings_window.title("Settings")
        cpu_info = platform.processor() or "Unknown CPU"
        tk.Label(settings_window, text=f"CPU: {cpu_info} ({os.cpu_count()} cores)").grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        if self.num_gpus > 0:
            tk.Label(settings_window, text=f"GPU: {self.gpu_name}").grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        else:
            tk.Label(settings_window, text="GPU: No CUDA-enabled device detected").grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        tk.Label(settings_window, text="Number of Workers:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.num_workers_var = tk.StringVar(value=str(self.global_settings["num_workers"]))
        tk.Entry(settings_window, textvariable=self.num_workers_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        tk.Label(settings_window, text="Batch Size:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.batch_size_var = tk.StringVar(value=str(self.global_settings["batch_size"]))
        tk.Entry(settings_window, textvariable=self.batch_size_var, width=10).grid(row=3, column=1, padx=5, pady=5)
        self.cuda_var = tk.BooleanVar(value=self.global_settings["cuda_enabled"])
        cuda_check = tk.Checkbutton(settings_window, text="Enable CUDA", variable=self.cuda_var)
        cuda_check.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        def show_about():
            about_text = (
                "NG Mark Detection\n\n"
                "Created by Dimitri Callow.\n"
                "Version 1.8\n\n"
                           )
            messagebox.showinfo("About", about_text)
        tk.Button(settings_window, text="About", command=show_about, width=10).grid(row=5, column=0, pady=5, sticky="e", padx=5)
        def save_settings():
            try:
                workers = int(self.num_workers_var.get())
                if workers < 1:
                    raise ValueError
                batch = int(self.batch_size_var.get())
                if batch < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Input", "Number of workers and batch size must be positive integers.")
                return
            self.global_settings["num_workers"] = workers
            self.global_settings["batch_size"] = batch
            if self.num_gpus > 0:
                self.global_settings["cuda_enabled"] = self.cuda_var.get()
            else:
                self.global_settings["cuda_enabled"] = False
            print(f"Settings updated: Workers={workers}, Batch Size={batch}, CUDA Enabled={self.global_settings['cuda_enabled']}")
            settings_window.destroy()
        tk.Button(settings_window, text="Save Settings", command=save_settings, width=10).grid(row=5, column=1, pady=5, sticky="w")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = DualInputApp()
    app.mainloop()
