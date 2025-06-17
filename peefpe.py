import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as ski_watershed
from skimage.morphology import remove_small_objects
import os
import glob

class FishDishAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("FISH/DISH Image Analyzer")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        self.analysis_type = None
        self.original_path = ""
        self.groundtruth_path = ""
        self.output_path = ""
        
        self.proceed_with_processing = False
        
        self.create_main_interface()
    
    def create_main_interface(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
        title_label = tk.Label(self.root, text="Image Analysis Tool", 
                              font=("Arial", 20, "bold"), pady=20)
        title_label.pack()
        
        subtitle_label = tk.Label(self.root, text="Pilih jenis analisis yang ingin dilakukan:", 
                                 font=("Arial", 12), pady=10)
        subtitle_label.pack()
        
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=30)
        
        fish_btn = tk.Button(button_frame, text="FISH ANALISIS", 
                            font=("Arial", 14, "bold"),
                            bg="#4CAF50", fg="white",
                            width=15, height=2,
                            command=lambda: self.select_analysis_type("FISH"))
        fish_btn.pack(side=tk.LEFT, padx=20)
        
        dish_btn = tk.Button(button_frame, text="DISH ANALISIS", 
                            font=("Arial", 14, "bold"),
                            bg="#2196F3", fg="white",
                            width=15, height=2,
                            command=lambda: self.select_analysis_type("DISH"))
        dish_btn.pack(side=tk.LEFT, padx=20)
    
    def select_analysis_type(self, analysis_type):
        self.analysis_type = analysis_type
        self.create_file_selection_interface()
    
    def create_file_selection_interface(self):
        for widget in self.root.winfo_children():
            widget.destroy()
            
        title_label = tk.Label(self.root, text=f"{self.analysis_type} Analysis", 
                              font=("Arial", 18, "bold"), pady=20)
        title_label.pack()
        
        main_frame = tk.Frame(self.root, padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        file_frame = tk.LabelFrame(main_frame, text="Pilih Path untuk File", 
                                  font=("Arial", 12, "bold"), padx=10, pady=10)
        file_frame.pack(fill=tk.X, pady=10)
        
        # === ORIGINAL === #
        tk.Label(file_frame, text="1. Path Original Images:", 
                font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        
        self.original_entry = tk.Entry(file_frame, width=50)
        self.original_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        tk.Button(file_frame, text="Browse", 
                 command=lambda: self.browse_and_set_path(self.original_entry)).grid(row=0, column=2, pady=5)
        
        # === GROUND TRUTH === #
        tk.Label(file_frame, text="2. Path Ground Truth:", 
                font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        
        self.groundtruth_entry = tk.Entry(file_frame, width=50)
        self.groundtruth_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        tk.Button(file_frame, text="Browse", 
                 command=lambda: self.browse_and_set_path(self.groundtruth_entry)).grid(row=1, column=2, pady=5)
        
        # === OUTPUT === #
        tk.Label(file_frame, text="3. Path Output Results:", 
                font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", pady=5)
        
        self.output_entry = tk.Entry(file_frame, width=50)
        self.output_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        tk.Button(file_frame, text="Browse", 
                 command=lambda: self.browse_and_set_path(self.output_entry)).grid(row=2, column=2, pady=5)
        
        file_frame.columnconfigure(1, weight=1)
        
        action_frame = tk.Frame(main_frame, pady=20)
        action_frame.pack()
        
        # === PREVIEW === #
        preview_btn = tk.Button(action_frame, text="Preview Images", 
                               font=("Arial", 12, "bold"),
                               bg="#FF9800", fg="white",
                               width=15, height=2,
                               command=self.preview_images)
        preview_btn.pack(side=tk.LEFT, padx=10)
        
        # === START === #
        process_btn = tk.Button(action_frame, text="Start Processing", 
                               font=("Arial", 12, "bold"),
                               bg="#4CAF50", fg="white",
                               width=15, height=2,
                               command=self.start_processing)
        process_btn.pack(side=tk.LEFT, padx=10)
       
        back_btn = tk.Button(action_frame, text="Back", 
                            font=("Arial", 12),
                            bg="#9E9E9E", fg="white",
                            width=10, height=2,
                            command=self.create_main_interface)
        back_btn.pack(side=tk.LEFT, padx=10)
        
        # === STATUS === #
        status_frame = tk.LabelFrame(main_frame, text="Status", 
                                    font=("Arial", 10, "bold"))
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = tk.Label(status_frame, text="Ready to select paths...", 
                                    font=("Arial", 10), fg="blue")
        self.status_label.pack(pady=10)
    
    def browse_and_set_path(self, entry_widget):
        folder_path = self.browse_directory()
        if folder_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, folder_path)
            self.update_status(f"Path selected: {os.path.basename(folder_path)}")
    
    def browse_directory(self):
        folder_path = filedialog.askdirectory(title="Select Directory")
        return folder_path
    
    def get_image_paths(self, directory):
        if not directory or not os.path.exists(directory):
            return []
        
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
        image_paths = []
        
        all_files = os.listdir(directory)
        
        for file in all_files:
            file_ext = file.lower().split('.')[-1]
            if file_ext in extensions:
                full_path = os.path.join(directory, file)
                if os.path.isfile(full_path):
                    image_paths.append(full_path)
        
        return sorted(image_paths)
    
    def preview_images(self):
        original_path = self.original_entry.get().strip()
        
        if not original_path:
            messagebox.showerror("Error", "Pilih Gambar Dulu Yak!!!")
            return
        
        image_paths = self.get_image_paths(original_path)
        
        if not image_paths:
            messagebox.showwarning("Warning", "Kacaw, Kamu Nggak Milih Gambar Ya?")
            return
        
        self.update_status(f"Found {len(image_paths)} images. Displaying preview...")
        self.display_images(image_paths)
    
    def display_images(self, image_paths, max_display=5):
        if not image_paths:
            print("Males Wes, Gaada Yang Kudu Ditampilkan")
            return False
        
        self.proceed_with_processing = False
        
        def yes_button_click(event):
            self.proceed_with_processing = True
            plt.close(fig)
            
        def no_button_click(event):
            self.proceed_with_processing = False
            plt.close(fig)
        
        display_count = min(len(image_paths), max_display)
        sample_images = image_paths[:display_count]
        
        fig, axes = plt.subplots(1, display_count, figsize=(15, 5))
        if display_count == 1:
            axes = [axes]
        
        for i, img_path in enumerate(sample_images):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img_rgb)
                    axes[i].set_title(f"Image {i+1}")
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, 'Cannot load image', 
                               horizontalalignment='center', verticalalignment='center')
                    axes[i].set_title(f"Image {i+1} (Error)")
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                           horizontalalignment='center', verticalalignment='center')
                axes[i].set_title(f"Image {i+1} (Error)")
        
        plt.tight_layout()
        fig.suptitle(f"Displaying {display_count} of {len(image_paths)} YEYYY KETEMU\nProses Nggak Nih???", 
                    fontsize=12, y=1.02)
        
        btn_yes_ax = plt.axes([0.7, 0.01, 0.1, 0.05])
        btn_no_ax = plt.axes([0.81, 0.01, 0.1, 0.05])
        
        btn_yes = Button(btn_yes_ax, 'Yes', color='lightgreen')
        btn_no = Button(btn_no_ax, 'No', color='salmon')
        
        btn_yes.on_clicked(yes_button_click)
        btn_no.on_clicked(no_button_click)
        
        plt.show()
        
        if self.proceed_with_processing:
            self.update_status("Oke, lanjut aja")
            return True
        else:
            self.update_status("Lah Nggak Mau Diproses")
            return False
    
    def detect_signals(self, channel, min_distance=10, threshold_abs=None):
        blurred = cv2.GaussianBlur(channel, (5, 5), 0)
        
        if threshold_abs is None:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(blurred, threshold_abs, 255, cv2.THRESH_BINARY)
        
        binary_cleaned = remove_small_objects(binary.astype(bool), min_size=20)
        binary_cleaned = binary_cleaned.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = 10
        max_area = 1000
        valid_contours = []
        signal_centers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.2:
                        valid_contours.append(contour)
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            signal_centers.append((cx, cy))
        
        return binary_cleaned, valid_contours, signal_centers, len(valid_contours)
    
    
    def classify_her2_status(self, her2_count, cen17_count):
        if cen17_count == 0:
            ratio = float('inf') if her2_count > 0 else 0
        else:
            ratio = her2_count / cen17_count
        
        if ratio > 2.0:
            status = "HER2-Positive"
            action = "Eligible for anti-HER2 therapy"
            color = "green"
        elif 1.8 <= ratio <= 2.0:
            status = "Equivocal"
            action = "Need further evaluation"
            color = "orange"
        else:
            status = "HER2-Negative"
            action = "No need for anti-HER2 therapy"
            color = "red"
        
        return ratio, status, action, color
    
    def segment_cells_fish(self, blue_channel):
        _, binary = cv2.threshold(blue_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = binary.astype(bool)

        distance = ndimage.distance_transform_edt(binary)

        coordinates = peak_local_max(distance, min_distance=20, labels=binary)

        local_max = np.zeros_like(binary, dtype=bool)
        if len(coordinates) > 0:
            local_max[tuple(coordinates.T)] = True

        markers = ndimage.label(local_max)[0]

        labels = ski_watershed(-distance, markers, mask=binary)
        return labels, distance, binary
    
    def segment_cells_dish(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
       
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
       
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
       
        enhanced = cv2.add(gray, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)
       
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(enhanced)
       
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if np.mean(binary) > 127:  # Jika background lebih terang dari foreground
            binary = cv2.bitwise_not(binary)
            
        binary_cleaned = remove_small_objects(binary.astype(bool), min_size=50)
        cleaned = binary_cleaned.astype(np.uint8) * 255    
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned_bool = cleaned.astype(bool)

        distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

        coordinates = peak_local_max(distance, min_distance=5, labels=cleaned_bool)
        local_max = np.zeros_like(cleaned_bool, dtype=bool)
        if len(coordinates) > 0:
            local_max[tuple(coordinates.T)] = True
        markers = ndimage.label(local_max)[0]
        labels_ws = ski_watershed(-distance, markers, mask=cleaned_bool)

        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay = gray_rgb.copy()
        overlay[labels_ws > 0] = [0, 255, 0]  # Green for segmented

        return labels_ws, distance, cleaned_bool, gray, 255-gray, binary, cleaned, overlay
    
    def calculate_metrics(self, gt_mask, seg_mask):
        intersection = np.logical_and(gt_mask, seg_mask).sum()
        union = np.logical_or(gt_mask, seg_mask).sum()
        iou = intersection / union if union != 0 else 0
        dice = (2 * intersection) / (gt_mask.sum() + seg_mask.sum()) if (gt_mask.sum() + seg_mask.sum()) != 0 else 0
        return iou, dice
    
    def process_fish_image(self, img_path, gt_path, output_dir, filename):
        try:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            gt = cv2.imread(gt_path)
            gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            
            blue, green, red = cv2.split(img)
            
            labels_ws, dist_map, binarized = self.segment_cells_fish(blue)
            
            her2_binary, her2_contours, her2_centers, her2_count = self.detect_signals(red)
            
            cen17_binary, cen17_contours, cen17_centers, cen17_count = self.detect_signals(green)
            
            ratio, status, action, status_color = self.classify_her2_status(her2_count, cen17_count)
            
            blue_rgb = cv2.cvtColor(blue, cv2.COLOR_GRAY2RGB)
            overlay = blue_rgb.copy()
            overlay[labels_ws > 0] = [0, 255, 0]  # Green overlay for cell nuclei
            
            signal_overlay = img_rgb.copy()
            
            for center in her2_centers:
                cv2.circle(signal_overlay, center, 8, (255, 0, 0), 2)
                cv2.circle(signal_overlay, center, 2, (255, 255, 0), -1)
            
            for center in cen17_centers:
                cv2.circle(signal_overlay, center, 6, (0, 255, 0), 2)
                cv2.circle(signal_overlay, center, 2, (255, 255, 0), -1)
            
            gt_resized = cv2.resize(gt_rgb, (overlay.shape[1], overlay.shape[0]))
            overlay_resized = cv2.resize(overlay, (gt_rgb.shape[1], gt_rgb.shape[0]))
            
            mask_segmented = np.all(overlay_resized == [0, 255, 0], axis=-1)
            gt_overlay = gt_rgb.copy()
            gt_overlay[mask_segmented] = [0, 255, 0]
            
            gt_gray = cv2.cvtColor(gt_resized, cv2.COLOR_RGB2GRAY)
            _, gt_mask = cv2.threshold(gt_gray, 10, 255, cv2.THRESH_BINARY)
            gt_mask = gt_mask.astype(bool)
            
            seg_mask = np.all(overlay == [0, 255, 0], axis=-1)
            
            iou, dice = self.calculate_metrics(gt_mask, seg_mask)
            
            plt.figure(figsize=(24, 16))
            
            plt.subplot(3, 4, 1)
            plt.imshow(img_rgb)
            plt.title("1. Original Image", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 2)
            plt.imshow(gt_rgb)
            plt.title("2. Ground Truth", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 3)
            plt.imshow(blue, cmap='gray')
            plt.title("3. Blue DAPI Channel", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 4)
            plt.imshow(binarized, cmap='gray')
            plt.title("4. Threshold Blue DAPI", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 5)
            plt.imshow(red, cmap='Reds')
            plt.title(f"5. HER2 Channel (Red)\nDetected: {her2_count} signals", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 6)
            plt.imshow(green, cmap='Greens')
            plt.title(f"6. CEN17 Channel (Green)\nDetected: {cen17_count} signals", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 7)
            plt.imshow(signal_overlay)
            plt.title("7. Signal Detection Overlay\nRed: HER2, Green: CEN17", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 8)
            plt.imshow(labels_ws, cmap='nipy_spectral')
            plt.title("8. Cell Segmentation", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 9)
            plt.imshow(gt_overlay)
            plt.title("9. Segmentation vs GT", fontsize=12)
            plt.axis('off')
            
            # HER2 Classification Result
            plt.subplot(3, 4, 10)
            plt.text(0.1, 0.9, "HER2/CEN17 Analysis", fontsize=16, weight='bold')
            plt.text(0.1, 0.8, f"HER2 Count: {her2_count}", fontsize=14)
            plt.text(0.1, 0.7, f"CEN17 Count: {cen17_count}", fontsize=14)
            plt.text(0.1, 0.6, f"Ratio: {ratio:.2f}", fontsize=14, weight='bold')
            plt.text(0.1, 0.5, f"Status: {status}", fontsize=14, weight='bold', color=status_color)
            plt.text(0.1, 0.4, f"Action: {action}", fontsize=12, wrap=True)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            
            plt.subplot(3, 4, 11)
            plt.text(0.1, 0.8, "Segmentation Metrics", fontsize=16, weight='bold')
            plt.text(0.1, 0.6, f"IoU: {iou:.4f}", fontsize=14)
            plt.text(0.1, 0.4, f"Dice: {dice:.4f}", fontsize=14)
            plt.text(0.1, 0.2, f"File: {filename}", fontsize=10)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            
            plt.subplot(3, 4, 12)
            ratios = ['>2.0', '1.8-2.0', '<1.8']
            statuses = ['HER2-Positive', 'Equivocal', 'HER2-Negative']
            colors = ['green', 'orange', 'red']
            
            highlight_colors = []
            for i, s in enumerate(statuses):
                if s == status:
                    highlight_colors.append(colors[i])
                else:
                    highlight_colors.append('lightgray')
            
            bars = plt.barh(ratios, [1, 1, 1], color=highlight_colors)
            plt.title("HER2 Classification", fontsize=12, weight='bold')
            plt.xlabel("Classification Range")
            
            for i, (ratio_range, stat) in enumerate(zip(ratios, statuses)):
                plt.text(0.5, i, stat, ha='center', va='center', fontsize=10, weight='bold')
            
            plt.xlim(0, 1)
            
            plt.tight_layout()
            
            output_filename = f"{os.path.splitext(filename)[0]}_fish_her2_analysis.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            return iou, dice, her2_count, cen17_count, ratio, status
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return 0, 0, 0, 0, 0, "Error"
    
    def process_dish_image(self, img_path, gt_path, output_dir, filename):
        try:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            gt = cv2.imread(gt_path)
            gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
            brown_lower = np.array([8, 50, 10])   # Lower bound for brown in HSV
            brown_upper = np.array([30, 255, 160]) # Upper bound for brown in HSV
            her2_mask = cv2.inRange(img_hsv, brown_lower, brown_upper)
            
            
            blue_lower = np.array([110, 50, 50])   # Lower bound for blue in HSV
            blue_upper = np.array([130, 255, 255]) # Upper bound for blue in HSV
            blue_mask = cv2.inRange(img_hsv, blue_lower, blue_upper)
            
            pink_lower = np.array([145, 100, 100])   # Lower bound for pink in HSV
            pink_upper = np.array([175, 255, 255]) # Upper bound for pink in HSV
            pink_mask = cv2.inRange(img_hsv, pink_lower, pink_upper)
            
            cen17_mask = cv2.bitwise_or(blue_mask, pink_mask)
            
            blue, green, red = cv2.split(img)
            
            # Segment cells using DISH method
            labels_ws, distance, cleaned_bool, gray, negative, binary, cleaned, overlay = self.segment_cells_dish(img)
            
            cell_count = np.max(labels_ws)
            
            # Detect HER2 signals using brown mask
            her2_binary, her2_contours, her2_centers, her2_count = self.detect_signals(her2_mask)
            
            # Detect CEN17 signals using blue/pink mask
            cen17_binary, cen17_contours, cen17_centers, cen17_count = self.detect_signals(cen17_mask)
            
            # Classify HER2 status
            ratio, status, action, status_color = self.classify_her2_status(her2_count, cen17_count)
            
            # Create overlay for cell nuclei
            blue_rgb = cv2.cvtColor(blue, cv2.COLOR_GRAY2RGB)
            overlay = blue_rgb.copy()
            overlay[labels_ws > 0] = [0, 255, 0]  # Green overlay for cell nuclei
            
            # Create signal overlay
            signal_overlay = img_rgb.copy()
            
            # Mark HER2 signals (brown) with red circles
            for center in her2_centers:
                cv2.circle(signal_overlay, center, 8, (255, 0, 0), 2)    # Red circle for HER2
                cv2.circle(signal_overlay, center, 2, (255, 255, 0), -1) # Yellow center
            
            # Mark CEN17 signals (blue/pink) with blue circles
            for center in cen17_centers:
                cv2.circle(signal_overlay, center, 6, (0, 0, 255), 2)    # Blue circle for CEN17
                cv2.circle(signal_overlay, center, 2, (255, 255, 0), -1) # Yellow center
            
            # Ground truth comparison
            gt_resized = cv2.resize(gt_rgb, (overlay.shape[1], overlay.shape[0]))
            overlay_resized = cv2.resize(overlay, (gt_rgb.shape[1], gt_rgb.shape[0]))
            
            mask_segmented = np.all(overlay_resized == [0, 255, 0], axis=-1)
            gt_overlay = gt_rgb.copy()
            gt_overlay[mask_segmented] = [0, 255, 0]
            
            gt_gray = cv2.cvtColor(gt_resized, cv2.COLOR_RGB2GRAY)
            _, gt_mask = cv2.threshold(gt_gray, 10, 255, cv2.THRESH_BINARY)
            gt_mask = gt_mask.astype(bool)
            
            seg_mask = np.all(overlay == [0, 255, 0], axis=-1)
            
            # Calculate metrics
            iou, dice = self.calculate_metrics(gt_mask, seg_mask)
            
            # Create visualization
            plt.figure(figsize=(24, 18))
            
            plt.subplot(3, 4, 1)
            plt.imshow(img_rgb)
            plt.title("1. Original DISH Image", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 2)
            plt.imshow(gt_rgb, cmap='gray')
            plt.title("2. Groundtruth", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 3)
            plt.imshow(gray, cmap='gray')
            plt.title("3. Grayscale", fontsize=12)
            plt.axis('off')
        
            plt.subplot(3, 4, 4)
            plt.imshow(negative, cmap='gray')
            plt.title("4. Negative Image", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 5)
            plt.imshow(cleaned, cmap='gray')
            plt.title("5. Otsu Thresholding + Cleaning", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 6)
            plt.imshow(her2_mask, cmap='Oranges')
            plt.title(f"6. HER2 Channel (Brown)\nDetected: {her2_count} signals", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 7)
            plt.imshow(cen17_mask, cmap='Blues')
            plt.title(f"7. CEN17 Channel (Blue/Pink)\nDetected: {cen17_count} signals", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 8)
            plt.imshow(signal_overlay)
            plt.title("8. Signal Detection Overlay\nRed: HER2 (Brown), Blue: CEN17 (Blue/Pink)", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 9)
            plt.imshow(labels_ws, cmap='nipy_spectral')
            plt.title("9. Cell Detection", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 10)
            plt.imshow(gt_overlay)
            plt.title("10. Segmentation vs GT", fontsize=12)
            plt.axis('off')
            
            plt.subplot(3, 4, 11)
            plt.text(0.1, 0.9, "HER2/CEN17 Analysis", fontsize=16, weight='bold')
            plt.text(0.1, 0.8, f"HER2 Count: {her2_count}", fontsize=14)
            plt.text(0.1, 0.7, f"CEN17 Count: {cen17_count}", fontsize=14)
            plt.text(0.1, 0.6, f"Ratio: {ratio:.2f}", fontsize=14, weight='bold')
            plt.text(0.1, 0.5, f"Status: {status}", fontsize=14, weight='bold', color=status_color)
            plt.text(0.1, 0.4, f"Action: {action}", fontsize=12, wrap=True)
            plt.text(0.1, 0.3, "Segmentation Metrics", fontsize=16, weight='bold')
            plt.text(0.1, 0.2, f"IoU: {iou:.4f}", fontsize=14)
            plt.text(0.1, 0.1, f"Dice: {dice:.4f}", fontsize=14)
            plt.text(0.1, 0.0, f"File: {filename}", fontsize=10)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            
            plt.subplot(3, 4, 12)
            ratios = ['>2.0', '1.8-2.0', '<1.8']
            statuses = ['HER2-Positive', 'Equivocal', 'HER2-Negative']
            colors = ['green', 'orange', 'red']
            
            highlight_colors = []
            for i, s in enumerate(statuses):
                if s == status:
                    highlight_colors.append(colors[i])
                else:
                    highlight_colors.append('lightgray')
            
            bars = plt.barh(ratios, [1, 1, 1], color=highlight_colors)
            plt.title("HER2 Classification", fontsize=12, weight='bold')
            plt.xlabel("Classification Range")
            
            for i, (ratio_range, stat) in enumerate(zip(ratios, statuses)):
                plt.text(0.5, i, stat, ha='center', va='center', fontsize=10, weight='bold')
            
            plt.xlim(0, 1)
            
            plt.tight_layout()
            output_filename = f"{os.path.splitext(filename)[0]}_dish_her2_analysis.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            return iou, dice, her2_count, cen17_count, ratio, status
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return 0, 0, 0, 0, 0, "Error"
            
    def start_processing(self):
        original_path = self.original_entry.get().strip()
        groundtruth_path = self.groundtruth_entry.get().strip()
        output_path = self.output_entry.get().strip()
        
        if not original_path:
            messagebox.showerror("Error", "Please select the original images path!")
            return
        
        if not groundtruth_path:
            messagebox.showerror("Error", "Please select the ground truth path!")
            return
        
        if not output_path:
            messagebox.showerror("Error", "Please select the output path!")
            return
        
        if not os.path.exists(original_path):
            messagebox.showerror("Error", "Original images path does not exist!")
            return
        
        if not os.path.exists(groundtruth_path):
            messagebox.showerror("Error", "Ground truth path does not exist!")
            return
        
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
                self.update_status(f"Created output directory: {output_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output directory: {str(e)}")
                return
        
        image_paths = self.get_image_paths(original_path)
        gt_paths = self.get_image_paths(groundtruth_path)
        
        if not image_paths:
            messagebox.showwarning("Warning", "No images found in the original images directory!")
            return
        
        if not gt_paths:
            messagebox.showwarning("Warning", "No images found in the ground truth directory!")
            return
        
        self.original_path = original_path
        self.groundtruth_path = groundtruth_path
        self.output_path = output_path
        
        self.update_status(f"Ready to process {len(image_paths)} images with {self.analysis_type} analysis.")
        
        result = messagebox.askyesno("Confirm Processing", 
                                f"Start {self.analysis_type} analysis?\n\n"
                                f"Original: {original_path}\n"
                                f"Ground Truth: {groundtruth_path}\n"
                                f"Output: {output_path}\n\n"
                                f"Found {len(image_paths)} images to process.")
        
        if result:
            self.update_status("Processing started...")
            
            if self.analysis_type == "FISH":
                self.process_fish_analysis(image_paths, gt_paths, output_path)
            elif self.analysis_type == "DISH":  
                self.process_dish_analysis(image_paths, gt_paths, output_path)
            else: 
                messagebox.showwarning("Warning", "Unknown analysis type selected!")
                self.update_status("Processing aborted - unknown analysis type.")
        
    def process_fish_analysis(self, image_paths, gt_paths, output_dir):
        total_iou = 0
        total_dice = 0
        processed_count = 0
        
        her2_stats = {
            'HER2-Positive': 0,
            'Equivocal': 0,
            'HER2-Negative': 0,
            'Error': 0
        }
        
        results_summary = []
        
        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            
            gt_path = None
            for gt_file in gt_paths:
                gt_name = os.path.splitext(os.path.basename(gt_file))[0]
                if base_name in gt_name or gt_name in base_name:
                    gt_path = gt_file
                    break
            
            if gt_path is None:
                print(f"No ground truth found for {filename}")
                continue
            
            self.update_status(f"Processing {i+1}/{len(image_paths)}: {filename}")
            
            iou, dice, her2_count, cen17_count, ratio, status = self.process_fish_image(
                img_path, gt_path, output_dir, filename)
            
            if iou > 0 or dice > 0:  
                total_iou += iou
                total_dice += dice
                processed_count += 1
                
                her2_stats[status] += 1
                
                results_summary.append({
                    'filename': filename,
                    'her2_count': her2_count,
                    'cen17_count': cen17_count,
                    'ratio': ratio,
                    'status': status,
                    'iou': iou,
                    'dice': dice
                })
                
        if processed_count > 0:
            avg_iou = total_iou / processed_count
            avg_dice = total_dice / processed_count
            
            self.show_analysis_results(processed_count, avg_iou, avg_dice, her2_stats, results_summary, output_dir)
        else:
             messagebox.showwarning("Warning", "No images were successfully processed!")
             self.update_status("Processing completed with errors.")
    
    def process_dish_analysis(self, image_paths, gt_paths, output_dir):
        total_iou = 0
        total_dice = 0
        processed_count = 0
        
        her2_stats = {
            'HER2-Positive': 0,
            'Equivocal': 0,
            'HER2-Negative': 0,
            'Error': 0
        }
        
        results_summary = []
        
        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            
            gt_path = None
            for gt_file in gt_paths:
                gt_name = os.path.splitext(os.path.basename(gt_file))[0]
                if base_name in gt_name or gt_name in base_name:
                    gt_path = gt_file
                    break
            
            if gt_path is None:
                print(f"No ground truth found for {filename}")
                continue
            
            self.update_status(f"Processing {i+1}/{len(image_paths)}: {filename}")
            
            iou, dice, her2_count, cen17_count, ratio, status = self.process_dish_image(img_path, gt_path, output_dir, filename)
            
            if iou > 0 or dice > 0:  
                total_iou += iou
                total_dice += dice
                processed_count += 1
                
                her2_stats[status] += 1
                
                results_summary.append({
                    'filename': filename,
                    'her2_count': her2_count,
                    'cen17_count': cen17_count,
                    'ratio': ratio,
                    'status': status,
                    'iou': iou,
                    'dice': dice
                })
                
        if processed_count > 0:
            avg_iou = total_iou / processed_count
            avg_dice = total_dice / processed_count
            
            self.show_analysis_results(processed_count, avg_iou, avg_dice, her2_stats, results_summary, output_dir)
        else:
            messagebox.showwarning("Warning", "No images were successfully processed!")
            self.update_status("Processing completed with errors.")
    
    def show_analysis_results(self, processed_count, avg_iou, avg_dice, her2_stats, results_summary, output_dir):
        results_window = tk.Toplevel(self.root)
        results_window.title(f"{self.analysis_type} Analysis Results")
        results_window.geometry("800x600")
        results_window.resizable(True, True)
        
        main_frame = tk.Frame(results_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        title_label = tk.Label(scrollable_frame, 
                              text=f"{self.analysis_type} Analysis Results with HER2 Classification", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        summary_frame = tk.LabelFrame(scrollable_frame, text="Summary Statistics", 
                                     font=("Arial", 12, "bold"), padx=10, pady=10)
        summary_frame.pack(fill=tk.X, pady=10)
        
        seg_frame = tk.Frame(summary_frame)
        seg_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(seg_frame, text="Segmentation Performance:", 
                font=("Arial", 11, "bold")).pack(anchor="w")
        tk.Label(seg_frame, text=f"• Images Processed: {processed_count}", 
                font=("Arial", 10)).pack(anchor="w", padx=20)
        tk.Label(seg_frame, text=f"• Average IoU: {avg_iou:.4f}", 
                font=("Arial", 10)).pack(anchor="w", padx=20)
        tk.Label(seg_frame, text=f"• Average Dice Score: {avg_dice:.4f}", 
                font=("Arial", 10)).pack(anchor="w", padx=20)
        
        her2_frame = tk.Frame(summary_frame)
        her2_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(her2_frame, text="HER2 Classification Summary:", 
                font=("Arial", 11, "bold")).pack(anchor="w")
        
        total_classified = sum(her2_stats.values()) - her2_stats.get('Error', 0)
        
        for status, count in her2_stats.items():
            if count > 0:
                percentage = (count / max(total_classified, 1)) * 100 if status != 'Error' else 0
                color_map = {
                    'HER2-Positive': 'green',
                    'Equivocal': 'orange', 
                    'HER2-Negative': 'red',
                    'Error': 'gray'
                }
                
                status_label = tk.Label(her2_frame, 
                                       text=f"• {status}: {count} images ({percentage:.1f}%)",
                                       font=("Arial", 10), 
                                       fg=color_map.get(status, 'black'))
                status_label.pack(anchor="w", padx=20)
        
        details_frame = tk.LabelFrame(scrollable_frame, text="Detailed Results", 
                                     font=("Arial", 12, "bold"), padx=10, pady=10)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        columns = ('File', 'HER2', 'CEN17', 'Ratio', 'Status', 'IoU', 'Dice')
        tree = ttk.Treeview(details_frame, columns=columns, show='headings', height=10)
        
        column_widths = {'File': 150, 'HER2': 60, 'CEN17': 60, 'Ratio': 80, 
                        'Status': 120, 'IoU': 80, 'Dice': 80}
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=column_widths.get(col, 100))
        
        for result in results_summary:
            tree.insert('', tk.END, values=(
                result['filename'],
                result['her2_count'],
                result['cen17_count'],
                f"{result['ratio']:.2f}",
                result['status'],
                f"{result['iou']:.4f}",
                f"{result['dice']:.4f}"
            ))
        
        tree_scroll = ttk.Scrollbar(details_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        buttons_frame = tk.Frame(scrollable_frame)
        buttons_frame.pack(pady=20)
        
        export_btn = tk.Button(buttons_frame, text="Export Results to CSV", 
                              font=("Arial", 11, "bold"),
                              bg="#4CAF50", fg="white",
                              command=lambda: self.export_results_to_csv(results_summary, output_dir))
        export_btn.pack(side=tk.LEFT, padx=10)
        
        open_folder_btn = tk.Button(buttons_frame, text="Open Output Folder", 
                                   font=("Arial", 11, "bold"),
                                   bg="#2196F3", fg="white",
                                   command=lambda: self.open_output_folder(output_dir))
        open_folder_btn.pack(side=tk.LEFT, padx=10)
        
        close_btn = tk.Button(buttons_frame, text="Close", 
                             font=("Arial", 11),
                             bg="#9E9E9E", fg="white",
                             command=results_window.destroy)
        close_btn.pack(side=tk.LEFT, padx=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.update_status(f"Analysis completed! Processed {processed_count} images.")
    
    def export_results_to_csv(self, results_summary, output_dir):
        try:
            import csv
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{self.analysis_type}_analysis_results_{timestamp}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'her2_count', 'cen17_count', 'ratio', 'status', 'iou', 'dice']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in results_summary:
                    writer.writerow(result)
            
            messagebox.showinfo("Export Successful", f"Results exported to:\n{csv_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")
    
    def open_output_folder(self, output_dir):
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Windows":
                os.startfile(output_dir)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
                
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open folder:\n{str(e)}")
            
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = FishDishAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()