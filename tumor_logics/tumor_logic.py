"""
Egyszerű YOLO + 2D Morfológiai Region Growing Detektor Modul.

Ez a modul egy alapvető, 2D-s megközelítést alkalmaz a tumorok szegmentálására.
A YOLO modell által meghatározott határoló dobozokon (Bounding Box) belül 
egy intenzitás-alapú (percentilis) morfológiai régió-növelést hajt végre.
"""

import os
import cv2 as cv
import numpy as np
import shutil
from ultralytics import YOLO

class TumorDetector:
    """
    2D alapú baseline tumor detektor és szegmentáló.
    """
    def __init__(self, model_path="Models/YOLO_3medfilt.pt"):
        self.model_path = model_path
        self.model = None
        
        if os.path.exists(model_path):
            abs_model_path = model_path
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.abspath(os.path.join(current_dir, ".."))
            abs_model_path = os.path.join(root_dir, model_path)
            
        if os.path.exists(abs_model_path):
            try:
                self.model = YOLO(abs_model_path)
            except Exception as e:
                print(f"[HIBA] Nem sikerült inicializálni a YOLO-t: {e}")
        else:
            print(f"[HIBA] Nem található a YOLO modell ezen az útvonalon: {abs_model_path}")

    def morphological_region_growing(self, img_roi, seed_percentile=92.5, limit_percentile=50, max_iter=100):
        """Egyszerű morfológiai intenzitás-alapú régió növelés."""
        if img_roi.size == 0:
            return np.zeros_like(img_roi, dtype=np.uint8)

        seed_thresh = np.percentile(img_roi, seed_percentile)
        limit_thresh = np.percentile(img_roi, limit_percentile)

        current_mask = (img_roi >= seed_thresh).astype(np.uint8) * 255
        limit_mask = (img_roi >= limit_thresh).astype(np.uint8) * 255
        
        kernel = np.ones((3,3), np.uint8)

        for i in range(max_iter):
            prev_mask = current_mask.copy()
            dilated = cv.dilate(current_mask, kernel, iterations=1)
            current_mask = cv.bitwise_and(dilated, limit_mask)
            if np.array_equal(current_mask, prev_mask):
                break
                
        return current_mask

    def detect_on_single_image(self, img_path):
        """Egyetlen kép detektálása YOLO-val és 2D RG szegmentációval."""
        if self.model is None:
            return False, None

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            return False, None
        
        if len(img.shape) == 3:
            img = img[:, :, 0]
        
        H, W = img.shape
        final_mask = np.zeros_like(img, dtype=np.uint8)
        found_tumor = False

        try:
            results = self.model.predict(img_path, conf=0.4, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes and len(result.boxes) > 0:
                    for box in result.boxes:
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        
                        if len(coords) >= 4:
                            x1, y1, x2, y2 = coords[:4]

                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(W, x2), min(H, y2)

                            roi = img[y1:y2, x1:x2]
                            if roi.size == 0: continue

                            roi_mask = self.morphological_region_growing(roi)
                            final_mask[y1:y2, x1:x2] = cv.bitwise_or(final_mask[y1:y2, x1:x2], roi_mask)
                            
                            if np.sum(roi_mask) > 0:
                                found_tumor = True
                                
        except Exception as e:
            print(f"[HIBA] Detektálás közben ({img_path}): {e}")
            return False, None

        return found_tumor, final_mask

    def run_batch_processing(self, input_folder, output_folder, progress_callback=None):
        """Mappa feldolgozása a baseline 2D detektorral."""
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder, ignore_errors=True)
        os.makedirs(output_folder, exist_ok=True)

        files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        total_files = len(files)
        
        h, w = 240, 240
        if files:
            sample = cv.imread(os.path.join(input_folder, files[0]), cv.IMREAD_GRAYSCALE)
            if sample is not None:
                h, w, _ = sample.shape

        final_generated_masks = []

        for idx, fname in enumerate(files):
            full_path = os.path.join(input_folder, fname)
            mask_to_save = np.zeros((h, w), dtype=np.uint8)
            
            try:
                res = self.detect_on_single_image(full_path)
                
                if isinstance(res, tuple) and len(res) == 2:
                    has_tumor, mask = res
                    if has_tumor and mask is not None:
                        mask_to_save = mask
            except Exception as e:
                pass

            mask_name = os.path.splitext(fname)[0] + ".png"
            save_path = os.path.join(output_folder, mask_name)
            cv.imwrite(save_path, mask_to_save)
            final_generated_masks.append(save_path)
            
            if progress_callback:
                progress_callback(int((idx + 1) / total_files * 100))

        return final_generated_masks