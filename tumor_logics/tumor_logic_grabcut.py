"""
YOLO + GrabCut (és Region Growing) Hibrid Detektor Modul.

Ez a modul a YOLO bounding boxok által határolt területen belül a 
számítógépes látás egyik klasszikus szegmentáló algoritmusát, a GrabCutot alkalmazza.

Két üzemmóddal rendelkezik:

1. Sima GrabCut (Bounding Box alapú inicializálás).
2. Kombinált módszer: Először egy durva Region Growing (RG) maszkot készít, majd ezt a maszkot adja át a GrabCut-nak precíziós finomhangolásra.
"""

import os
import cv2 as cv
import numpy as np
import shutil
from ultralytics import YOLO

class TumorDetectorGrabCut:
    """
    A YOLO és GrabCut alapú szegmentálást végző osztály.

    Attributes:
        model_path (str): A YOLO neurális hálózat súlyait tartalmazó fájl (.pt) elérési útja.
        use_region_growing (bool): Kapcsoló a hibrid (RG + GrabCut) és a sima GrabCut mód között.
        model (ultralytics.YOLO): A betöltött YOLO modell példánya, vagy None, ha a betöltés sikertelen.
    """

    def __init__(self, model_path="Models/YOLO_3medfilt.pt", use_region_growing=False):
        self.model_path = model_path
        self.model = None
        self.use_region_growing = use_region_growing
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, ".."))
        abs_model_path = os.path.join(root_dir, model_path)
        
        if os.path.exists(abs_model_path):
            try:
                self.model = YOLO(abs_model_path)
            except Exception as e:
                print(f"[HIBA] Nem sikerült inicializálni a YOLO-t: {e}")
        else:
            print(f"[HIBA] Nem található a modell ezen az útvonalon: {abs_model_path}")

    def morphological_region_growing(self, img_roi, seed_percentile=85, limit_percentile=40, max_iter=50):
        """
        Durva 2D-s morfológiai régió-növelést végez, amely a GrabCut inicializálásához (maszk) kell.

        Args:
            img_roi (numpy.ndarray): A vizsgálandó 2D-s szürkeárnyalatos képrészlet (YOLO doboz).
            seed_percentile (int, optional): A kiinduló magok fényességi percentilise. Alapértelmezett: 85.
            limit_percentile (int, optional): A tágulási határ fényességi percentilise. Alapértelmezett: 40.
            max_iter (int, optional): A tágítási iterációk maximális száma. Alapértelmezett: 50.

        Returns:
            numpy.ndarray: Egy durva, 2D bináris maszk a GrabCut előtér/háttér meghatározásához.
        """
        if img_roi.size == 0: return np.zeros_like(img_roi, dtype=np.uint8)

        seed_thresh = np.percentile(img_roi, seed_percentile)
        limit_thresh = np.percentile(img_roi, limit_percentile)

        current_mask = (img_roi >= seed_thresh).astype(np.uint8) * 255
        limit_mask = (img_roi >= limit_thresh).astype(np.uint8) * 255
        kernel = np.ones((3,3), np.uint8)

        for i in range(max_iter):
            prev_mask = current_mask.copy()
            dilated = cv.dilate(current_mask, kernel, iterations=1)
            current_mask = cv.bitwise_and(dilated, limit_mask)
            if np.array_equal(current_mask, prev_mask): break
                
        return current_mask

    def apply_grabcut(self, img_roi, use_rect=False, mask_input=None):
        """
        GrabCut szegmentációt hajt végre a megadott képrészleten.

        A függvény vagy egy Bounding Box-ot (use_rect=True), vagy egy előzetesen 
        kiszámolt bináris maszkot (mask_input) használ a háttér és az előtér inicializálásához.

        Args:
            img_roi (numpy.ndarray): A szegmentálandó szürkeárnyalatos képrészlet.
            use_rect (bool, optional): Használjon-e téglalap-alapú inicializálást. Alapértelmezett: False.
            mask_input (numpy.ndarray, optional): Egy előzetes bináris maszk a pontosabb 
                                                  inicializáláshoz (GC_INIT_WITH_MASK).

        Returns:
            numpy.ndarray: A GrabCut által finomított végső bináris maszk.
        """
        h, w = img_roi.shape[:2]
        
        if h < 10 or w < 10:
            if mask_input is not None: return mask_input
            return np.zeros((h, w), dtype=np.uint8)

        if len(img_roi.shape) == 2:
            img_color = cv.cvtColor(img_roi, cv.COLOR_GRAY2BGR)
        else:
            img_color = img_roi.copy()
            
        img_color = cv.GaussianBlur(img_color, (5, 5), 0)

        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        try:
            if mask_input is not None:
                pad = 15
                
                mask[:] = cv.GC_PR_BGD 
                mask[0:pad, :] = cv.GC_BGD
                mask[-pad:, :] = cv.GC_BGD
                mask[:, 0:pad] = cv.GC_BGD
                mask[:, -pad:] = cv.GC_BGD
                
                mask[mask_input > 0] = cv.GC_PR_FGD
                
                kernel = np.ones((5,5), np.uint8)
                core_mask = cv.erode(mask_input, kernel, iterations=2)
                mask[core_mask > 0] = cv.GC_FGD
                
                cv.grabCut(img_color, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)

            elif use_rect:
                pad = 15
                rect = (pad, pad, w - 2*pad, h - 2*pad)
                if rect[2] > 0 and rect[3] > 0:
                    cv.grabCut(img_color, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
                else:
                    return np.zeros((h, w), dtype=np.uint8)
            else:
                return np.zeros((h, w), dtype=np.uint8)

            mask2 = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
            return mask2

        except Exception as e:
            print(f"[HIBA] GrabCut futás: {e}")
            return np.zeros((h, w), dtype=np.uint8)

    def detect_on_single_image(self, img_path):
        """
        Egyetlen képen futtatja le a YOLO detektálást és a GrabCut szegmentálást.

        Args:
            img_path (str): A vizsgálandó képfájl elérési útja.

        Returns:
            tuple: (has_tumor, mask) bináris találati flag és az elkészült maszk.
        """
        if self.model is None: return False, None

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None: return False, None
        
        if len(img.shape) == 3:
            if img.shape[2] == 3: img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            elif img.shape[2] == 4: img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
            elif img.shape[2] == 1: img = img[:, :, 0]
        
        H, W = img.shape
        final_mask = np.zeros_like(img, dtype=np.uint8)
        found_tumor = False

        try:
            results = self.model.predict(img_path, conf=0.2, verbose=False)
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    if len(coords) >= 4:
                        x1, y1, x2, y2 = coords[:4]
                        
                        pad = 15
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(W, x2 + pad)
                        y2 = min(H, y2 + pad)

                        roi = img[y1:y2, x1:x2]
                        if roi.size == 0: continue

                        roi_mask = None
                        
                        if self.use_region_growing:
                            rg_mask = self.morphological_region_growing(roi)
                            if np.sum(rg_mask) == 0: continue
                            roi_mask = self.apply_grabcut(roi, mask_input=rg_mask)
                        else:
                            roi_mask = self.apply_grabcut(roi, use_rect=True)

                        final_mask[y1:y2, x1:x2] = cv.bitwise_or(final_mask[y1:y2, x1:x2], roi_mask)
                        
                        if np.sum(roi_mask) > 0:
                            found_tumor = True
                                
        except Exception as e:
            print(f"[HIBA] {img_path}: {e}")
            return False, None

        return found_tumor, final_mask

    def run_batch_processing(self, input_folder, output_folder, progress_callback=None):
        """
        A teljes folyamatot futtatja egy mappa összes MRI szeletén.

        Args:
            input_folder (str): A bemeneti képeket tartalmazó forrásmappa.
            output_folder (str): A generált PNG maszkok célmappája.
            progress_callback (callable, optional): Függvény a GUI progress bar frissítéséhez (0-100).

        Returns:
            list of str: A generált maszkfájlok teljes elérési útjainak listája.
        """
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder, ignore_errors=True)
        os.makedirs(output_folder, exist_ok=True)

        files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        total_files = len(files)
        
        h, w = 240, 240
        if files:
            sample = cv.imread(os.path.join(input_folder, files[0]), cv.IMREAD_GRAYSCALE)
            if sample is not None:
                h, w = sample.shape[:2]

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