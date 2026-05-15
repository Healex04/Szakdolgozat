"""
YOLO + Alap 3D Region Growing (Adaptív Térfogatfigyeléssel).

Ez a modul egy tisztán 3D-s térfogati növelést alkalmaz. Különlegessége az 
"Adaptív Térfogat-figyelő" (Volume Watcher) algoritmus, amely megakadályozza, 
hogy a maszk kiterjedjen az egészséges agyszövetre a küszöbértékek fokozatos 
csökkentése és a növekedési ütem (growth rate) monitorozása révén.
"""

import os
import cv2 as cv
import numpy as np
import shutil
from ultralytics import YOLO
import scipy.ndimage as ndi

class TumorDetector3D:
    """
    3D térfogati daganatszegmentáló osztály adaptív küszöbérték-védelemmel.

    Attributes:
        model_path (str): A YOLO neurális hálózat súlyait tartalmazó fájl (.pt) elérési útja.
        model (ultralytics.YOLO): A betöltött YOLO modell példánya, vagy None, ha a betöltés sikertelen.
    """

    def __init__(self, model_path="Models/YOLO_3medfilt.pt"):
        self.model_path = model_path
        self.model = None
        
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"[HIBA] Nem sikerült inicializálni a YOLO-t: {e}")
        else:
            print(f"[HIBA] Nem található a modell: {model_path}")

    def run_batch_processing(self, input_folder, output_folder, progress_callback=None):
        """
        Lefuttatja a teljes 3D-s szegmentációt és az adaptív térfogat-ellenőrzést 
        egy mappa összes MRI szeletén.

        A folyamat lépései: 1) 3D Volumen felépítése -> 2) YOLO dobozok kinyerése ->
        3) 3D ROI vágás -> 4) Mag (seed) lerakása a legerősebb szeleten -> 
        5) Otsu küszöb kiszámítása -> 6) Adaptív 3D terjeszkedés a robbanás megelőzésével.

        Args:
            input_folder (str): A bemeneti képeket tartalmazó forrásmappa.
            output_folder (str): A generált PNG maszkok célmappája.
            progress_callback (callable, optional): Függvény a GUI progress bar frissítéséhez (0-100).

        Returns:
            list of str: A kimentett PNG maszkfájlok teljes elérési útjainak listája.
        """
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        Z_max = len(files)
        
        if Z_max == 0: return []
        
        volume = []
        for f in files:
            img_path = os.path.join(input_folder, f)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                if len(img.shape) > 2: img = img[:,:,0]
                volume.append(img)
                
        volume = np.array(volume) 
        _, H, W = volume.shape

        # 2. YOLO BOXOK
        yolo_boxes = {}
        max_area = 0
        best_z = -1
        best_box = None
        
        for z in range(Z_max):
            img_path = os.path.join(input_folder, files[z])
            results = self.model.predict(img_path, conf=0.2, verbose=False)
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    if len(coords) >= 4:
                        x1, y1, x2, y2 = coords[:4]
                        area = (x2 - x1) * (y2 - y1)
                        if z not in yolo_boxes: yolo_boxes[z] = []
                        yolo_boxes[z].append((x1, y1, x2, y2))
                        if area > max_area:
                            max_area = area
                            best_z = z
                            best_box = (x1, y1, x2, y2)
                            
            if progress_callback: progress_callback(int((z / Z_max) * 20))

        generated_masks = []
        if best_z == -1:
            for f in files:
                mask_name = os.path.splitext(f)[0] + ".png"
                save_path = os.path.join(output_folder, mask_name)
                cv.imwrite(save_path, np.zeros((H, W), dtype=np.uint8))
                generated_masks.append(save_path)
            return generated_masks

        # 3. 3D ROI KIVÁGÁSA
        all_x1 = min([b[0] for boxes in yolo_boxes.values() for b in boxes])
        all_y1 = min([b[1] for boxes in yolo_boxes.values() for b in boxes])
        all_x2 = max([b[2] for boxes in yolo_boxes.values() for b in boxes])
        all_y2 = max([b[3] for boxes in yolo_boxes.values() for b in boxes])
        min_z, max_z = min(yolo_boxes.keys()), max(yolo_boxes.keys())
        
        pad_z, pad_xy = 3, 15
        z1, z2 = max(0, min_z - pad_z), min(Z_max, max_z + pad_z + 1)
        x1, x2 = max(0, all_x1 - pad_xy), min(W, all_x2 + pad_xy)
        y1, y2 = max(0, all_y1 - pad_xy), min(H, all_y2 + pad_xy)
        
        vol_roi = volume[z1:z2, y1:y2, x1:x2]

        # 4. A MAG INICIALIZÁLÁSA
        current_mask_3d = np.zeros_like(vol_roi, dtype=bool)
        seed_z_roi = best_z - z1
        bx1, by1, bx2, by2 = best_box
        rx1, ry1 = max(0, bx1 - x1), max(0, by1 - y1)
        rx2, ry2 = min(vol_roi.shape[2], bx2 - x1), min(vol_roi.shape[1], by2 - y1)
        
        seed_roi_2d = vol_roi[seed_z_roi, ry1:ry2, rx1:rx2]
        seed_thresh = 255
        if seed_roi_2d.size > 0:
            seed_thresh = np.percentile(seed_roi_2d, 90)
            current_mask_3d[seed_z_roi, ry1:ry2, rx1:rx2] = seed_roi_2d >= seed_thresh

        # 5. OTSU ALAPÉRTÉK
        valid_pixels = vol_roi[vol_roi > 10].astype(np.uint8)
        if len(valid_pixels) > 0:
            otsu_val, _ = cv.threshold(valid_pixels, 0, 255, cv.THRESH_OTSU)
        else:
            otsu_val = 50

        # =================================================================
        # 6. ADAPTÍV TÉRFOGAT-FIGYELÉS
        # =================================================================
        struct = ndi.generate_binary_structure(3, 1)
        multipliers = [1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7]
        
        best_3d_mask = current_mask_3d.copy()
        prev_volume = np.sum(best_3d_mask)

        for i, mult in enumerate(multipliers):
            limit_thresh = min(otsu_val * mult, seed_thresh * 0.95)
            limit_mask = vol_roi >= limit_thresh
            
            temp_mask = best_3d_mask.copy()
            
            for _ in range(30): 
                prev = temp_mask.copy()
                temp_mask = ndi.binary_dilation(temp_mask, structure=struct, mask=limit_mask)
                if np.array_equal(temp_mask, prev): break
            
            current_volume = np.sum(temp_mask)
            
            if prev_volume > 0:
                growth_rate = (current_volume - prev_volume) / prev_volume
                
                # Térfogat robbanás detektálva, visszalépés
                if i != 0 and growth_rate > 0.75: 
                    break
            
            best_3d_mask = temp_mask.copy()
            prev_volume = current_volume
            
            if progress_callback: progress_callback(20 + int((i / len(multipliers)) * 50))

        current_mask_3d = best_3d_mask 

        # =================================================================
        # 8. AZ EREDMÉNY VISSZASZELETELÉSE ÉS MENTÉSE
        # =================================================================
        final_volume_mask = np.zeros_like(volume, dtype=np.uint8)
        final_volume_mask[z1:z2, y1:y2, x1:x2] = current_mask_3d.astype(np.uint8) * 255

        for z, fname in enumerate(files):
            mask_name = os.path.splitext(fname)[0] + ".png"
            save_path = os.path.join(output_folder, mask_name)
            cv.imwrite(save_path, final_volume_mask[z])
            generated_masks.append(save_path)
            
            if progress_callback: progress_callback(70 + int((z / Z_max) * 30))

        return generated_masks