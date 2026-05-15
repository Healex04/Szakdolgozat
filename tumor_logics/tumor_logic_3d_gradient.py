"""
YOLO + 3D Gradiens Vizesárok (Sobel Edge) Detektor Modul.

Ez a modul egy számítógépes látásbeli módszert (Sobel gradiens) vet be. 
Az algoritmus megkeresi a képen az éles kontrasztváltásokat (éleket), és ezeket 
matematikailag kivonja az eredeti képből. Ezzel egy mesterséges "vizesárkot" képez 
a daganat peremén, ami megakadályozza, hogy a 3D Region Growing kifolyjon az egészséges szövetekre.
"""

import os
import cv2 as cv
import numpy as np
import shutil
from ultralytics import YOLO
import scipy.ndimage as ndi

class TumorDetector3D_Gradient:
    """
    3D daganatszegmentáló osztály mesterséges éldetektálásos (Sobel) korlátokkal.

    Attributes:
        model_path (str): A YOLO neurális hálózat súlyait tartalmazó fájl (.pt) elérési útja.
        model (ultralytics.YOLO): A betöltött YOLO modell példánya, vagy None hiba esetén.
        last_best_z (int): Az azonosított daganatmag (seed) szelet-indexe a Z-tengelyen.
    """

    def __init__(self, model_path="Models/YOLO_3medfilt.pt"):
        self.model_path = model_path
        self.model = None
        self.last_best_z = -1 
        
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"[HIBA] Nem sikerült inicializálni a YOLO-t: {e}")
        else:
            print(f"[HIBA] Nem található a modell: {model_path}")

    def run_batch_processing(self, input_folder, output_folder, progress_callback=None):
        """
        Lefuttatja a Gradiens Vizesárok alapú 3D szegmentációt.

        A metódus kiszámolja az x és y irányú Sobel gradienseket, és kivonja 
        őket a simított MRI képekből. A keletkezett "vizesárkokat" kombinálja 
        egy "YOLO ketreccel" (biztonsági sáv a bounding box szélén), hogy 
        a gyors 3D tágulás tökéletesen a daganat határán álljon meg. A diagnosztikai 
        képek külön DEBUG mappába kerülnek mentésre.

        Args:
            input_folder (str): A bemeneti MRI képeket tartalmazó forrásmappa.
            output_folder (str): A generált PNG maszkok célmappája.
            progress_callback (callable, optional): Függvény a GUI progress bar frissítéséhez (0-100).

        Returns:
            list of str: A kimentett PNG maszkfájlok teljes elérési útjainak listája.
        """
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        
        debug_folder = os.path.join(output_folder, "DEBUG_LATOVICS")
        os.makedirs(debug_folder, exist_ok=True)

        files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        Z_max = len(files)
        
        if Z_max == 0: return []
        self.last_best_z = -1

        volume = []
        for f in files:
            img_path = os.path.join(input_folder, f)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                if len(img.shape) > 2: img = img[:,:,0]
                volume.append(img)
                
        volume = np.array(volume) 
        _, H, W = volume.shape

        # =================================================================
        # 1. YOLO BOXOK (Legmagasabb magabiztosság keresése)
        # =================================================================
        yolo_boxes = {}
        max_conf = 0.0
        best_z = -1
        best_box = None
        
        for z in range(Z_max):
            img_path = os.path.join(input_folder, files[z])
            results = self.model.predict(img_path, conf=0.4, verbose=False)
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    
                    if len(coords) >= 4:
                        x1, y1, x2, y2 = coords[:4]
                        
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area > (W * H * 0.40): continue
                            
                        if z not in yolo_boxes: yolo_boxes[z] = []
                        yolo_boxes[z].append((x1, y1, x2, y2))
                        
                        if conf > max_conf:
                            max_conf = conf
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

        self.last_best_z = best_z 

        # =================================================================
        # 2. 3D ROI KIVÁGÁSA
        # =================================================================
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

        # =================================================================
        # 3. A MAG INICIALIZÁLÁSA
        # =================================================================
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

        # =================================================================
        # 4. SIMÍTÁS ÉS GRADIENS VIZESÁROK
        # =================================================================
        enhanced_roi = np.zeros_like(vol_roi, dtype=np.uint8)
        grad_mag_debug = np.zeros_like(vol_roi, dtype=np.uint8)
        
        alpha = 1.2 

        for z in range(vol_roi.shape[0]):
            smoothed_img = cv.medianBlur(vol_roi[z], 3)
            img_float = smoothed_img.astype(np.float32)
            
            grad_x = cv.Sobel(img_float, cv.CV_32F, 1, 0, ksize=3)
            grad_y = cv.Sobel(img_float, cv.CV_32F, 0, 1, ksize=3)
            grad_mag = cv.magnitude(grad_x, grad_y)
            
            grad_mag_debug[z] = np.clip(grad_mag, 0, 255).astype(np.uint8)
            
            moat_img = img_float - (alpha * grad_mag)
            enhanced_roi[z] = np.clip(moat_img, 0, 255).astype(np.uint8)

        # =================================================================
        # 5. KÜSZÖB ÉS YOLO KETREC BEÁLLÍTÁSA
        # =================================================================
        struct = ndi.generate_binary_structure(3, 1)
        limit_thresh = seed_thresh * 0.60
        limit_mask = enhanced_roi >= limit_thresh
        
        pad_wall = 8 
        limit_mask[:, :pad_wall, :] = False  
        limit_mask[:, -pad_wall:, :] = False 
        limit_mask[:, :, :pad_wall] = False  
        limit_mask[:, :, -pad_wall:] = False 
        
        cv.imwrite(os.path.join(debug_folder, "1_Eredeti_Kivagas.png"), vol_roi[seed_z_roi])
        cv.imwrite(os.path.join(debug_folder, "2_Tiszta_Gradiens.png"), grad_mag_debug[seed_z_roi])
        cv.imwrite(os.path.join(debug_folder, "3_Vizesarkos_Kep.png"), enhanced_roi[seed_z_roi])
        cv.imwrite(os.path.join(debug_folder, "4_Engedelyezett_Terulet_Mask.png"), (limit_mask[seed_z_roi] * 255).astype(np.uint8))

        # =================================================================
        # 6. VILLÁMGYORS 3D NÖVELÉS A KETRECEN BELÜL
        # =================================================================
        temp_mask = current_mask_3d.copy()
        
        for _ in range(80): 
            prev = temp_mask.copy()
            temp_mask = ndi.binary_dilation(temp_mask, structure=struct, mask=limit_mask)
            if np.array_equal(temp_mask, prev): break
                
        current_mask_3d = temp_mask
        if progress_callback: progress_callback(60)

        # =================================================================
        # 7. MORFOLÓGIAI TISZTÍTÁS
        # =================================================================
        morph_struct = ndi.generate_binary_structure(3, 1)
        
        current_mask_3d = ndi.binary_closing(current_mask_3d, structure=morph_struct, iterations=3)
        current_mask_3d = ndi.binary_opening(current_mask_3d, structure=morph_struct, iterations=1)

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