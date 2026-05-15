"""
YOLO + GMM + 3D Gradiens Vizesárok (EDT) Csúcsmodell.

Ez a modul a kutatás legjobb szegmentációs algoritmusa.
Az eljárás ötvözi a deep learning alapú lokalizációt (YOLO), a statisztikai 
pixel-analízist (Gauss-keverékmodell - GMM) és a morfológiai 3D térfogat-növelést.
A modell egyedi vonásai:
- YOLO által vezérelt, szigorúan lokalizált GMM küszöb-számítás minden szeleten.
- Euklideszi távolság-transzformáción (EDT) alapuló organikus térbeli büntetés.
- Automatikus statisztikai (hisztogram) diagnosztikai ábrák generálása.
"""

import os
import cv2 as cv
import numpy as np
import shutil
from ultralytics import YOLO
import scipy.ndimage as ndi
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import norm

class TumorDetector3D_Gradient_gauss:
    """
    3D MRI daganatdetektáló és szegmentáló osztály.

    A rendszer a YOLO dobozokon belül Gauss-keverékmodellel (GMM) izolálja a 
    daganat intenzitás-csúcsát, többpontos magvetést (multi-seeding) alkalmaz, 
    majd egy Sobel-gradiensekkel és EDT-vel határolt 3D teret tölt ki.

    Attributes:
        model_path (str): A YOLO neurális hálózat súlyait tartalmazó fájl (.pt) elérési útja.
        model (ultralytics.YOLO): A betöltött YOLO modell példánya, vagy None hiba esetén.
        last_best_z (int): Az utolsó futtatás során azonosított legdominánsabb tumorszelet Z-indexe.
    """

    def __init__(self, model_path="Models/YOLO_3medfilt.pt"):
        self.model_path = model_path
        self.model = None
        self.last_best_z = -1 
        
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"[HIBA] YOLO inicializálás sikertelen: {e}")
        else:
            print(f"[HIBA] Modell nem található: {model_path}")

    def run_batch_processing(self, input_folder, output_folder, progress_callback=None, 
                             yolo_conf=0.4,
                             alpha=0.47,
                             dist_penalty=0.88,
                             base_thresh_mult=0.85,
                             strict_boundary_mult=0.33,
                             closing_iters=8,
                             opening_iters=4,
                             dilation_iters=4):
        """
        Lefuttatja a teljes, intelligens 3D tumor-szegmentációs pipeline-t.

        A folyamat lépései:
        1. YOLO lokalizáció.
        2. Szeletenkénti GMM analízis a YOLO dobozokon belül (Küszöb-meghatározás).
        3. Diagnosztikai GMM hisztogram generálása a reprezentatív középső szeleten.
        4. Többpontos magvetés (Multi-seeding) az összes validált szeleten.
        5. Sobel-gradiens alapú élkiemelés (Vizesárok).
        6. EDT alapú organikus távolsági büntetés kiszámítása a magoktól.
        7. 3D Region Growing a dinamikus küszöb-térkép alapján.
        8. 3D Morfológiai tisztítás (Opening, Closing, Lyuktömés).

        Args:
            input_folder (str): A bemeneti MRI képeket tartalmazó forrásmappa.
            output_folder (str): A generált 2D bináris PNG maszkok célmappája.
            progress_callback (callable, optional): Függvény a GUI progress bar frissítéséhez (0-100).
            yolo_conf (float, optional): A YOLO detekció konfidencia küszöbe. Alapértelmezett: 0.4.
            alpha (float, optional): A Sobel gradiens (vizesárok) levonásának intenzitása. Alapértelmezett: 0.47.
            dist_penalty (float, optional): Az EDT távolság alapú küszöb-szigorítás szorzója. Alapértelmezett: 0.88.
            base_thresh_mult (float, optional): A GMM csúcsértékének magvetési engedékenysége. Alapértelmezett: 0.85.
            strict_boundary_mult (float, optional): A végső metszetképzés (vágás) szigorúsága. Alapértelmezett: 0.33.

        Returns:
            list of str: A kimentett PNG maszkfájlok (és a GMM grafikon) teljes elérési útjainak listája.
        """
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        
        files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        Z_max = len(files)
        if Z_max == 0: return []
        self.last_best_z = -1
        
        volume = []
        for f in files:
            img_path = os.path.join(input_folder, f)
            img_array = np.fromfile(img_path, np.uint8)
            img = cv.imdecode(img_array, cv.IMREAD_GRAYSCALE)
            if img is not None:
                if len(img.shape) > 2: img = img[:,:,0]
                volume.append(img)
                
        volume = np.array(volume) 
        _, H, W = volume.shape

        # =================================================================
        # 1. YOLO BOXOK
        # =================================================================
        yolo_boxes = {}
        max_conf = 0.0
        best_z = -1
        best_box = None
        
        for z in range(Z_max):
            img_path = os.path.join(input_folder, files[z])
            results = self.model.predict(img_path, conf=yolo_conf, verbose=False)
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    
                    if len(coords) >= 4:
                        x1, y1, x2, y2 = coords[:4]
                        if (x2 - x1) * (y2 - y1) > (W * H * 0.40): continue
                            
                        if z not in yolo_boxes: yolo_boxes[z] = []
                        yolo_boxes[z].append((x1, y1, x2, y2))
                        
                        if conf > max_conf:
                            max_conf = conf
                            best_z = z
                            best_box = (x1, y1, x2, y2)
                            
            if progress_callback: progress_callback(int((z / Z_max) * 10))

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
        
        pad_z, pad_xy = 5, 25
        z1, z2 = max(0, min_z - pad_z), min(Z_max, max_z + pad_z + 1)
        x1, x2 = max(0, all_x1 - pad_xy), min(W, all_x2 + pad_xy)
        y1, y2 = max(0, all_y1 - pad_xy), min(H, all_y2 + pad_xy)
        
        vol_roi = volume[z1:z2, y1:y2, x1:x2]

        # =================================================================
        # 3. YOLO-VEZÉRELT DINAMIKUS GMM (Szeletenkénti Okos Küszöb)
        # =================================================================
        current_mask_3d = np.zeros_like(vol_roi, dtype=bool)
        slice_thresholds = np.zeros(vol_roi.shape[0])
        slice_thresholds[:] = 150.0 
        
        valid_z_keys = sorted(list(yolo_boxes.keys()))
        middle_tumor_z = valid_z_keys[len(valid_z_keys) // 2] if valid_z_keys else -1
        
        prev_peak = None
        for z_orig in valid_z_keys:
            z_roi = z_orig - z1
            if not (0 <= z_roi < vol_roi.shape[0]): continue
            
            boxes = yolo_boxes[z_orig]
            
            box_mask = np.zeros_like(vol_roi[z_roi], dtype=bool)
            for (bx1, by1, bx2, by2) in boxes:
                rx1, ry1 = max(0, bx1 - x1), max(0, by1 - y1)
                rx2, ry2 = min(vol_roi.shape[2], bx2 - x1), min(vol_roi.shape[1], by2 - y1)
                box_mask[ry1:ry2, rx1:rx2] = True
                
            roi_pixels = vol_roi[z_roi][box_mask]
            valid_px = roi_pixels[roi_pixels > 20].reshape(-1, 1).astype(np.float32)
            
            if valid_px.size > 50: 
                gmm = GaussianMixture(n_components=3, random_state=42, n_init=1)
                gmm.fit(valid_px)
                means = gmm.means_.flatten()
                
                current_peak = np.max(means)
                
                if prev_peak is not None and abs(current_peak - prev_peak) > 30:
                    current_peak = (current_peak + prev_peak) / 2.0
                    
                slice_thresholds[z_roi] = current_peak
                prev_peak = current_peak
                
                # --- GMM DIAGNOSZTIKAI GRAFIKON MENTÉSE ---
                if z_orig == middle_tumor_z:
                    plt.figure(figsize=(8, 5)) 
                    plt.hist(valid_px, bins=50, density=True, alpha=0.4, color='gray', label='Képpontok (YOLO ROI)')
                    
                    x_axis = np.linspace(min(valid_px), max(valid_px), 1000).flatten()
                    combined_pdf = np.zeros_like(x_axis)
                    
                    for weight, mean, covar in zip(gmm.weights_.flatten(), gmm.means_.flatten(), gmm.covariances_.flatten()):
                        std = np.sqrt(covar)
                        pdf = weight * norm.pdf(x_axis, mean, std)
                        combined_pdf += pdf
                        plt.plot(x_axis, pdf, '--', linewidth=1.2, alpha=0.7) 
                        
                    plt.plot(x_axis, combined_pdf, color='black', linewidth=2, label='Kombinált GMM Modell')
                    plt.axvline(x=current_peak, color='red', linestyle='-', linewidth=2, label=f'GMM Csúcs (μ={current_peak:.1f})')
                    
                    shifted_thresh = current_peak * base_thresh_mult
                    plt.axvline(x=shifted_thresh, color='green', linestyle='--', linewidth=2.5, 
                                label=f'Eltolt Küszöb ({(base_thresh_mult*100):.0f}%): {shifted_thresh:.1f}')
                    
                    plt.title(f'Eltolt Csúcs Stratégia a {z_orig+1}. Szeleten')
                    plt.xlabel('Pixel Intenzitás (Fényesség)')
                    plt.ylabel('Sűrűség')
                    plt.legend(loc='upper left', fontsize=9)
                    plt.grid(True, alpha=0.3)
                    
                    plot_filename = os.path.join(output_folder, f"GMM_Eltolt_Csucs_Szelet_{z_orig+1}.png")
                    plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
                    plt.close()
            else:
                if prev_peak is not None:
                    slice_thresholds[z_roi] = prev_peak
                    
        known_z = np.array([z - z1 for z in valid_z_keys if 0 <= z - z1 < vol_roi.shape[0]])
        known_th = np.array([slice_thresholds[z] for z in known_z])
        if len(known_z) > 0:
            slice_thresholds = np.interp(np.arange(vol_roi.shape[0]), known_z, known_th)

        for z_orig in valid_z_keys:
            z_roi = z_orig - z1
            if 0 <= z_roi < vol_roi.shape[0]:
                boxes = yolo_boxes[z_orig]
                th = slice_thresholds[z_roi] * base_thresh_mult
                
                for (bx1, by1, bx2, by2) in boxes:
                    rx1, ry1 = max(0, bx1 - x1), max(0, by1 - y1)
                    rx2, ry2 = min(vol_roi.shape[2], bx2 - x1), min(vol_roi.shape[1], by2 - y1)
                    
                    seed_roi = vol_roi[z_roi, ry1:ry2, rx1:rx2]
                    current_mask_3d[z_roi, ry1:ry2, rx1:rx2] |= (seed_roi >= th)
                    
        struct_3d = ndi.generate_binary_structure(3, 1)
        current_mask_3d = ndi.binary_closing(current_mask_3d, structure=struct_3d, iterations=2)
        if progress_callback: progress_callback(25)

        # =================================================================
        # 4. SIMÍTÁS ÉS GRADIENS VIZESÁROK
        # =================================================================
        enhanced_roi = np.zeros_like(vol_roi, dtype=np.float32)

        for z in range(vol_roi.shape[0]):
            smoothed_img = cv.medianBlur(vol_roi[z], 3).astype(np.float32)
            grad_x = cv.Sobel(smoothed_img, cv.CV_32F, 1, 0, ksize=3)
            grad_y = cv.Sobel(smoothed_img, cv.CV_32F, 0, 1, ksize=3)
            enhanced_roi[z] = smoothed_img - (alpha * cv.magnitude(grad_x, grad_y))

        # =================================================================
        # 5. DINAMIKUS TÁVOLSÁGI BÜNTETÉS (MAGOKTÓL SZÁMOLVA)
        # =================================================================
        dist_from_seeds = ndi.distance_transform_edt(~current_mask_3d)
        
        base_thresh_3d = np.zeros_like(vol_roi, dtype=np.float32)
        for z in range(vol_roi.shape[0]):
            base_thresh_3d[z, :, :] = slice_thresholds[z] * base_thresh_mult

        dynamic_thresh = base_thresh_3d + (dist_from_seeds * dist_penalty)
        limit_mask = enhanced_roi >= dynamic_thresh
        
        # =================================================================
        # 6. 3D NÖVELÉS
        # =================================================================
        temp_mask = current_mask_3d.copy()
        
        for _ in range(120): 
            prev = temp_mask.copy()
            temp_mask = ndi.binary_dilation(temp_mask, structure=struct_3d, mask=limit_mask)
            if np.array_equal(temp_mask, prev): break
                
        current_mask_3d = temp_mask
        if progress_callback: progress_callback(60)

        # =================================================================
        # 7. MORFOLÓGIAI TISZTÍTÁS ÉS LYUKTÖMÉS 
        # =================================================================        
        current_mask_3d = ndi.binary_closing(current_mask_3d, structure=struct_3d, iterations=closing_iters)
        current_mask_3d = ndi.binary_opening(current_mask_3d, structure=struct_3d, iterations=opening_iters)
        current_mask_3d = ndi.binary_fill_holes(current_mask_3d)

        # =================================================================
        # 8. VÉGSŐ TÁGÍTÁS ÉS METSZETKÉPZÉS 
        # =================================================================
        dilated_mask = ndi.binary_dilation(current_mask_3d, structure=struct_3d, iterations=dilation_iters)
        
        strict_boundary_mask = np.zeros_like(vol_roi, dtype=bool)
        for z in range(vol_roi.shape[0]):
            strict_boundary_mask[z] = enhanced_roi[z] >= (slice_thresholds[z] * strict_boundary_mult)

        current_mask_3d = np.logical_and(dilated_mask, strict_boundary_mask)
        
        if np.sum(dilated_mask) > 0 and np.sum(current_mask_3d) == 0:
            print("[FIGYELMEZTETÉS] A metszet során elveszett minden pixel! Nincs átfedés a tágítás és a küszöb között.")

        current_mask_3d = ndi.binary_closing(current_mask_3d, structure=struct_3d, iterations=3)
        current_mask_3d = ndi.binary_dilation(current_mask_3d, structure=struct_3d, iterations=1)

        # =================================================================
        # 9. EREDMÉNY MENTÉSE
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