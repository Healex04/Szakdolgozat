"""
Volumetrikus Kiértékelő és Metrika Modul.

Ez a modul végzi a prediktált 3D tumor maszkok és az orvosi Ground Truth (címkék) 
pixel-szintű és térfogati (volumetric) összehasonlítását. Kiszámítja az orvosi képfeldolgozásban 
standardnak számító metrikákat (Dice, Sensitivity, Specificity).
"""

import os
import cv2 as cv
import numpy as np
from plotting_logic import plot_confusion_matrix, plot_dice_histogram

class Evaluator:
    """
    A szegmentációs eredményeket statisztikailag kiértékelő osztály.
    """
    def __init__(self):
        pass

    def calculate_metrics(self, pred_mask, true_mask):
        """
        Kiszámítja az alapvető statisztikai metrikákat két 2D bináris maszk között.

        Args:
            pred_mask (numpy.ndarray): Az algoritmus által generált bináris maszk.
            true_mask (numpy.ndarray): Az orvosi (Ground Truth) bináris maszk.

        Returns:
            tuple: (tp, tn, fp, fn, dice, sensitivity, specificity, balanced_acc)
                - tp, tn, fp, fn (int): A téveszmátrix elemei pixel darabszámban.
                - dice, sensitivity, specificity, balanced_acc (float): 0.0 és 1.0 közötti teljesítménymutatók.
        """
        pred_bool = (pred_mask > 0)
        true_bool = (true_mask > 0)

        tp = np.sum(np.logical_and(pred_bool, true_bool))
        tn = np.sum(np.logical_and(~pred_bool, ~true_bool))
        fp = np.sum(np.logical_and(pred_bool, ~true_bool))
        fn = np.sum(np.logical_and(~pred_bool, true_bool))
        
        # Simító tényező a nullával való osztás elkerülésére (Laplace smoothing)
        smooth = 1e-6
        dice = (2. * tp + smooth) / (np.sum(pred_bool) + np.sum(true_bool) + smooth)
        
        sensitivity = tp / (tp + fn + smooth)
        specificity = tn / (tn + fp + smooth)
        balanced_acc = (sensitivity + specificity) / 2.0
        
        return tp, tn, fp, fn, dice, sensitivity, specificity, balanced_acc

    def run_evaluation(self, generated_masks_folder, ground_truth_folder, output_base_folder, patient_name, method_name):
        """
        Végrehajtja egy teljes beteg (MRI kötet) 3D volumetrikus kiértékelését.

        A függvény betölti a prediktált és a valós maszkokat, pixelenként összeveti őket, 
        kiszámolja a globális 3D metrikákat, majd generál egy szöveges analitikai riportot 
        és a vizuális ábrákat.

        Args:
            generated_masks_folder (str): A modell által generált maszkok mappája.
            ground_truth_folder (str): A manuálisan annotált orvosi maszkok mappája.
            output_base_folder (str): A főkönyvtár, ahova az eredmény mappa kerül.
            patient_name (str): Az aktuális MRI kötet azonosítója/neve.
            method_name (str): A kiértékelt algoritmus kódja (pl. '3D_GRADIENT_GAUSS').

        Returns:
            tuple: (success, save_dir, vol_dice)
                - success (bool): Igaz, ha sikeres volt az értékelés, Hamis ha nem talált fájlokat.
                - save_dir (str): A generált riportokat tartalmazó mappa elérési útja.
                - vol_dice (float): A végső, 3D globális Dice Score (0.0 - 1.0).
        """
        save_dir = os.path.join(output_base_folder, f"EVAL_{patient_name}_{method_name}")
        os.makedirs(save_dir, exist_ok=True)

        gen_files = sorted([f for f in os.listdir(generated_masks_folder) if f.lower().endswith('.png')])
        
        g_tp, g_tn, g_fp, g_fn = 0, 0, 0, 0
        dice_scores = []
        processed_count = 0

        for fname in gen_files:
            gen_path = os.path.join(generated_masks_folder, fname)
            base_name = os.path.splitext(fname)[0]
            
            possible_names = [base_name + "_mask", base_name, base_name.replace("_mask", "")]
            gt_path = None
            
            for name in possible_names:
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
                    full_p = os.path.join(ground_truth_folder, name + ext)
                    if os.path.exists(full_p):
                        gt_path = full_p
                        break
                if gt_path: 
                    break
            
            if not gt_path: 
                continue

            pred_img = cv.imread(gen_path, cv.IMREAD_GRAYSCALE)
            true_img = cv.imread(gt_path, cv.IMREAD_GRAYSCALE)
            
            if pred_img is None or true_img is None: 
                continue
            
            if pred_img.shape != true_img.shape:
                true_img = cv.resize(true_img, (pred_img.shape[1], pred_img.shape[0]), interpolation=cv.INTER_NEAREST)

            tp, tn, fp, fn, dice, _, _, _ = self.calculate_metrics(pred_img, true_img)
            
            g_tp += tp
            g_tn += tn
            g_fp += fp
            g_fn += fn
            
            dice_scores.append(dice)
            processed_count += 1

        if processed_count == 0:
            return False, None, 0.0

        # --- 3D Volumetrikus Metrikák Számítása ---
        sum_dice_denom = 2.0 * g_tp + g_fp + g_fn
        vol_dice = (2.0 * g_tp) / sum_dice_denom if sum_dice_denom > 0 else 1.0
        
        sum_sens_denom = g_tp + g_fn
        vol_sens = g_tp / sum_sens_denom if sum_sens_denom > 0 else 1.0
            
        sum_spec_denom = g_tn + g_fp
        vol_spec = g_tn / sum_spec_denom if sum_spec_denom > 0 else 1.0
            
        vol_bal_acc = (vol_sens + vol_spec) / 2.0

        # --- Kimeneti Fájlok Generálása ---
        global_cm_path = os.path.join(save_dir, f"GLOBAL_CM_{method_name}.png")
        plot_confusion_matrix(g_tp, g_tn, g_fp, g_fn, global_cm_path, title=f"Globális CM - {patient_name}")
        
        hist_path = os.path.join(save_dir, f"Dice_Histogram_{method_name}.png")
        plot_dice_histogram(dice_scores, hist_path)
        
        report_path = os.path.join(save_dir, "report.txt")
        with open(report_path, "w", encoding='utf-8') as f:
            f.write("=== MRI TUMOR DETEKTÁLÁS EREDMÉNYEK ===\n")
            f.write(f"Beteg azonosító: {patient_name}\n")
            f.write(f"Használt módszer: {method_name}\n")
            f.write(f"Kiértékelt szeletek száma: {processed_count}\n\n")
            
            f.write("--- FŐ 3D VOLUMETRIKUS METRIKÁK ---\n")
            f.write(f"3D Dice Score (Átfedés):    {vol_dice:.4f}\n")
            f.write(f"3D Balanced Accuracy:       {vol_bal_acc:.4f}\n")
            f.write(f"3D Sensitivity (Recall):    {vol_sens:.4f}\n")
            f.write(f"3D Specificity:             {vol_spec:.4f}\n\n")
            
            f.write("--- PIXEL ALAPÚ CONFUSION MATRIX (Globális összeg) ---\n")
            f.write(f"True Positive (Találat):    {g_tp}\n")
            f.write(f"True Negative (Háttér):     {g_tn}\n")
            f.write(f"False Positive (Téves):     {g_fp}\n")
            f.write(f"False Negative (Hiányzó):   {g_fn}\n")
        
        return True, save_dir, vol_dice