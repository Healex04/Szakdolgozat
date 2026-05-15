import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2 as cv
import re

def get_best_dice_and_transform(pred, gt):
    best_dice = 0.0
    best_transform = "Eredeti"
    
    transforms = [
        ("Eredeti", pred),
        ("Z-tengely fordítva", pred[::-1, :, :]),
        ("Y-tengely tükrözés", pred[:, ::-1, :]),
        ("X-tengely tükrözés", pred[:, :, ::-1]),
        ("Z és Y fordítva", pred[::-1, ::-1, :]),
        ("X és Y tükrözés", pred[:, ::-1, ::-1])
    ]
    
    if pred.shape[1] == pred.shape[2]:
        transforms.extend([
            ("Transzponált", np.transpose(pred, (0, 2, 1))),
            ("Transzponált + Y tükrözés", np.transpose(pred[:, ::-1, :], (0, 2, 1))),
            ("Transzponált + Z fordítva", np.transpose(pred[::-1, :, :], (0, 2, 1)))
        ])
        
    for name, trans_pred in transforms:
        intersection = np.sum(trans_pred * gt)
        p_sum = np.sum(trans_pred)
        g_sum = np.sum(gt)
        
        dice = (2.0 * intersection) / (p_sum + g_sum) if (p_sum + g_sum) > 0 else 1.0
        
        if dice > best_dice:
            best_dice = dice
            best_transform = name
            
    return best_dice, best_transform

def evaluate_nifti_vs_png():
    pred_dir = r"C:\nnunet_data\temp_nnunet_out"
    gt_base_dir = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Labels\Val"
    results_csv = "nnunet_final_benchmark_results.csv"

    print("[INFO] Kiértékelés indítása...\n")

    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
    all_results = []

    for pf in pred_files:
        patient = pf.replace('.nii.gz', '')
        pred_path = os.path.join(pred_dir, pf)
        gt_patient_folder = os.path.join(gt_base_dir, patient)

        if not os.path.exists(gt_patient_folder):
            continue

        pred_img = sitk.ReadImage(pred_path)
        pred_array = sitk.GetArrayFromImage(pred_img)
        pred_array = (pred_array > 0).astype(np.uint8) 

        gt_files = [f for f in os.listdir(gt_patient_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        gt_files = sorted(gt_files, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        
        gt_slices = []
        for gf in gt_files:
            img_path = os.path.join(gt_patient_folder, gf)
            try:
                with open(img_path, "rb") as f:
                    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                slice_img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
            except Exception:
                continue

            slice_img = (slice_img > 127).astype(np.uint8)
            gt_slices.append(slice_img)
        
        if len(gt_slices) == 0:
            continue

        gt_array = np.array(gt_slices)

        if pred_array.shape != gt_array.shape:
            print(f"[HIBA] Méret eltérés a(z) {patient} esetében!")
            continue

        best_dice, transform_used = get_best_dice_and_transform(pred_array, gt_array)

        print(f" -> {patient} | Dice: {best_dice:.4f}")
        
        all_results.append({
            'Patient': patient,
            'Dice_Score': best_dice,
        })

    if all_results:
        df = pd.DataFrame(all_results)
        
        # Mappa létrehozása és a mentési útvonal összeállítása
        output_folder = "optimalizacio"
        os.makedirs(output_folder, exist_ok=True)
        final_save_path = os.path.join(output_folder, results_csv)
        
        # Mentés az új mappába
        df.to_csv(final_save_path, index=False)
        
        print("\n=== KIÉRTÉKELÉS VÉGE ===")
        print(f"Átlagos nnU-Net Dice pontszám: {df['Dice_Score'].mean():.4f}")
        print(f"Eredmények kimentve: {final_save_path}")

if __name__ == "__main__":
    evaluate_nifti_vs_png()