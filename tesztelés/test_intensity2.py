import os
import cv2 as cv
import numpy as np

def analyze_masked_intensity_drift(images_dir, masks_dir):
    patient_folders = sorted([f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))])
    if not patient_folders: return

    brain_drift_1_list, brain_drift_2_list = [], []
    tumor_drift_1_list, tumor_drift_2_list = [], []

    for patient in patient_folders:
        patient_img_path = os.path.join(images_dir, patient)
        patient_mask_path = os.path.join(masks_dir, patient)
        if not os.path.exists(patient_mask_path): continue

        img_files = sorted([f for f in os.listdir(patient_img_path) if f.lower().endswith(('.jpg', '.png'))])
        brain_intensities, tumor_intensities, valid_z_indices = {}, {}, []

        for z, f in enumerate(img_files):
            img_path = os.path.join(patient_img_path, f)
            img = cv.imdecode(np.fromfile(img_path, np.uint8), cv.IMREAD_GRAYSCALE)
            
            base_name, ext = os.path.splitext(f)
            mask_path = os.path.join(patient_mask_path, base_name + "_mask" + ext)
            if not os.path.exists(mask_path): mask_path = os.path.join(patient_mask_path, f)
            if not os.path.exists(mask_path): continue
                
            mask = cv.imdecode(np.fromfile(mask_path, np.uint8), cv.IMREAD_GRAYSCALE)

            if img is not None and mask is not None:
                if len(img.shape) > 2: img = img[:,:,0]
                if len(mask.shape) > 2: mask = mask[:,:,0]

                mask_bool = mask > 127
                tumor_pixels = img[mask_bool]
                brain_pixels = img[(mask_bool == False) & (img > 15)]

                if len(tumor_pixels) > 50 and len(brain_pixels) > 1000:
                    tumor_intensities[z] = np.median(tumor_pixels)
                    brain_intensities[z] = np.median(brain_pixels)
                    valid_z_indices.append(z)

        if len(valid_z_indices) >= 5:
            idx_start, idx_mid, idx_end = valid_z_indices[0], valid_z_indices[len(valid_z_indices) // 2], valid_z_indices[-1]

            t_drift_1 = tumor_intensities[idx_mid] - tumor_intensities[idx_start]
            t_drift_2 = tumor_intensities[idx_end] - tumor_intensities[idx_mid]
            tumor_drift_1_list.append(t_drift_1)
            tumor_drift_2_list.append(t_drift_2)

            b_drift_1 = brain_intensities[idx_mid] - brain_intensities[idx_start]
            b_drift_2 = brain_intensities[idx_end] - brain_intensities[idx_mid]
            brain_drift_1_list.append(b_drift_1)
            brain_drift_2_list.append(b_drift_2)

    if brain_drift_1_list:
        print("\n--- AGYSZÖVET (Drift) ---")
        print(f"Eleje -> Közép: {np.mean(brain_drift_1_list):+.2f}")
        print(f"Közép -> Vége:  {np.mean(brain_drift_2_list):+.2f}")
        
        print("\n--- TUMOR (Biológiai + Gép okozta drift) ---")
        print(f"Eleje -> Közép: {np.mean(tumor_drift_1_list):+.2f}")
        print(f"Közép -> Vége:  {np.mean(tumor_drift_2_list):+.2f}")

if __name__ == "__main__":
    IMAGES_DIR = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val"
    MASKS_DIR = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Labels\Val"
    analyze_masked_intensity_drift(IMAGES_DIR, MASKS_DIR)