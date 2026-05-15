import os
import cv2 as cv
import numpy as np

def analyze_intensity_drift(dataset_folder):
    patient_folders = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    if not patient_folders: return

    total_drift_start_to_mid = []
    total_drift_mid_to_end = []

    for patient in patient_folders:
        patient_path = os.path.join(dataset_folder, patient)
        files = sorted([f for f in os.listdir(patient_path) if f.lower().endswith(('.jpg', '.png'))])
        if len(files) < 10: continue
            
        slice_intensities = []
        for f in files:
            img_path = os.path.join(patient_path, f)
            img = cv.imdecode(np.fromfile(img_path, np.uint8), cv.IMREAD_GRAYSCALE)
            
            if img is not None:
                if len(img.shape) > 2: img = img[:,:,0]
                valid_pixels = img[img > 15]
                if len(valid_pixels) > 5000:
                    slice_intensities.append(np.median(valid_pixels))

        if len(slice_intensities) > 10:
            idx_start, idx_mid, idx_end = 0, len(slice_intensities) // 2, len(slice_intensities) - 1
            drift_1 = slice_intensities[idx_mid] - slice_intensities[idx_start]
            drift_2 = slice_intensities[idx_end] - slice_intensities[idx_mid]
            total_drift_start_to_mid.append(drift_1)
            total_drift_mid_to_end.append(drift_2)

    if total_drift_start_to_mid:
        avg_drift_1 = np.mean(total_drift_start_to_mid)
        avg_drift_2 = np.mean(total_drift_mid_to_end)
        
        print(f"Átlagos drift (Eleje -> Közép): {avg_drift_1:+.2f}")
        print(f"Átlagos drift (Közép -> Vég):   {avg_drift_2:+.2f}")

if __name__ == "__main__":
    DATASET_DIR = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val" 
    analyze_intensity_drift(DATASET_DIR)