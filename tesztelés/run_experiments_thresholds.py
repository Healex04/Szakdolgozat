import os
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def find_intersection(m1, m2, std1, std2, w1, w2):
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 /(2*std2**2) - np.log((w1/std1) / (w2/std2))
    
    roots = np.roots([a, b, c])
    valid_roots = [r for r in roots if min(m1, m2) < r < max(m1, m2)]
    return valid_roots[0] if valid_roots else (m1+m2)/2.0

def calculate_dice(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    if total == 0: return 1.0
    return 2.0 * intersection / total

def run_threshold_experiments(images_dir, masks_dir):
    patient_folders = sorted([f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))])
    
    results_dice = {
        "0. Alapvonal (90. Percentilis)": [],
        "0. Alapvonal (95. Percentilis)": [],
        "1. K-Means (3-4 klaszter köze)": [],
        "2. K-Means (4-5 klaszter köze)": [],
        "2,5. K-Means (3-5 klaszter köze)": [],
        "3. GMM Csúcs (4. görbe)": [],
        "4. GMM Eltolt Csúcs (0.85x)": [],
        "5. Egészséges Agy Vége (GMM)": [],
        "6. GMM 3-4 görbe metszéspontja": [],
        "7. GMM 3-5 görbe metszéspontja": []
    }

    for patient in patient_folders[:10]:
        patient_img_path = os.path.join(images_dir, patient)
        patient_mask_path = os.path.join(masks_dir, patient)
        
        if not os.path.exists(patient_mask_path): continue
        img_files = sorted([f for f in os.listdir(patient_img_path) if f.lower().endswith(('.jpg', '.png'))])
        
        for f in img_files:
            img_path = os.path.join(patient_img_path, f)
            img = cv.imdecode(np.fromfile(img_path, np.uint8), cv.IMREAD_GRAYSCALE)
            
            base_name, ext = os.path.splitext(f)
            mask_path = os.path.join(patient_mask_path, base_name + "_mask" + ext)
            if not os.path.exists(mask_path): mask_path = os.path.join(patient_mask_path, f)
            if not os.path.exists(mask_path): continue
                
            mask = cv.imdecode(np.fromfile(mask_path, np.uint8), cv.IMREAD_GRAYSCALE)

            if img is None or mask is None: continue
            if len(img.shape) > 2: img = img[:,:,0]
            if len(mask.shape) > 2: mask = mask[:,:,0]

            gt_bool = mask > 127
            if gt_bool.sum() < 100: continue
            
            valid_pixels = img[img > 15].reshape(-1, 1).astype(np.float32)
            if valid_pixels.size < 1000: continue

            kmeans = KMeans(n_clusters=5, random_state=42, n_init=1)
            kmeans.fit(valid_pixels)
            km_centers = np.sort(kmeans.cluster_centers_.flatten())
            
            gmm = GaussianMixture(n_components=5, random_state=42, n_init=1)
            gmm.fit(valid_pixels)
            
            order = np.argsort(gmm.means_.flatten())
            gmm_means = gmm.means_.flatten()[order]
            gmm_stds = np.sqrt(gmm.covariances_.flatten()[order])
            gmm_weights = gmm.weights_.flatten()[order]

            thresholds = {
                "0. Alapvonal (90. Percentilis)": np.percentile(valid_pixels, 90),
                "0. Alapvonal (95. Percentilis)": np.percentile(valid_pixels, 95),
                "1. K-Means (3-4 klaszter köze)": (km_centers[2] + km_centers[3]) / 2.0,
                "2. K-Means (4-5 klaszter köze)": (km_centers[3] + km_centers[4]) / 2.0,
                "2,5. K-Means (3-5 klaszter köze)": (km_centers[2] + km_centers[4]) / 2.0,
                "3. GMM Csúcs (4. görbe)":       gmm_means[3],
                "4. GMM Eltolt Csúcs (0.85x)": gmm_means[3] * 0.85,
                "5. Egészséges Agy Vége (GMM)": gmm_means[2] + (2 * gmm_stds[2]),
                "6. GMM 3-4 görbe metszéspontja": find_intersection(
                    gmm_means[2], gmm_means[3], gmm_stds[2], gmm_stds[3], gmm_weights[2], gmm_weights[3]),
                "7. GMM 3-5 görbe metszéspontja": find_intersection(
                    gmm_means[2], gmm_means[4], gmm_stds[2], gmm_stds[4], gmm_weights[2], gmm_weights[4])
            }

            for name, thresh_val in thresholds.items():
                pred_mask = img >= thresh_val
                intersection = np.logical_and(pred_mask, gt_bool)
                if intersection.sum() > 0:
                    dice = calculate_dice(pred_mask, gt_bool)
                    results_dice[name].append(dice)

    sorted_results = sorted(results_dice.items(), key=lambda item: np.mean(item[1]) if item[1] else 0, reverse=True)
    
    print("\n--- Eredmények ---")
    for name, scores in sorted_results:
        if scores:
            print(f"{name:40} : {np.mean(scores) * 100:.2f}% Dice")

if __name__ == "__main__":
    IMAGES_DIR = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val"
    MASKS_DIR = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Labels\Val"
    run_threshold_experiments(IMAGES_DIR, MASKS_DIR)