import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

def visualize_kmeans_clusters(image_path, mask_path):
    img = cv.imdecode(np.fromfile(image_path, np.uint8), cv.IMREAD_GRAYSCALE)
    gt_mask = cv.imdecode(np.fromfile(mask_path, np.uint8), cv.IMREAD_GRAYSCALE)
    
    if img is None or gt_mask is None: return
        
    if len(img.shape) > 2: img = img[:,:,0]
    if len(gt_mask.shape) > 2: gt_mask = gt_mask[:,:,0]

    valid_mask = img > 15
    valid_pixels = img[valid_mask].reshape(-1, 1).astype(np.float32)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=5)
    labels = kmeans.fit_predict(valid_pixels)
    centers = kmeans.cluster_centers_.flatten()
    
    order = np.argsort(centers)
    label_mapping = {order[i]: i for i in range(5)}
    sorted_labels = np.array([label_mapping[l] for l in labels])

    h, w = img.shape
    colored_img = np.zeros((h, w, 3), dtype=np.uint8)

    colors = [
        [15, 15, 15], [80, 80, 80], [140, 140, 140], 
        [200, 200, 200], [250, 250, 250]
    ]

    flat_colored = np.zeros((valid_pixels.shape[0], 3), dtype=np.uint8)
    for i in range(5):
        flat_colored[sorted_labels == i] = colors[i]
    
    colored_img[valid_mask] = flat_colored

    gt_bool = (gt_mask > 127).astype(np.uint8)
    contours, _ = cv.findContours(gt_bool, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(colored_img, contours, -1, (0, 255, 0), 1)

    original_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.drawContours(original_bgr, contours, -1, (0, 255, 0), 1)

    show_colored = True
    while True:
        display_img = colored_img if show_colored else original_bgr
        display_img = cv.resize(display_img, (w * 2, h * 2), interpolation=cv.INTER_NEAREST)
        
        cv.imshow("K-Means Vizualizacio (Szokoz: Valtas | ESC: Kilepes)", display_img)
        
        key = cv.waitKey(0)
        if key == 27: break
        elif key == 32: show_colored = not show_colored

    cv.destroyAllWindows()

if __name__ == "__main__":
    IMG = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val\BRATS_020\080.jpg"
    MASK = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Labels\Val\BRATS_020\080_mask.jpg"
    visualize_kmeans_clusters(IMG, MASK)