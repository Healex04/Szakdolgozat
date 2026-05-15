import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def analyze_mri_histogram(image_path, n_clusters=4):
    img_array = np.fromfile(image_path, np.uint8)
    img = cv.imdecode(img_array, cv.IMREAD_GRAYSCALE)
    if img is None: return

    brain_pixels = img[img > 15].reshape(-1, 1)
    if len(brain_pixels) == 0: return

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(brain_pixels)
    centers = np.sort(kmeans.cluster_centers_.flatten())
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(brain_pixels)
    
    gmm_means = np.sort(gmm.means_.flatten())
    order = np.argsort(gmm.means_.flatten())
    gmm_covariances = gmm.covariances_[order].flatten()
    gmm_weights = gmm.weights_[order].flatten()
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Eredeti MRI Szelet")
    axs[0].axis('off')
    
    axs[1].hist(brain_pixels, bins=256, range=(1, 256), density=True, color='lightgray')
    for i, c in enumerate(centers):
        axs[1].axvline(x=c, color=plt.cm.jet(i / n_clusters), linestyle='--', linewidth=2, label=f'K-Means {i+1} ({int(c)})')
    axs[1].set_title(f"K-Means ({n_clusters} klaszter)")
    axs[1].legend()
    
    axs[2].hist(brain_pixels, bins=256, range=(1, 256), density=True, color='lightgray')
    x_axis = np.arange(1, 256).reshape(-1, 1)
    
    for i in range(n_clusters):
        mean = gmm_means[i]
        covar = gmm_covariances[i]
        weight = gmm_weights[i]
        y_axis = weight * (1.0 / (np.sqrt(2 * np.pi * covar))) * np.exp(-0.5 * ((x_axis - mean) ** 2) / covar)
        axs[2].plot(x_axis, y_axis, linewidth=2, color=plt.cm.jet(i / n_clusters), label=f'Gauss {i+1} ($\mu$={int(mean)})')
    
    axs[2].set_title(f"Gaussian Mixture Model ({n_clusters} görbe)")
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    TEST_IMAGE = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val\BRATS_011\084.jpg"
    analyze_mri_histogram(TEST_IMAGE, n_clusters=5)