import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os

def plot_gmm_for_two_images(img_path1, img_path2, n_components=3):
    paths = [img_path1, img_path2]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    
    for i, path in enumerate(paths):
        if not os.path.exists(path): continue
            
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv.imdecode(img_array, cv.IMREAD_GRAYSCALE)
        pixels = img[img > 10].reshape(-1, 1)
        
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(pixels)
        x = np.linspace(10, 255, 256).reshape(-1, 1)
        
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Eredeti MRI metszet ({i+1}. kép)", fontsize=14)
        axes[i, 0].axis('off')
        
        axes[i, 1].hist(pixels, bins=50, density=True, alpha=0.5, color='gray', edgecolor='black', label='Képpontok eloszlása')
        colors = ['red', 'blue', 'green', 'orange']
        
        for j in range(n_components):
            weight = gmm.weights_[j]
            mean = gmm.means_[j, 0]
            covar = gmm.covariances_[j, 0, 0]
            comp_pdf = weight * (1.0 / (np.sqrt(2 * np.pi * covar))) * np.exp(-0.5 * ((x[:, 0] - mean) ** 2) / covar)
            axes[i, 1].plot(x, comp_pdf, '--', lw=2.5, color=colors[j], label=f'{j+1}. komponens')
            
        logprob = gmm.score_samples(x)
        pdf = np.exp(logprob)
        axes[i, 1].plot(x, pdf, color='black', lw=2, label='Összesített GMM modell')
        
        axes[i, 1].set_title(f"Intenzitás-hisztogram és GMM illesztés", fontsize=14)
        axes[i, 1].set_xlabel("Pixel intenzitás (0-255)", fontsize=12)
        axes[i, 1].set_ylabel("Sűrűség", fontsize=12)
        axes[i, 1].legend(loc='upper right')
        axes[i, 1].grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    KEP_1 = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val\BRATS_020\076.jpg"
    KEP_2 = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val\BRATS_011\094.jpg"
    plot_gmm_for_two_images(KEP_1, KEP_2, n_components=3)