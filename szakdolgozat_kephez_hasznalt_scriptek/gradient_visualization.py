import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_academic_gradients(img_path):
    if not os.path.exists(img_path): return

    img_array = np.fromfile(img_path, dtype=np.uint8)
    original_img = cv.imdecode(img_array, cv.IMREAD_GRAYSCALE)
    if original_img is None: return
    
    blurred_img = cv.GaussianBlur(original_img, (5, 5), 0)
    sobel_x = cv.Sobel(blurred_img, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(blurred_img, cv.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = cv.magnitude(sobel_x, sobel_y)
    gradient_magnitude = np.uint8(cv.normalize(gradient_magnitude, None, 0, 255, cv.NORM_MINMAX))
    sobel_x_abs = np.uint8(np.absolute(sobel_x))
    sobel_y_abs = np.uint8(np.absolute(sobel_y))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    axes[0, 0].imshow(original_img, cmap='gray'); axes[0, 0].set_title("1. Eredeti felvétel", fontsize=14); axes[0, 0].axis('off')
    axes[0, 1].imshow(blurred_img, cmap='gray'); axes[0, 1].set_title("2. Gauss-szűrt kép", fontsize=14); axes[0, 1].axis('off')
    axes[1, 0].imshow(sobel_x_abs, cmap='gray'); axes[1, 0].set_title("3. Sobel X (vízszintes)", fontsize=14); axes[1, 0].axis('off')
    axes[1, 1].imshow(sobel_y_abs, cmap='gray'); axes[1, 1].set_title("4. Sobel Y (függőleges)", fontsize=14); axes[1, 1].axis('off')
    axes[0, 2].imshow(gradient_magnitude, cmap='gray_r'); axes[0, 2].set_title("5. Gradiens magnitúdó", fontsize=14); axes[0, 2].axis('off')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    IMG_PATH = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val\BRATS_011\094.jpg"
    visualize_academic_gradients(IMG_PATH)