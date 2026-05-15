import glob
import cv2
import numpy as np

def create_image_grid(input_folder, output_path, rows=2, cols=4):
    total_images = rows * cols
    files = glob.glob(f"{input_folder}/*.*")[:total_images]

    images = []
    for f in files:
        img_array = np.fromfile(f, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if len(images) == total_images:
        h, w = images[0].shape[:2]
        images = [cv2.resize(img, (w, h)) for img in images]
        
        row1 = np.hstack(images[0:cols])
        row2 = np.hstack(images[cols:total_images])
        final_grid = np.vstack([row1, row2])
        
        cv2.imencode('.png', final_grid)[1].tofile(output_path)
        print(f"[INFO] Sikeres mentés: {total_images} kép összefűzve ({rows}x{cols}). Útvonal: {output_path}")
    else:
        print(f"[HIBA] {total_images} helyett csak {len(images)} képet sikerült beolvasni.")

if __name__ == "__main__":
    FOLDER = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\6.Félév\_____szakdoga_irasos\egybevonnikepek\3"
    OUTPUT = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\6.Félév\_____szakdoga_irasos\egybevonnikepek\vegeredmeny_grid.png"
    create_image_grid(FOLDER, OUTPUT)