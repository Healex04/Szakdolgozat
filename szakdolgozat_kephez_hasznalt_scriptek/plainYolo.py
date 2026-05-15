import cv2
import numpy as np
from ultralytics import YOLO

def generate_yolo_prediction_figure():
    model_path = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\6.Félév\____Szakdoga\Models\YOLO_3medfilt.pt" 
    image_path = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val\BRATS_011\104.jpg"
    output_path = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\6.Félév\____Szakdoga\tesztelés\yolo_valos_predikcio.png"
    
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.25, save=False)
    
    img = cv2.imread(image_path)
    if img is None: return
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0: return
        
    box_x_min, box_y_min, box_x_max, box_y_max = map(int, boxes[0])
    
    cv2.rectangle(img, (box_x_min, box_y_min), (box_x_max, box_y_max), (0, 255, 0), 1)
    center_x = (box_x_min + box_x_max) // 2
    center_y = (box_y_min + box_y_max) // 2
    cv2.circle(img, (center_x, center_y), 2, (0, 0, 255), -1)

    cv2.imwrite(output_path, img)
    print(f"[INFO] YOLO vizualizáció mentve: {output_path}")

if __name__ == "__main__":
    generate_yolo_prediction_figure()