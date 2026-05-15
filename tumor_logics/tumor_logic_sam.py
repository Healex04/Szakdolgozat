"""
YOLO + SAM 2 (Segment Anything Model 2) Video Detektor Modul.

Ez a modul a Meta SAM 2 videószegmentációs modelljét használja fel a tumorok 
3D-s követésére. A YOLO modell automatikusan generálja a 
kezdeti promptokat (pozitív pontokat) a daganat területén, amelyeket a SAM 2 
propagál a Z-tengely (szeletek) mentén, mint egy videó képkockáin.
"""

import os
import cv2 as cv
import numpy as np
import torch
import shutil
import sys
from ultralytics import YOLO

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SAM2_DIR = os.path.join(ROOT_DIR, "segment-anything-2")

if SAM2_DIR not in sys.path:
    sys.path.append(SAM2_DIR)

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print(f"[HIBA] Nem sikerült importálni a SAM 2-t a következő helyről: {SAM2_DIR}")

class TumorDetectorSAM:
    """
    SAM 2 videó szegmentáló és YOLO prompt generáló osztály.
    """
    def __init__(self, 
                 yolo_path="Models/YOLO_3medfilt.pt",
                 sam_checkpoint="segment-anything-2/sam2_hiera_large.pt",
                 sam_config_dir="segment-anything-2/sam2/configs/sam2",
                 sam_config_name="sam2_hiera_l.yaml"):
        
        self.yolo_model = None
        self.sam_predictor = None
        
        abs_yolo_path = os.path.join(ROOT_DIR, yolo_path)
        abs_sam_checkpoint = os.path.join(ROOT_DIR, sam_checkpoint)
        abs_sam_config_dir = os.path.join(ROOT_DIR, sam_config_dir)
        
        if os.path.exists(abs_yolo_path):
            self.yolo_model = YOLO(abs_yolo_path)
        else:
            print(f"[HIBA] Nincs YOLO modell: {abs_yolo_path}")

        if os.path.exists(abs_sam_checkpoint):
            if torch.cuda.is_available():
                device = "cuda"
                torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            else:
                device = "cpu"
                print("[FIGYELEM] Nincs GPU! A SAM 2 futtatása erősen lelassulhat.")

            try:
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
                
                with initialize_config_dir(config_dir=abs_sam_config_dir, version_base=None):
                    self.sam_predictor = build_sam2_video_predictor(sam_config_name, abs_sam_checkpoint, device=device)
            except Exception as e:
                print(f"[HIBA] SAM 2 konfigurációs hiba: {e}")
        else:
            print(f"[HIBA] Nincs SAM checkpoint: {abs_sam_checkpoint}")

    def get_yolo_prompts(self, image_paths):
        """
        Geometriai prompt pontok generálása a YOLO bounding boxok alapján.
        """
        prompts = {}
        
        viz_dir = os.path.join(ROOT_DIR, "yolo_prompt_viz_temp")
        os.makedirs(viz_dir, exist_ok=True)
        
        for idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path): continue
            
            results = self.yolo_model.predict(img_path, conf=0.2, verbose=False)
            
            frame_points = []
            frame_labels = []
            viz_img = cv.imread(img_path)
            has_tumor = False
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    if len(coords) >= 4:
                        has_tumor = True
                        x1, y1, x2, y2 = coords[:4]
                        
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        w, h = x2 - x1, y2 - y1
                        
                        offset_x = int(w * 0.15)
                        offset_y = int(h * 0.15)
                        
                        current_points = [
                            [cx, cy],              
                            [cx, cy - offset_y],   
                            [cx, cy + offset_y],   
                            [cx - offset_x, cy],   
                            [cx + offset_x, cy]    
                        ]
                        
                        for p in current_points:
                            frame_points.append(p)
                            frame_labels.append(1)

                        if viz_img is not None:
                            cv.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            for px, py in current_points:
                                cv.circle(viz_img, (px, py), 2, (0, 0, 255), -1)
            
            if has_tumor and viz_img is not None:
                cv.rectangle(viz_img, (2, 2), (160, 45), (0, 0, 0), -1)
                cv.rectangle(viz_img, (8, 10), (20, 20), (0, 255, 0), 2)
                cv.putText(viz_img, "YOLO Box", (28, 18), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv.circle(viz_img, (14, 34), 2, (0, 0, 255), -1)
                cv.putText(viz_img, "SAM 2 Prompts", (28, 38), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                base_name = os.path.basename(img_path)
                save_path = os.path.join(viz_dir, f"viz_{base_name}")
                cv.imwrite(save_path, viz_img)

            if frame_points:
                prompts[idx] = {
                    "points": np.array(frame_points, dtype=np.float32),
                    "labels": np.array(frame_labels, dtype=np.int32),
                    "obj_id": 1
                }
                
        return prompts
    
    def run_batch_processing(self, input_folder, output_folder, progress_callback=None):
        """SAM 2 batch feldolgozás futtatása a megadott mappán."""
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder, ignore_errors=True)
        os.makedirs(output_folder, exist_ok=True)

        files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        image_paths = [os.path.join(input_folder, f) for f in files]
        
        h, w = 240, 240
        if files:
            sample = cv.imread(image_paths[0], cv.IMREAD_GRAYSCALE)
            if sample is not None:
                h, w = sample.shape[:2]

        if self.sam_predictor is None or self.yolo_model is None:
            generated_masks = []
            for fname in files:
                save_path = os.path.join(output_folder, os.path.splitext(fname)[0] + ".png")
                cv.imwrite(save_path, np.zeros((h, w), dtype=np.uint8))
                generated_masks.append(save_path)
            return generated_masks

        if not files:
            return []

        if progress_callback: progress_callback(10)
        prompts = self.get_yolo_prompts(image_paths)
        
        if not prompts:
            generated_masks = []
            for fname in files:
                save_path = os.path.join(output_folder, os.path.splitext(fname)[0] + ".png")
                cv.imwrite(save_path, np.zeros((h, w), dtype=np.uint8))
                generated_masks.append(save_path)
            return generated_masks

        if progress_callback: progress_callback(30)
        
        try:
            inference_state = self.sam_predictor.init_state(video_path=input_folder)
            
            for frame_idx, data in prompts.items():
                self.sam_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=data['obj_id'],
                    points=data['points'],
                    labels=data['labels'],
                    clear_old_points=True
                )
            
            if progress_callback: progress_callback(50)

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    'mask': (out_mask_logits[0] > 0.0).cpu().numpy() 
                }
                
                if progress_callback:
                    percent = 50 + int((out_frame_idx / len(files)) * 50)
                    progress_callback(percent)

            generated_masks = []
            for idx, fname in enumerate(files):
                final_mask = np.zeros((h, w), dtype=np.uint8)

                if idx in video_segments:
                    mask_data = video_segments[idx]['mask']
                    if len(mask_data.shape) == 3:
                        mask_data = mask_data[0]
                    final_mask[mask_data] = 255

                save_path = os.path.join(output_folder, os.path.splitext(fname)[0] + ".png")
                cv.imwrite(save_path, final_mask)
                generated_masks.append(save_path)
            
            self.sam_predictor.reset_state(inference_state)
            
            if progress_callback: progress_callback(100)
            return generated_masks

        except Exception as e:
            print(f"[HIBA] SAM futtatás közben: {e}")
            generated_masks = []
            for fname in files:
                save_path = os.path.join(output_folder, os.path.splitext(fname)[0] + ".png")
                cv.imwrite(save_path, np.zeros((h, w), dtype=np.uint8))
                generated_masks.append(save_path)
            return generated_masks