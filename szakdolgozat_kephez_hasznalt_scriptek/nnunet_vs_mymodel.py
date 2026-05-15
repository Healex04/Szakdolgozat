import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import SimpleITK as sitk

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tumor_logics.tumor_logic_3d_gradient_gauss import TumorDetector3D_Gradient_gauss

os.environ['nnUNet_raw'] = r"C:\nnunet_data\nnUNet_raw"                   
os.environ['nnUNet_preprocessed'] = r"C:\nnunet_data\nnUNet_preprocessed" 
os.environ['nnUNet_results'] = r"C:\nnunet_data\nnUNet_results"

def dice_coef_3d(y_true_vol, y_pred_vol):
    y_true_bin = (y_true_vol > 0).astype(np.uint8)
    y_pred_bin = (y_pred_vol > 0).astype(np.uint8)
    intersection = np.sum(y_true_bin & y_pred_bin)
    volume_sum = np.sum(y_true_bin) + np.sum(y_pred_bin)
    if volume_sum == 0: return 1.0
    return (2. * intersection) / volume_sum

def evaluate_nnunet_nifti(nifti_path, gt_folder):
    img_sitk = sitk.ReadImage(nifti_path)
    pred_array = sitk.GetArrayFromImage(img_sitk)
    
    if pred_array.shape[1] == pred_array.shape[2]:
        pred_array = pred_array.transpose(0, 2, 1)

    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(('.png', '.jpg'))])
    gt_slices = []
    
    for gf in gt_files:
        img_path = os.path.join(gt_folder, gf)
        with open(img_path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        slice_img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
        slice_img = (slice_img > 127).astype(np.uint8)
        gt_slices.append(slice_img)
        
    gt_array = np.array(gt_slices)
    if pred_array.shape != gt_array.shape: return 0.0
    return dice_coef_3d(gt_array, pred_array)

def run_comprehensive_benchmark(patient_list, base_input_dir, gt_dir, safe_nn_in, yolo_model_p):
    detector = TumorDetector3D_Gradient_gauss(model_path=yolo_model_p)
    all_results = []
    
    temp_hybrid_output = r"C:\nnunet_data\benchmark_hybrid_out"
    safe_nn_out = r"C:\nnunet_data\benchmark_nn_out"
    
    for folder in [temp_hybrid_output, safe_nn_in, safe_nn_out]:
        os.makedirs(folder, exist_ok=True)

    print(f"{'Páciens':<12} | {'Módszer':<15} | {'Idő (mp)':<10} | {'Dice Score':<10}")
    print("-" * 65)

    for patient in patient_list:
        patient_input = os.path.join(base_input_dir, patient)
        patient_gt = os.path.join(gt_dir, patient)

        start_h = time.time()
        detector.run_batch_processing(patient_input, temp_hybrid_output)
        end_h = time.time() - start_h
        
        pred_files = sorted([f for f in os.listdir(temp_hybrid_output) if f.endswith('.png') and not f.startswith('GMM')])
        pred_vol, gt_vol = [], []
        
        for pf in pred_files:
            p_img = cv.imread(os.path.join(temp_hybrid_output, pf), cv.IMREAD_GRAYSCALE)
            gt_img = cv.imread(os.path.join(patient_gt, pf.replace('.png', '_mask.jpg')), cv.IMREAD_GRAYSCALE)
            if p_img is not None and gt_img is not None:
                pred_vol.append(p_img); gt_vol.append(gt_img)
                
        dice_h = dice_coef_3d(np.array(gt_vol), np.array(pred_vol))
        print(f"{patient:<12} | {'Saját Hibrid':<15} | {end_h:<10.2f} | {dice_h:<10.4f}")
        all_results.append({'Patient': patient, 'Method': 'Saját Hibrid', 'Time': end_h, 'Dice': dice_h})

        start_g = time.time()
        cmd_gpu = f"nnUNetv2_predict -d 999 -i {safe_nn_in} -o {safe_nn_out} -c 2d -f 0 -chk checkpoint_best.pth -device cuda --disable_tta"
        subprocess.run(cmd_gpu, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end_g = time.time() - start_g
        
        nifti_file = os.path.join(safe_nn_out, f"{patient}.nii.gz")
        dice_g = evaluate_nnunet_nifti(nifti_file, patient_gt) if os.path.exists(nifti_file) else 0.0
        print(f"{'':<12} | {'nnU-Net GPU':<15} | {end_g:<10.2f} | {dice_g:<10.4f}")
        all_results.append({'Patient': patient, 'Method': 'nnU-Net GPU', 'Time': end_g, 'Dice': dice_g})

        start_c = time.time()
        cmd_cpu = f"nnUNetv2_predict -d 999 -i {safe_nn_in} -o {safe_nn_out} -c 2d -f 0 -chk checkpoint_best.pth -device cpu --disable_tta"
        subprocess.run(cmd_cpu, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end_c = time.time() - start_c
        
        dice_c = evaluate_nnunet_nifti(nifti_file, patient_gt) if os.path.exists(nifti_file) else 0.0
        print(f"{'':<12} | {'nnU-Net CPU':<15} | {end_c:<10.2f} | {dice_c:<10.4f}")
        all_results.append({'Patient': patient, 'Method': 'nnU-Net CPU', 'Time': end_c, 'Dice': dice_c})
        print("-" * 65)

    return pd.DataFrame(all_results)

def visualize_benchmark(df):
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    methods = ['Saját Hibrid', 'nnU-Net GPU', 'nnU-Net CPU']
    
    dice_means = [df[df['Method'] == m]['Dice'].mean() for m in methods]
    ax1.bar(methods, dice_means, color=['#3498db', '#e74c3c', '#c0392b'], alpha=0.8, edgecolor='black')
    ax1.set_title('Átlagos Dice-koefficiens (Valós mérés)')
    ax1.set_ylim(0, 1.0); ax1.set_ylabel('Dice Score')
    for i, v in enumerate(dice_means): ax1.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold', fontsize=11)

    time_means = [df[df['Method'] == m]['Time'].mean() for m in methods]
    ax2.bar(methods, time_means, color=['#2ecc71', '#f1c40f', '#e67e22'], alpha=0.8, edgecolor='black')
    ax2.set_yscale('log'); ax2.set_title('Átlagos futási idő - Logaritmikus skála (Valós mérés)')
    ax2.set_ylabel('Másodperc (log)')
    for i, v in enumerate(time_means): ax2.text(i, v * 1.1, f"{v:.1f} s", ha='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_patients = ['BRATS_010', 'BRATS_011', 'BRATS_012', 'BRATS_018', 'BRATS_020', 'BRATS_028', 'BRATS_029', 'BRATS_032', 'BRATS_034', 'BRATS_041']
    yolo_model_path = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\6.Félév\____Szakdoga\Models\YOLO_3medfilt.pt" 
    base_input_folder = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val"          
    gt_folder = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Labels\Val"                   
    nnunet_nifti_folder = r"C:\nnunet_data\run_benchmark"  

    print("[INFO] Benchmark indítása...")
    results_df = run_comprehensive_benchmark(test_patients, base_input_folder, gt_folder, nnunet_nifti_folder, yolo_model_path)
    
    if not results_df.empty:
        visualize_benchmark(results_df)
        h_time = results_df[results_df['Method'] == 'Saját Hibrid']['Time'].mean()
        c_time = results_df[results_df['Method'] == 'nnU-Net CPU']['Time'].mean()
        if h_time > 0:
            print(f"\n[ÖSSZEGZÉS] A hibrid modelled átlagosan {c_time / h_time:.1f}x GYORSABB, mint az nnU-Net CPU-n!")