import os
import time
import shutil
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

def run_nifti_nnunet_benchmark():
    base_nnunet_dir = r"C:\nnunet_data"
    
    nnunet_input_dir = os.path.join(base_nnunet_dir, "temp_nnunet_in")
    nnunet_output_dir = os.path.join(base_nnunet_dir, "temp_nnunet_out")

    val_dataset_folder = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val"
    nifti_source_dir = r"C:\nnunet_data\nnUNet_raw\Dataset999_BrainTumour\imagesVal" 

    os.environ['nnUNet_raw'] = os.path.join(base_nnunet_dir, "nnUNet_raw")
    os.environ['nnUNet_preprocessed'] = os.path.join(base_nnunet_dir, "nnUNet_preprocessed")
    os.environ['nnUNet_results'] = os.path.join(base_nnunet_dir, "nnUNet_results")

    for d in [nnunet_input_dir, nnunet_output_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    all_patients = sorted([f for f in os.listdir(val_dataset_folder) if os.path.isdir(os.path.join(val_dataset_folder, f))])
    test_patients = all_patients

    print(f"[INFO] {len(test_patients)} beteg multimodális (4 csatornás) NIfTI fájljainak másolása...")
    for patient in test_patients:
        for i in range(4):
            modality_file = f"{patient}_000{i}.nii.gz"
            src = os.path.join(nifti_source_dir, modality_file)
            dst = os.path.join(nnunet_input_dir, modality_file)
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"[HIBA] Hiányzik a fájl: {src}")

    cmd = (f"nnUNetv2_predict -d 999 -i {nnunet_input_dir} -o {nnunet_output_dir} "
           f"-c 2d -f 0 -chk checkpoint_best.pth -device cuda --disable_tta")
    
    print(f"\n[INFO] nnU-Net Inferencia INDÍTÁSA")
    print(f"Parancs: {cmd}\n")
    
    start_time = time.time()
    exit_code = os.system(cmd) 
    end_time = time.time()
    total_time = end_time - start_time
    
    if exit_code != 0:
        print("\n[HIBA] Az nnU-Net folyamat hibával leállt!")
        return
        
    print(f"\n[INFO] nnU-Net inferencia befejeződött. Összes idő: {total_time:.2f} másodperc.")
    print(f"[INFO] Átlagos idő / beteg: {total_time / len(test_patients):.2f} másodperc.")
    print(f"[INFO] A prediktált maszkok (.nii.gz) kimentve: {nnunet_output_dir}")

if __name__ == "__main__":
    run_nifti_nnunet_benchmark()