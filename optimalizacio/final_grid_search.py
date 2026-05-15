import os
import sys
import time
import itertools
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tumor_logics.tumor_logic_3d_gradient_gauss import TumorDetector3D_Gradient_gauss
from evaluation_logic import Evaluator

def parse_report_for_metrics(report_path):
    metrics = {'Dice': 0.0, 'Bal_Acc': 0.0, 'Sensitivity': 0.0, 'Specificity': 0.0}
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "3D Dice Score" in line: metrics['Dice'] = float(line.split()[-1])
                elif "3D Balanced Accuracy" in line: metrics['Bal_Acc'] = float(line.split()[-1])
                elif "3D Sensitivity" in line: metrics['Sensitivity'] = float(line.split()[-1])
                elif "3D Specificity" in line: metrics['Specificity'] = float(line.split()[-1])
    return metrics

def get_param_fingerprint(params_dict, grid_keys):
    return str({k: float(params_dict[k]) for k in grid_keys})

def run_grid_search():
    dataset_folder = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val"
    ground_truth_base = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Labels\Val"
    
    temp_output_dir = os.path.join(current_dir, "temp_grid_search_masks")
    eval_output_dir = os.path.join(current_dir, "temp_grid_search_evals")
    results_excel_path = os.path.join(current_dir, "final_grid_search_results.xlsx")

    all_patients = sorted([f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))])
    patients = all_patients[:10] 

    if not patients:
        print("[HIBA] Nem találtam beteg mappákat!")
        return

    param_grid = {
        'yolo_conf': [0.3, 0.4], 
        'alpha': [0.45, 0.47],
        'dist_penalty': [0.85, 0.88, 0.9],
        'base_thresh_mult': [0.85],
        'strict_boundary_mult': [0.27, 0.3, 0.33],
        'closing_iters': [8, 9],
        'opening_iters': [4, 5, 6],
        'dilation_iters': [3, 4]
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Kiválasztott betegek száma: {len(patients)}")
    print(f"Összes lehetséges kombináció: {len(combinations)}")
    
    all_results = []
    tested_params = set()
    
    if os.path.exists(results_excel_path):
        print(f"\n[INFO] Korábbi Excel fájl megtalálva: {results_excel_path}")
        old_df = pd.read_excel(results_excel_path)
        all_results = old_df.to_dict('records')
        
        for row in all_results:
            try:
                fingerprint = get_param_fingerprint(row, keys)
                tested_params.add(fingerprint)
            except KeyError:
                pass
                
        print(f"[INFO] {len(tested_params)} kombináció betöltve. Ezeket átugorjuk!\n")

    model_path_abs = os.path.join(parent_dir, "Models", "YOLO_3medfilt.pt")
    detector = TumorDetector3D_Gradient_gauss(model_path=model_path_abs)
    evaluator = Evaluator()

    for i, params in enumerate(combinations):
        fingerprint = get_param_fingerprint(params, keys)
        
        if fingerprint in tested_params:
            print(f"[{i+1}/{len(combinations)}] UGRÁS: {params}")
            continue
            
        print(f"\n[{i+1}/{len(combinations)}] Tesztelés: {params}")
        
        run_dices, run_sens, run_spec, run_acc, run_times = [], [], [], [], []
        
        for patient in patients:
            input_folder = os.path.join(dataset_folder, patient)
            output_folder = os.path.join(temp_output_dir, patient)
            gt_patient_folder = os.path.join(ground_truth_base, patient)
            
            try:
                t0 = time.time()
                detector.run_batch_processing(input_folder, output_folder, **params)
                run_times.append(time.time() - t0)

                method_name = f"GS_{i}"
                success, save_dir, vol_dice = evaluator.run_evaluation(
                    generated_masks_folder=output_folder,
                    ground_truth_folder=gt_patient_folder, 
                    output_base_folder=eval_output_dir,
                    patient_name=patient,
                    method_name=method_name
                )
                
                if success:
                    report_path = os.path.join(save_dir, "report.txt")
                    metrics = parse_report_for_metrics(report_path)
                    run_dices.append(metrics['Dice'])
                    run_sens.append(metrics['Sensitivity'])
                    run_spec.append(metrics['Specificity'])
                    run_acc.append(metrics['Bal_Acc'])
                    
            except Exception as e:
                print(f"[HIBA] {patient} esetén: {e}")

        if run_dices:
            mean_time = np.mean(run_times)
            row = {
                **params, 
                'Mean_3D_Dice': np.mean(run_dices),
                'Mean_Sensitivity': np.mean(run_sens),
                'Mean_Specificity': np.mean(run_spec),
                'Mean_Bal_Acc': np.mean(run_acc),
                'Mean_Time_sec': mean_time
            }
            all_results.append(row)
            tested_params.add(fingerprint)
            
            df = pd.DataFrame(all_results)
            df = df.sort_values(by="Mean_3D_Dice", ascending=False)
            df.to_excel(results_excel_path, index=False)
            print(f" -> EREDMÉNY MENTVE. Idő/beteg: {mean_time:.2f}s | 3D Dice: {np.mean(run_dices):.4f}")

    print(f"\n[VÉGE] A folyamat befejeződött. Eredménytábla: {results_excel_path}")

if __name__ == "__main__":
    run_grid_search()