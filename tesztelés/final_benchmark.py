import os
import sys
import shutil
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
os.chdir(parent_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_registry import AVAILABLE_MODELS

IMAGES_DIR = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Images\Val"
MASKS_DIR = r"C:\Users\ASUS\Desktop\Tároló\___Egyetem\5.Félév\Kutatásmódszertan\BRATS_Labels\Val"
TEMP_PRED_DIR = os.path.join(current_dir, "temp_benchmark_preds")
NUM_PATIENTS_TO_TEST = 97
CSV_RESULTS_PATH = os.path.join(current_dir, "vegso_benchmark_eredmenyek.csv")

def calculate_3d_volumetric_dice(generated_folder, ground_truth_folder):
    gen_files = sorted([f for f in os.listdir(generated_folder) if f.lower().endswith('.png')])
    g_tp, g_fp, g_fn = 0, 0, 0
    processed_count = 0

    for fname in gen_files:
        gen_path = os.path.join(generated_folder, fname)
        base_name = os.path.splitext(fname)[0]
        possible_names = [base_name + "_mask", base_name, base_name.replace("_mask", "")]
        gt_path = None
        
        for name in possible_names:
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
                full_p = os.path.join(ground_truth_folder, name + ext)
                if os.path.exists(full_p):
                    gt_path = full_p
                    break
            if gt_path: break
                
        if not gt_path: continue
        pred_img = cv.imread(gen_path, cv.IMREAD_GRAYSCALE)
        true_img = cv.imread(gt_path, cv.IMREAD_GRAYSCALE)
        
        if pred_img is None or true_img is None: continue
        if pred_img.shape != true_img.shape:
            true_img = cv.resize(true_img, (pred_img.shape[1], pred_img.shape[0]), interpolation=cv.INTER_NEAREST)

        pred_bool = (pred_img > 0)
        true_bool = (true_img > 0)
        g_tp += np.sum(np.logical_and(pred_bool, true_bool))
        g_fp += np.sum(np.logical_and(pred_bool, ~true_bool))
        g_fn += np.sum(np.logical_and(~pred_bool, true_bool))
        processed_count += 1

    if processed_count == 0: return 0.0
    sum_dice_denom = 2.0 * g_tp + g_fp + g_fn
    vol_dice = (2.0 * g_tp) / sum_dice_denom if sum_dice_denom > 0 else 1.0
    return vol_dice * 100.0 

def load_or_create_csv():
    if os.path.exists(CSV_RESULTS_PATH):
        return pd.read_csv(CSV_RESULTS_PATH)
    return pd.DataFrame(columns=["Módszer", "Beteg_ID", "Dice Score (%)"])

def save_single_result(df, method, patient_id, score):
    new_row = pd.DataFrame([{"Módszer": method, "Beteg_ID": patient_id, "Dice Score (%)": score}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_RESULTS_PATH, index=False)
    return df

def clear_vram():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def run_benchmarks(mode):
    models_to_run = {}
    for name, data in AVAILABLE_MODELS.items():
        if mode == '1' and "SAM" not in name: models_to_run[name] = data
        elif mode == '2' and "SAM" in name: models_to_run[name] = data

    patient_folders = sorted([f for f in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, f))])[:NUM_PATIENTS_TO_TEST]
    df_results = load_or_create_csv()

    for p_idx, patient in enumerate(patient_folders):
        patient_id = f"P{p_idx+1:02d}"
        patient_img_path = os.path.join(IMAGES_DIR, patient)
        patient_mask_path = os.path.join(MASKS_DIR, patient)
        
        for method_name, (model_constructor, model_id) in models_to_run.items():
            clean_method = method_name.replace("YOLO + ", "")
            if not df_results[(df_results["Módszer"] == clean_method) & (df_results["Beteg_ID"] == patient_id)].empty:
                continue

            print(f"[{patient_id}] Feldolgozás: {clean_method}")
            method_out_dir = os.path.join(TEMP_PRED_DIR, model_id, patient)
            os.makedirs(method_out_dir, exist_ok=True)
            
            try:
                model_instance = model_constructor()
                model_instance.run_batch_processing(patient_img_path, method_out_dir)
                dice_score = calculate_3d_volumetric_dice(method_out_dir, patient_mask_path)
                df_results = save_single_result(df_results, clean_method, patient_id, dice_score)
            except Exception as e:
                print(f"[HIBA] {e}")
            finally:
                if 'model_instance' in locals(): del model_instance
                clear_vram()

    if os.path.exists(TEMP_PRED_DIR): shutil.rmtree(TEMP_PRED_DIR, ignore_errors=True)
    return df_results

def plot_publication_graphs(df):
    if df.empty: return
    sns.set_theme(style="whitegrid")
    palette = "viridis"

    method_order = sorted(df["Módszer"].unique().tolist())
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="Dice Score (%)", y="Módszer", data=df, palette=palette, showmeans=True, 
                order=method_order,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
    plt.title("Szegmentációs algoritmusok Dice-koefficiens eloszlása", fontsize=16)
    plt.xlim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "Eredmeny_1_Boxplot.png"), dpi=300)
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(20, 15), sharey=True)
    axes = axes.flatten()
    colors = sns.color_palette("husl", len(method_order))
    
    for i, method in enumerate(method_order):
        if i >= 9: break 
        ax = axes[i]
        m_data = df[df["Módszer"] == method]
        ax.bar(m_data["Beteg_ID"], m_data["Dice Score (%)"], color=colors[i], edgecolor='black')
        ax.axhline(m_data["Dice Score (%)"].mean(), color='red', linestyle='--', label=f'Átlag: {m_data["Dice Score (%)"].mean():.1f}%')
        ax.set_title(method, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc="lower right")
        
    for j in range(len(method_order), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "Eredmeny_2_Reszletes_Stabilitas.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(14, 8))
    stats = df.groupby('Módszer', sort=False)['Dice Score (%)'].agg(['mean', 'std', 'count']).reset_index()
    stats['Módszer'] = pd.Categorical(stats['Módszer'], categories=method_order, ordered=True)
    stats = stats.sort_values('Módszer').reset_index(drop=True)
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    stats['error'] = 2 * stats['sem']
    colors = sns.color_palette(palette, len(stats))

    for i, row in stats.iterrows():
        plt.hlines(i, row['mean'] - row['error'], row['mean'] + row['error'], colors=colors[i], lw=10, alpha=0.8)
        plt.vlines([row['mean'] - row['error'], row['mean'] + row['error']], i - 0.2, i + 0.2, colors='black', lw=2)
        plt.plot(row['mean'], i, 'o', color='white', markeredgecolor='black', markersize=10, markeredgewidth=2)
        plt.text(row['mean'], i - 0.3, f"μ={row['mean']:.1f}% (N={int(row['count'])})", 
                 ha='center', va='top', fontweight='bold', fontsize=10)

    plt.yticks(range(len(stats)), stats['Módszer'], fontsize=12)
    plt.xlabel("Dice Score (%)", fontsize=14)
    plt.title("Várható érték becslése (Mean ± 2*SEM)", fontsize=16, pad=15)
    plt.xlim(0, 105)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "Eredmeny_3_Statisztikai_Becsles.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("1 - ÖSSZES (SAM nélkül)\n2 - CSAK SAM 2\n3 - GRAFIKONOK GENERÁLÁSA")
    v = input("Válassz: ").strip()
    if v in ['1', '2']:
        df = run_benchmarks(v)
        plot_publication_graphs(df)
    elif v == '3':
        df = load_or_create_csv()
        if not df.empty:
            plot_publication_graphs(df)