import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def analyze_grid_search(excel_filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, excel_filename)
    
    try:
        df = pd.read_excel(excel_path)
        print(f"[INFO] Betöltve {len(df)} futás eredménye a {excel_filename} fájlból.\n")
    except FileNotFoundError:
        print(f"[HIBA] Nem található a fájl: {excel_path}")
        return

    cols_to_drop = ['Mean_3D_Dice', 'Mean_Sensitivity', 'Mean_Specificity', 'Mean_Bal_Acc', 'Mean_Time_sec']
    
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df['Mean_3D_Dice']

    # --- Korrelációs analízis ---
    print("=== KORRELÁCIÓ ===")
    
    df_for_corr = pd.concat([X, y], axis=1)
    corr = df_for_corr.corr()['Mean_3D_Dice'].drop('Mean_3D_Dice').sort_values(ascending=False)
    
    for index, value in corr.items():
        print(f"{index:20}: {value:+.4f}")
    print("\n")

    # --- Feature Importance (Random Forest) ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    print("=== PARAMÉTEREK FONTOSSÁGA (Random Forest) ===")
    for index, value in importance.items():
        print(f"{index:20}: {value:.4f}")

    # --- Vizualizáció ---
    plt.figure(figsize=(9, 5))
    importance.plot(kind='bar', color='#2e7d32', edgecolor='black')
    plt.title("Hiperparaméterek fontossága a 3D Dice-koefficiensre", fontsize=14)
    plt.ylabel("Relatív fontosság (Feature Importance)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(current_dir, "RF_Fontossag.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[INFO] A grafikon elmentve: {save_path}")

    plt.show()

if __name__ == "__main__":
    analyze_grid_search("final_grid_search_results.xlsx")