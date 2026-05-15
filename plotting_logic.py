"""
Vizualizációs és Plotting Modul.

Ez a modul felelős a kiértékelés során keletkező statisztikai adatok 
(Confusion Matrix, Dice Histogram) grafikus megjelenítéséért és mentéséért.
A generált ábrák a dokumentációban és az analitikai riportokban kerülnek felhasználásra.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(tp, tn, fp, fn, save_path, title="Confusion Matrix"):
    """
    Legenerálja és elmenti a 2x2-es tévesztésmátrix (Confusion Matrix) hőtérképét.

    Args:
        tp (int): True Positive (Helyes találat) pixelek száma globálisan.
        tn (int): True Negative (Helyes háttér) pixelek száma globálisan.
        fp (int): False Positive (Téves riasztás) pixelek száma globálisan.
        fn (int): False Negative (Hiányzó találat) pixelek száma globálisan.
        save_path (str): A kimeneti PNG fájl teljes elérési útja.
        title (str, optional): A grafikon címe. Alapértelmezett: "Confusion Matrix".
    """
    matrix = np.array([[tp, fn], 
                       [fp, tn]])
    
    group_names = ['True Pos', 'False Neg', 'False Pos', 'True Neg']
    
    total = np.sum(matrix)
    if total == 0: total = 1 
    
    group_counts = ["{0:0.0f}".format(value) for value in matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in matrix.flatten()/total]
    
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Tumor (Pred)', 'Háttér (Pred)'],
                yticklabels=['Tumor (Valós)', 'Háttér (Valós)'])
    
    plt.title(title)
    plt.ylabel('Valós érték')
    plt.xlabel('Prediktált érték')
    
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
    except Exception as e:
        print(f"[HIBA] Plot mentése sikertelen: {e}")
    finally:
        plt.close()

def plot_dice_histogram(dice_scores, save_path):
    """
    Legenerál és elment egy hisztogramot az egyedi szeleteken elért Dice pontszámokból.

    A grafikon megmutatja a teljesítmény eloszlását az MRI köteten belül 

    Args:
        dice_scores (list of float): A 2D szeletenként kiszámolt Dice értékek listája.
        save_path (str): A kimeneti PNG fájl elérési útja.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(dice_scores, bins=20, color='skyblue', edgecolor='black')
    plt.title('Dice Score Eloszlás')
    plt.xlabel('Dice Score (0.0 - 1.0)')
    plt.ylabel('Képek száma')
    plt.grid(axis='y', alpha=0.5)
    
    try:
        plt.savefig(save_path, bbox_inches='tight')
    except Exception as e: 
        print(f"[HIBA] Hisztogram mentése sikertelen: {e}")
    finally:
        plt.close()