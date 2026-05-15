"""
MRI Tumor Detektor - Fő Alkalmazás (Main)
=========================================

Ez a modul a projekt központi grafikus felhasználói felülete (GUI), 
amely PyQt6 keretrendszerre épül. 

Szoftverarchitektúra
--------------------
A rendszer szigorúan elválasztja a grafikus megjelenítést a matematikai logikától:
- **Modell Réteg:** A mesterséges intelligencia modellek (YOLO, SAM, GMM) a `tumor_logics` mappában találhatók.
- **Katalógus:** A `model_registry` modul dinamikusan tölti be ezeket a modelleket.
- **Nézet Réteg:** Ez a modul (`main.py`) végzi a 3D térfogatok memóriakezelését és MPR (Multi-Planar Reconstruction) megjelenítését.

3D Megjelenítés (MPR)
---------------------
A szoftver képes az Axiális (Z), Koronális (Y) és Szagittális (X) metszetek valós idejű
képzésére a betöltött 3D Numpy tömbök (Volume) átrendezésével (transpose, flip, resize).
"""

import sys
import os
import shutil
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QListWidget, QPushButton, 
                             QSlider, QFileDialog, QStatusBar, QSizePolicy,
                             QProgressDialog, QComboBox, QFrame, QTextEdit,
                             QMessageBox)
from PyQt6.QtGui import QPixmap, QImage, QCloseEvent, QAction
from PyQt6.QtCore import Qt

from model_registry import AVAILABLE_MODELS
from evaluation_logic import Evaluator

class MRITumorApp(QMainWindow):
    """
    A fő ablakot és a globális állapottérképet (State) kezelő osztály.

    Attributes:
        image_paths (list of str): A betöltött MRI szeletek fájlelérési útjai.
        volume_3d (numpy.ndarray): Az eredeti MRI felvételeket tartalmazó 3D (Z, Y, X) szürkeárnyalatos tömb.
        mask_3d (numpy.ndarray): A modell által prediktált bináris 3D maszk.
        gt_3d (numpy.ndarray): Az orvosi Ground Truth 3D maszk.
        current_folder_path (str): Az aktuális MRI forrásmappa.
        current_label_folder (str): Az aktuális Ground Truth mappa.
        base_temp_dir (str): Az átmeneti predikciók mentési helye.
        eval_results_dir (str): A generált analitikai riportok mentési helye.
        loaded_models (dict): A memóriába már betöltött AI modellek (példányok) gyorsítótára.
        current_view_idx (int): Az aktuális 3D nézet azonosítója (0: Axiális, 1: Koronális, 2: Szagittális).
    """
    def __init__(self):
        super().__init__()
        
        # --- 1. Állapotváltozók és Memória ---
        self.image_paths = [] 
        self.volume_3d = None 
        self.mask_3d = None   
        self.gt_3d = None     
        
        self.current_folder_path = None
        self.current_folder_name = None
        self.current_label_folder = None
        
        self.base_temp_dir = os.path.join(os.getcwd(), "temp_masks")
        self.eval_results_dir = os.path.join(os.getcwd(), "Evaluation_Results")
        
        self.loaded_models = {}
        self.evaluator = Evaluator()
        self.current_view_idx = 0
        
        # --- 2. GUI Inicializálás ---
        self.setWindowTitle("MRI Tumor Detektor - Széchenyi Egyetem")
        self.resize(1300, 850)
        self.set_dark_theme()
        
        self.init_menu_bar()
        self.init_ui_layout()

    # =========================================================================
    # FELHASZNÁLÓI FELÜLET (UI) ÉS STÍLUS
    # =========================================================================

    def set_dark_theme(self):
        """Modern, sötét tónusú stílus (CSS/QSS) beállítása."""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #d4d4d4; }
            QFrame#SidePanel { background-color: #252526; border-right: 1px solid #3c3c3c; border-left: 1px solid #3c3c3c; }
            QLabel { color: #cccccc; font-family: 'Segoe UI', sans-serif; }
            QLabel#ImageDisplay { background-color: #050505; border: 1px solid #333; border-radius: 4px; }
            
            QPushButton { background-color: #333; color: #eee; border: 1px solid #444; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #444; border: 1px solid #555; }
            QPushButton#Primary { background-color: #007acc; font-weight: bold; border: none; }
            QPushButton#Primary:hover { background-color: #0098ff; }
            
            QListWidget { background-color: #1e1e1e; color: #ffffff; border: none; font-size: 13px; outline: none; }
            QListWidget::item { color: #ffffff; padding: 4px; }
            QListWidget::item:selected { background-color: #37373d; color: #007acc; }
            
            QComboBox { background-color: #3c3c3c; color: white; border: 1px solid #555; padding: 4px; border-radius: 4px; }
            QTextEdit#Console { background-color: #1e1e1e; color: #4ec9b0; border: 1px solid #333; font-family: 'Consolas', monospace; font-size: 12px; }
            
            QMenuBar { background-color: #333; color: #ddd; }
            QMenuBar::item:selected { background-color: #444; }
            QMenu { background-color: #252526; color: white; border: 1px solid #333; }
            
            QSlider::groove:horizontal { background: #3c3c3c; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #007acc; width: 14px; margin: -4px 0; border-radius: 7px; }
        """)

    def init_menu_bar(self):
        """A felső gördülő menürendszer felépítése és az események (Action) bekötése."""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("Fájl")
        file_menu.addAction("Mappa Megnyitása", "Ctrl+O", self.open_folder)
        file_menu.addAction("Címkék (GT) Betöltése", "Ctrl+L", self.open_label_folder)
        
        view_menu = menubar.addMenu("Nézet")
        self.axial_act = QAction("Axiális (Z)", self, checkable=True, triggered=lambda: self.change_view(0))
        self.coronal_act = QAction("Koronális (Y)", self, checkable=True, triggered=lambda: self.change_view(1))
        self.sagittal_act = QAction("Szagittális (X)", self, checkable=True, triggered=lambda: self.change_view(2))
        self.axial_act.setChecked(True)
        view_menu.addActions([self.axial_act, self.coronal_act, self.sagittal_act])
        
        view_menu.addSeparator()
        self.overlay_act = QAction("Maszk Megjelenítése", self, checkable=True, triggered=self.refresh_display)
        self.overlay_act.setChecked(True)
        self.debug_act = QAction("Analitikai Színezés (TP/FP/FN)", self, checkable=True, triggered=self.refresh_display)
        view_menu.addActions([self.overlay_act, self.debug_act])

    def init_ui_layout(self):
        """A hárompaneles (Bal: Vezérlés, Közép: Nézegető, Jobb: Analitika) főablak elrendezésének felépítése."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- BAL PANEL (Eszköztár és Fájlok) ---
        left_panel = QFrame(); left_panel.setObjectName("SidePanel"); left_panel.setFixedWidth(270)
        left_layout = QVBoxLayout(left_panel)
        
        left_layout.addWidget(QLabel("AKTÍV MODELL", styleSheet="font-weight: bold; color: #569cd6;"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(AVAILABLE_MODELS.keys())
        left_layout.addWidget(self.model_selector)
        
        self.run_btn = QPushButton("Tumor Keresése (Run)"); self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self.run_detection); self.run_btn.setEnabled(False)
        left_layout.addWidget(self.run_btn)
        
        left_layout.addWidget(QLabel("MRI SZELETEK", styleSheet="margin-top: 20px; font-weight: bold; color: #569cd6;"))
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        left_layout.addWidget(self.file_list)

        # --- KÖZÉPSŐ PANEL (Megjelenítő) ---
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        
        self.image_display = QLabel("Válasszon egy forrás mappát a Fájl menüben..."); self.image_display.setObjectName("ImageDisplay")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_display.setMinimumSize(400, 400)
        
        nav_layout = QHBoxLayout()
        self.slice_slider = QSlider(Qt.Orientation.Horizontal); self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.on_slider_change)
        self.slice_label = QLabel("Index: -"); self.slice_label.setFixedWidth(100)
        self.mag_label = QLabel(""); self.mag_label.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        nav_layout.addWidget(self.slice_slider); nav_layout.addWidget(self.slice_label); self.mag_lbl_ref = QLabel(""); nav_layout.addWidget(self.mag_lbl_ref)
        nav_layout.addWidget(self.mag_label)
        
        center_layout.addWidget(self.image_display, stretch=1); center_layout.addLayout(nav_layout)

        # --- JOBB PANEL (Statisztika és Konzol) ---
        right_panel = QFrame(); right_panel.setObjectName("SidePanel"); right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)
        
        right_layout.addWidget(QLabel("JELMAGYARÁZAT", styleSheet="font-weight: bold; color: #569cd6;"))
        legend_layout = QVBoxLayout()
        legend_layout.addWidget(QLabel("<span style='color:#4caf50;'>■</span> Találat (True Positive)"))
        legend_layout.addWidget(QLabel("<span style='color:#f44336;'>■</span> Téves Jelzés (False Positive)"))
        legend_layout.addWidget(QLabel("<span style='color:#2196f3;'>■</span> Hiányzó Rész (False Negative)"))
        right_layout.addLayout(legend_layout)
        
        right_layout.addSpacing(30)
        right_layout.addWidget(QLabel("EREDMÉNYEK KONZOL", styleSheet="font-weight: bold; color: #569cd6;"))
        self.eval_btn = QPushButton("Kiértékelés Indítása")
        self.eval_btn.clicked.connect(self.run_evaluation); self.eval_btn.setEnabled(False)
        right_layout.addWidget(self.eval_btn)
        
        self.console = QTextEdit(); self.console.setObjectName("Console"); self.console.setReadOnly(True)
        self.console.setPlaceholderText("Az analitikai adatok itt jelennek meg...")
        right_layout.addWidget(self.console)

        main_layout.addWidget(left_panel); main_layout.addWidget(center_panel); main_layout.addWidget(right_panel)
        self.setStatusBar(QStatusBar())

    # =========================================================================
    # LOGIKA: BETÖLTÉS ÉS DETEKTÁLÁS
    # =========================================================================

    def open_folder(self):
        """Dialógusablakon keresztül betölti a páciens MRI képeit egy 3D NumPy tömbbe."""
        folder = QFileDialog.getExistingDirectory(self, "MRI Mappa Kiválasztása")
        if not folder: return
        self.current_folder_path = folder
        self.current_folder_name = os.path.basename(folder)
        self.image_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        self.file_list.clear()
        self.file_list.addItems([os.path.basename(p) for p in self.image_paths])
        
        if self.image_paths:
            self.volume_3d = np.array([cv2.imread(p, 0) for p in self.image_paths])
            self.mask_3d = self.gt_3d = None
            self.run_btn.setEnabled(True); self.change_view(0)

    def open_label_folder(self):
        """Dialógusablakon keresztül betölti a manuálisan annotált Ground Truth maszkokat."""
        if not self.current_folder_path:
            QMessageBox.warning(self, "Hiba", "Előbb egy MRI mappát kell megnyitni!")
            return
        label_folder = QFileDialog.getExistingDirectory(self, "Ground Truth Mappa Kiválasztása")
        if label_folder:
            self.current_label_folder = label_folder
            self.load_gt_into_memory()
            self.debug_act.setEnabled(True); self.refresh_display()

    def run_detection(self):
        """
        Példányosítja a kiválasztott AI modellt (Registry Pattern), majd elindítja a 
        futtatást (szegmentálást) a teljes 3D köteten egy aszinkron progress bar kíséretében.
        """
        self.mask_3d = None; self.refresh_display()
        model_name = self.model_selector.currentText()
        model_cls, model_code = AVAILABLE_MODELS[model_name]
        out_folder = os.path.join(self.base_temp_dir, f"{self.current_folder_name}_{model_code}")
        
        if model_name not in self.loaded_models: self.loaded_models[model_name] = model_cls()
        detector = self.loaded_models[model_name]

        progress = QProgressDialog("Számítás folyamatban...", "Mégse", 0, 100, self)
        progress.show()

        try:
            detector.run_batch_processing(self.current_folder_path, out_folder, progress.setValue)
            self.mag_label.setText(f"Mag (Z): {detector.last_best_z + 1}" if hasattr(detector, 'last_best_z') else "")
            self.load_masks_into_memory(out_folder)
            self.refresh_display(); self.eval_btn.setEnabled(True)
        except Exception as e:
            self.console.setText(f"HIBA a detektálás során: {e}")
        progress.close()

    def load_masks_into_memory(self, path):
        """A temp mappából betölti a generált predikciós maszkokat."""
        masks = []
        for p in self.image_paths:
            m_p = os.path.join(path, os.path.splitext(os.path.basename(p))[0] + ".png")
            img = cv2.imread(m_p, 0) if os.path.exists(m_p) else None
            masks.append(cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] if img is not None else np.zeros_like(self.volume_3d[0]))
        self.mask_3d = np.array(masks)

    def load_gt_into_memory(self):
        """A Ground Truth (referencia) maszkok betöltése a memóriába analitikai célokra."""
        gts = []
        for p in self.image_paths:
            fname = os.path.basename(p); b, e = os.path.splitext(fname)
            gp = os.path.join(self.current_label_folder, b + "_mask" + e)
            if not os.path.exists(gp): gp = os.path.join(self.current_label_folder, fname)
            img = cv2.imread(gp, 0) if os.path.exists(gp) else None
            gts.append(cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] if img is not None else np.zeros_like(self.volume_3d[0]))
        self.gt_3d = np.array(gts)

    def run_evaluation(self):
        """Összeköti az `Evaluator` modult a GUI-val, és megjeleníti a 3D metrikákat a konzolon."""
        self.console.setText("Analitika futtatása..."); QApplication.processEvents()
        _, code = AVAILABLE_MODELS[self.model_selector.currentText()]
        try:
            res = self.evaluator.run_evaluation(os.path.join(self.base_temp_dir, f"{self.current_folder_name}_{code}"), 
                                               self.current_label_folder, self.eval_results_dir, self.current_folder_name, code)
            if res[0]:
                with open(os.path.join(res[1], "report.txt"), "r", encoding='utf-8') as f: self.console.setText(f.read())
        except Exception as e: self.console.setText(f"Hiba a kiértékelés során: {e}")

    # =========================================================================
    # 3D MPR ÉS MEGJELENÍTÉS
    # =========================================================================

    def change_view(self, i):
        """Nézetváltás a 3D térben (Axiális, Koronális, Szagittális)."""
        self.axial_act.setChecked(i==0); self.coronal_act.setChecked(i==1); self.sagittal_act.setChecked(i==2)
        self.current_view_idx = i
        if self.volume_3d is None: return
        Z, Y, X = self.volume_3d.shape[:3]
        max_idx = (Z-1) if i==0 else (X-1 if i==1 else Y-1)
        self.slice_slider.setEnabled(True); self.slice_slider.setRange(0, max_idx); self.slice_slider.setValue(max_idx//2); self.refresh_display()

    def update_image_display(self, idx):
        """
        Az aktuálisan kiválasztott 3D metszet (MPR) kiszámítása és renderelése a Qt felületen.
        """
        if self.volume_3d is None: return
        try:
            v_idx = self.current_view_idx
            self.slice_label.setText(f"Szelet: {idx+1}/{self.slice_slider.maximum()+1}")
            
            # Szelet vágás
            if v_idx == 0: b, m, g = self.volume_3d[idx], (self.mask_3d[idx] if self.mask_3d is not None else None), (self.gt_3d[idx] if self.gt_3d is not None else None)
            elif v_idx == 1: b, m, g = self.volume_3d[:,:,idx], (self.mask_3d[:,:,idx] if self.mask_3d is not None else None), (self.gt_3d[:,:,idx] if self.gt_3d is not None else None)
            else: b, m, g = self.volume_3d[:,idx,:], (self.mask_3d[:,idx,:] if self.mask_3d is not None else None), (self.gt_3d[:,idx,:] if self.gt_3d is not None else None)

            # Forgatás és Nyújtás (A voxelmélység kompenzálása)
            if v_idx == 0: b = cv2.transpose(b); m = cv2.transpose(m) if m is not None else None; g = cv2.transpose(g) if g is not None else None
            else: 
                b = cv2.resize(cv2.flip(b, 0), (b.shape[1], int(b.shape[0]*2)))
                m = cv2.resize(cv2.flip(m, 0), (m.shape[1], int(m.shape[0]*2)), interpolation=cv2.INTER_NEAREST) if m is not None else None
                g = cv2.resize(cv2.flip(g, 0), (g.shape[1], int(g.shape[0]*2)), interpolation=cv2.INTER_NEAREST) if g is not None else None

            disp = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
            
            # Analitikai színezés alkalmazása (TP: Zöld, FP: Piros, FN: Kék)
            if self.debug_act.isChecked() and m is not None and g is not None:
                P, T = (m > 127), (g > 127)
                ov = disp.copy(); ov[P & T] = [76, 175, 80]; ov[P & ~T] = [244, 67, 54]; ov[~P & T] = [33, 150, 243]
                cv2.addWeighted(ov, 0.4, disp, 0.6, 0, disp)
            elif self.overlay_act.isChecked() and m is not None:
                ov = disp.copy(); ov[m > 127] = [255, 0, 0]
                cv2.addWeighted(ov, 0.4, disp, 0.6, 0, disp)

            # Konvertálás és renderelés QLabel-en
            pix = QPixmap.fromImage(QImage(disp.data, disp.shape[1], disp.shape[0], 3*disp.shape[1], QImage.Format.Format_RGB888))
            self.image_display.setPixmap(pix.scaled(self.image_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except: pass

    def on_slider_change(self, v): 
        """Szinkronizálja a fájllistát a csúszkával (Axiális nézet esetén)."""
        if self.current_view_idx == 0: self.file_list.setCurrentRow(v)
        self.update_image_display(v)
        
    def on_file_selected(self, r): 
        """Szinkronizálja a csúszkát a fájllistával (Axiális nézet esetén)."""
        if self.current_view_idx == 0: self.slice_slider.setValue(r)
        
    def refresh_display(self): 
        """Aktualizálja a képernyőt külső beavatkozások (pl. nézetváltás) után."""
        self.update_image_display(self.slice_slider.value())
        
    def resizeEvent(self, e): 
        """Kezeli az ablak átméretezését, hogy a kép kitöltse a szabad teret."""
        self.refresh_display(); super().resizeEvent(e)
        
    def closeEvent(self, e): 
        """Az alkalmazás bezárásakor törli a generált átmeneti fájlokat."""
        if os.path.exists(self.base_temp_dir): shutil.rmtree(self.base_temp_dir)
        e.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv); w = MRITumorApp(); w.show(); sys.exit(app.exec())