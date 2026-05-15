"""
Modell Katalógus (Registry Pattern).

Ez a modul felelős a mesterséges intelligencia és képfeldolgozó modellek 
dinamikus nyilvántartásáért. A főprogram (GUI) ebből a szótárból olvassa ki 
az elérhető módszereket, így új modell hozzáadásához elegendő csak ezt a fájlt bővíteni.
"""

from tumor_logics.tumor_logic import TumorDetector          
from tumor_logics.tumor_logic_sam import TumorDetectorSAM   
from tumor_logics.tumor_logic_grabcut import TumorDetectorGrabCut 
from tumor_logics.tumor_logic_3D import TumorDetector3D  
from tumor_logics.tumor_logic_3d_bbox import TumorDetector3D_BBoxEdge 
from tumor_logics.tumor_logic_3d_gradient import TumorDetector3D_Gradient
from tumor_logics.tumor_logic_3d_gradient_gauss import TumorDetector3D_Gradient_gauss

AVAILABLE_MODELS = {
    "1. YOLO + Region Growing (SIMPLE)": (TumorDetector, "SIMPLE"),
    "2. YOLO + SAM 2 (AI)": (TumorDetectorSAM, "SAM"),
    "3. YOLO + GrabCut (Sima)": (lambda: TumorDetectorGrabCut(use_region_growing=False), "GRABCUT_RECT"),
    "4. YOLO + RG + GrabCut (Kombinált)": (lambda: TumorDetectorGrabCut(use_region_growing=True), "GRABCUT_RG"),
    "5. YOLO + 3D Region Growing": (TumorDetector3D, "3D_RG"),
    "6. YOLO + 3D Külső agyszövet (Kísérlet)": (TumorDetector3D_BBoxEdge, "3D_BBOX_EDGE"),
    "7. YOLO + 3D Gradiens": (TumorDetector3D_Gradient, "3D_GRADIENT"),
    "8. YOLO + 3D Gradiens (Gauss)": (TumorDetector3D_Gradient_gauss, "3D_GRADIENT_GAUSS")
}