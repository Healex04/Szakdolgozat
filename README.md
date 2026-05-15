# MRI Tumor Szegmentációs Keretrendszer (Szakdolgozat)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyQt6-orange.svg)
![AI](https://img.shields.io/badge/AI-YOLOv8%20%7C%20SAM2-green.svg)

Ez a repozitórium egy agydaganatok detektálására és szegmentálására szolgáló szoftver teljes forráskódját tartalmazza. A rendszer egy **hibrid megközelítést** alkalmaz: a lokalizációt YOLOv8 végzi, míg a pontos szegmentálást Gauss-keverékmodellek (GMM), 3D gradiens alapú régió-növelés (Region Growing) vagy a Meta-féle SAM 2 (Segment Anything Model 2) biztosítja.

---

## ⚙️ Telepítés és Előkészületek

A projekt futtatásához **Anaconda** vagy **Miniconda** használata javasolt. A nagy méretű adathalmazok és modell-súlyok a GitHub korlátai miatt nincsenek feltöltve, ezért azokat manuálisan kell pótolni.

### 1. A projekt klónozása
Nyiss egy terminált, és klónozd le a repozitóriumot:
```bash
git clone [https://github.com/Healex04/Szakdolgozat.git](https://github.com/Healex04/Szakdolgozat.git)
cd Szakdolgozat
```

### 2. A SAM 2 (Segment Anything Model 2) integrálása
A SAM 2 alapú szegmentációhoz a Meta hivatalos forráskódját **közvetlenül a projekt főkönyvtárába** kell klónozni:
```bash
git clone [https://github.com/facebookresearch/segment-anything-2.git](https://github.com/facebookresearch/segment-anything-2.git)
```
> **Megjegyzés:** A SAM 2 modell súlyait (pl. `sam2_hiera_large.pt`) töltsd le a hivatalos oldalról, és helyezd el a `segment-anything-2/checkpoints` mappába.

### 3. Modellek és Könyvtárak beállítása
1. Hozd létre a főkönyvtárban a `Models/` mappát.
2. Másold bele a betanított YOLOv8 súlyokat (pl. `YOLO_3medfilt.pt`).
3. Hozd létre a virtuális környezetet a mellékelt `environment.yml` fájlból:
```bash
conda env create -f environment.yml
conda activate <a_kornyezet_neve>
```

---

## 🚀 Használati Útmutató

A szoftver grafikus kezelőfelülete (GUI) az alábbi paranccsal indítható a főkönyvtárból:
```bash
python main.py
```

### A szoftver funkciói:
1. **Adatbetöltés:** A *Fájl -> Mappa Megnyitása* menüben tölts be egy MRI szeleteket tartalmazó mappát (PNG/JPG formátum).
2. **Referencia maszkok:** Ha rendelkezel Ground Truth adatokkal, töltsd be őket az összehasonlító statisztikákhoz.
3. **Modellválasztás:** A bal oldali panelen válassz a hibrid algoritmusok vagy a SAM 2 közül.
4. **3D MPR Nézet:** A szoftver automatikusan előállítja a koronális és szagittális metszeteket, amik között a csúszkákkal navigálhatsz.
5. **Eredmények:** A rendszer kiszámítja a daganat térfogatát és (referencia esetén) a Dice-koefficienst.

---

## 📂 Repozitórium Struktúra

A projekt moduláris felépítésű, különválasztva a logikát a vizualizációtól:

* 📂 **`tumor_logics/`**: Az összes szegmentációs algoritmus (YOLO, GMM, 3D Region Growing, GrabCut, SAM2) forráskódja.
* 📂 **`optimalizacio/`**: Hiperparaméter-optimalizáló (Grid Search) szkriptek és az nnU-Net benchmark mérések.
* 📂 **`docs/`**: A forráskódhoz generált **Sphinx** dokumentáció. (Megnyitás: `docs/_build/html/index.html`).
* 📂 **`teszteles_es_kiserletek/`**: A kutatást megalapozó mérések (Intensity drift analízis, statisztikai tesztek).
* 📄 **`main.py`**: A PyQt6 alapú grafikus keretrendszer és vezérlőlogika.

---

## 📚 Dokumentáció és Fejlesztés

A kód részletes, metódus-szintű leírását a generált Sphinx dokumentáció tartalmazza. 
Új dokumentáció generálásához használd a következő parancsot a `docs/` mappában Windowson:
```bash
.\make.bat html
```

---

## 🎓 Szerző
**Alex** - Programtervező Informatikus hallgató, Széchenyi István Egyetem.
**Téma:** Agydaganatok automatizált detektálása és szegmentálása mesterséges intelligencia segítségével.
