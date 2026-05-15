# MRI Tumor Szegmentációs Keretrendszer (Szakdolgozat)

Ez a repozitórium egy hibrid (YOLO + GMM + 3D Region Growing), valamint a Meta SAM 2 modelljére épülő, agydaganatokat detektáló és szegmentáló szoftver teljes forráskódját tartalmazza. A rendszer grafikus felhasználói felülettel (GUI) rendelkezik, amely támogatja a 3D Multi-Planar Reconstruction (MPR) nézeteket és a valós idejű analitikát.

## ⚙️ Telepítés és Előkészületek

A projekt futtatásához az Anaconda (vagy Miniconda) környezetkezelő használatát javasoljuk. Mivel a GitHub tárhelykorlátozásai miatt a nagy méretű adathalmazok, modell súlyok és külső repók nincsenek feltöltve, az alábbi lépéseket kell követni:

### 1. A projekt klónozása
Nyiss egy terminált, és klónozd le ezt a repozitóriumot:
```bash
git clone [https://github.com/Healex04/Szakdolgozat.git](https://github.com/Healex04/Szakdolgozat.git)
cd Szakdolgozat
2. A SAM 2 (Segment Anything Model 2) integrálása
A SAM 2 videószegmentációs funkciók működéséhez a Meta hivatalos repozitóriumát közvetlenül a projekt főkönyvtárába kell klónozni, pontosan segment-anything-2 néven:

Bash
git clone [https://github.com/facebookresearch/segment-anything-2.git](https://github.com/facebookresearch/segment-anything-2.git)
(A szükséges SAM 2 súlyokat, pl. sam2_hiera_large.pt, a letöltött repozitórium leírása alapján kell beszerezni, és a megfelelő könyvtárba helyezni.)

3. Hiányzó modell súlyok pótlása
Hozd létre a főkönyvtárban a Models mappát, és másold bele a betanított YOLO modellt (pl. YOLO_3medfilt.pt). (A .gitignore fájl biztonsági okokból szűri a .pt kiterjesztéseket, ezért ezeket manuálisan kell pótolni).

4. Conda környezet felépítése
A mellékelt environment.yml fájl alapján másodpercek alatt felépíthető a teljes függőségi háló (OpenCV, PyQt6, Ultralytics, scikit-learn, stb.):

Bash
conda env create -f environment.yml
conda activate <a_kornyezet_neve_az_yml_fajlbol>
🚀 Használat (Főprogram futtatása)
Ha a környezet aktív és a SAM 2 is a helyén van, a grafikus felhasználói felület (GUI) az alábbi paranccsal indítható el a főkönyvtárból:

Bash
python main.py
A szoftver használata:

A Fájl -> Mappa Megnyitása menüpontban tölts be egy páciens MRI szeleteit (2D PNG/JPG) tartalmazó mappát.

(Opcionális) Töltsd be az orvosi referenciamaszkokat a Címkék (GT) Betöltése opcióval az analitikához.

A bal oldali panelen válaszd ki a kívánt modellt (pl. YOLO + 3D Gradiens (Gauss) vagy SAM 2).

Kattints a Tumor Keresése gombra. A feldolgozás befejeztével a 3D nézegetőben (Axiális/Koronális/Szagittális) csúszkával navigálhatsz a szegmentált eredmények között.

📚 Fejlesztői Dokumentáció (Sphinx)
A forráskódhoz egy automatikusan generált, interaktív, a modulokat és osztályokat részletező Sphinx HTML dokumentáció tartozik.
A megtekintéséhez nyisd meg a kedvenc böngésződben az alábbi fájlt:
docs/_build/html/index.html

📂 Mappastruktúra és Extra Szkriptek
A fő funkciókon túl a repozitórium tartalmazza a kutatás során használt összes kiegészítő szkriptet is:

/tumor_logics: A különböző szegmentációs algoritmusokat (YOLO, GMM, GrabCut, 3D Region Growing, SAM 2) tartalmazó objektum-orientált Python modulok.

/optimalizacio: A paraméterek finomhangolását (Grid Search) és a state-of-the-art (nnU-Net) algoritmussal történő objektív összemérést végző benchmark szkriptek.

/teszteles_es_kiserletek: A kutatást megalapozó hipotézis-tesztelő kódok (pl. Intensity drift mérése, K-Means vizualizáció, matematikai küszöb-analízis).

/abrageneralo: A szakdolgozatban szereplő publikációs minőségű demonstrációs ábrák és grafikonok legenerálásáért felelős szkriptek.
