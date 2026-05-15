# MRI Tumor Szegmentációs Keretrendszer (Szakdolgozat)

Ez a repozitórium egy hibrid (YOLO + GMM + 3D Region Growing), valamint a Meta SAM 2 modelljére épülő, agydaganatokat detektáló és szegmentáló szoftver teljes forráskódját tartalmazza. A rendszer grafikus felhasználói felülettel (GUI) rendelkezik, amely támogatja a 3D Multi-Planar Reconstruction (MPR) nézeteket és a valós idejű analitikát.

## ⚙️ Telepítés és Előkészületek

A projekt futtatásához az Anaconda (vagy Miniconda) környezetkezelő használatát javasoljuk. Mivel a GitHub tárhelykorlátozásai miatt a nagy méretű adathalmazok, modell súlyok és külső repók nincsenek feltöltve, az alábbi lépéseket kell követni:

### 1. A projekt klónozása
Nyiss egy terminált, és klónozd le ezt a repozitóriumot:
```bash
git clone [https://github.com/Healex04/Szakdolgozat.git](https://github.com/Healex04/Szakdolgozat.git)
cd Szakdolgozat
