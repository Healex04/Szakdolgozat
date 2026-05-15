.. MRI Tumor Detektor documentation master file.

MRI Tumor Detektor Dokumentáció
===============================

Üdvözöljük az **MRI Tumor Detektor** projekt hivatalos fejlesztői dokumentációjában! 
Ez a szoftver egy komplex keretrendszert biztosít agyi daganatok 3D MRI felvételeken történő 
automatikus szegmentálásához és orvosi metrikák alapú kiértékeléséhez.

.. toctree::
   :maxdepth: 2
   :caption: Tartalomjegyzék:

Rendszerarchitektúra Áttekintés
===============================

Az alkalmazás moduláris felépítésű. Az alábbi folyamatábra azt szemlélteti, 
hogyan áramlik az adat (MRI képek) a grafikus felülettől a Mesterséges Intelligencia 
modelleken át egészen a statisztikai kiértékelésig:

.. graphviz::

   digraph RendszerArchitektura {
      // LR (balról jobbra) helyett TB (fentről lefelé) használata a kompaktabb nézetért
      rankdir=TB; 
      nodesep=0.8; // Csomópontok közötti vízszintes távolság
      ranksep=0.6; // Szintek közötti függőleges távolság
      
      node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled", fillcolor="#f9f9f9", color="#333333"];
      edge [fontname="Helvetica", fontsize=9, color="#666666"];

      // Csomópontok definiálása
      Input [label="MRI 3D Kötet\n(Képek & Mappák)", shape=folder, fillcolor="#fff3e0"];
      GUI [label="MRITumorApp\n(Felhasználói Felület)", fillcolor="#e1f5fe"];
      Registry [label="Model Registry\n(Dinamikus Betöltő)", fillcolor="#f3e5f5"];
      
      subgraph cluster_ai {
         label="Mesterséges Intelligencia Motor";
         style=dashed;
         color="#999999";
         // Ezeket egy sorba rendezzük a dobozon belül
         {rank=same; YOLO; GMM; RG}
         YOLO [label="YOLO\n(Lokalizáció)"];
         GMM [label="GMM & Sobel\n(Pixel Analízis)"];
         RG [label="3D Region Growing\n(Térfogat Növelés)"];
         
         YOLO -> GMM [label=" ROI Bounding Box"];
         GMM -> RG [label=" Dinamikus Küszöbök"];
      }

      Evaluator [label="Evaluator Modul\n(3D Metrikák)", fillcolor="#e8f5e9"];
      Plotter [label="Plotting Logic\n(Grafikonok)", fillcolor="#fce4ec"];
      Output [label="Riportok & Maszkok\n(.png, .txt)", shape=note, fillcolor="#fffde7"];

      // Szintek (Ranks) kényszerítése, hogy szép réteges legyen
      {rank=same; Evaluator; Plotter}

      // Fő folyamat nyilak
      Input -> GUI [label=" Betöltés a memóriába"];
      GUI -> Registry [label=" Kiválasztott modell"];
      Registry -> YOLO [label=" Példányosítás"];
      RG -> GUI [label=" Generált 3D Maszk"];
      
      GUI -> Evaluator [label=" Predikció + Ground Truth"];
      Evaluator -> Plotter [label=" Statisztikai adatok"];
      Plotter -> Output [label=" Grafikon mentése"];
      Evaluator -> Output [label=" Riport mentése"];
   }

A Felhasználói Felület (GUI)
============================

A központi vezérlőmodul, amely az aszinkron folyamatokat, a 3D megjelenítést 
és a felhasználói interakciókat kezeli.

.. automodule:: main
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: model_registry
   :members:
   :undoc-members:

Analitikai és Statisztikai Motor
================================

Ez a szekció tartalmazza a volumetrikus (3D) kiértékelésért és a statisztikai 
vizualizációkért felelős logikát.

.. automodule:: evaluation_logic
   :members:
   :undoc-members:

.. automodule:: plotting_logic
   :members:
   :undoc-members:

Tumor Szegmentáló AI Modellek
=============================

Itt találhatók a kutatás során kidolgozott különböző mélységű algoritmusok, 
a legegyszerűbb 2D megoldástól a legfejlettebb, GMM-alapú 3D gradiens vizesárokig.

Hibrid GMM és Gradiens Model (Zászlóshajó)
------------------------------------------
Ez a modell képviseli a projekt technológiai csúcsát, kombinálva a statisztikai 
pixel-klasszifikációt az organikus távolsági büntetéssel.

.. inheritance-diagram:: tumor_logics.tumor_logic_3d_gradient_gauss.TumorDetector3D_Gradient_gauss
   :parts: 1

.. automodule:: tumor_logics.tumor_logic_3d_gradient_gauss
   :members:
   :show-inheritance:

További 3D és Gradiens alapú modellek
-------------------------------------

.. automodule:: tumor_logics.tumor_logic_3d_gradient
   :members:

.. automodule:: tumor_logics.tumor_logic_3d_bbox
   :members:

.. automodule:: tumor_logics.tumor_logic_3D
   :members:

Klasszikus és AI alapú 2D/Video modellek
----------------------------------------

.. automodule:: tumor_logics.tumor_logic_grabcut
   :members:

.. automodule:: tumor_logics.tumor_logic_sam
   :members:

.. automodule:: tumor_logics.tumor_logic
   :members:

Tárgymutató és Keresés
======================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`