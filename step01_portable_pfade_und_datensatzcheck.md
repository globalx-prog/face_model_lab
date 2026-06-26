# Step 01: Portable Pfade und Datensatzcheck

Kurzreferenz zum Notebook `step001_pfade_und_datensatz_setup.ipynb`.

## Zweck

`step001` ist der portable Start vor `step01_training_lab.ipynb`.

Es setzt keine festen lokalen Pfade wie `/home/clemi/projekte/MIM`, sondern findet den Projektroot automatisch oder nutzt Umgebungsvariablen.

## Wichtige Variablen

- `FACE_LAB_ROOT`: optionaler Projektroot. Das ist der Ordner, der `face_model_lab/` enthaelt.
- `FACE_LAB_PYTHON`: optionaler Python-Interpreter fuer Trainingskommandos.
- `WIDER_FACE_SOURCE_DIR`: optionaler lokaler Ordner mit WIDER-FACE-ZIPs oder entpackten WIDER-FACE-Ordnern.

## Erwartete Struktur

```text
ROOT/
├── face_model_lab/
│   ├── step00_common.py
│   ├── step001_pfade_und_datensatz_setup.ipynb
│   └── step01_training_lab.ipynb
└── datasets/
    └── wider_face/
        ├── WIDER_train/WIDER_train/images/
        ├── WIDER_val/WIDER_val/images/
        └── wider_face_split/wider_face_split/
            ├── wider_face_train_bbx_gt.txt
            └── wider_face_val_bbx_gt.txt
```

## Ablauf

1. `step001_pfade_und_datensatz_setup.ipynb` oeffnen.
2. Setup-Zellen ausfuehren.
3. Falls der Datensatz fehlt, entweder:
   - `LOAD_DATASET = True` setzen und `SOURCE_DIR`/`WIDER_FACE_SOURCE_DIR` auf lokale Daten zeigen lassen, oder
   - WIDER FACE manuell unter `ROOT/datasets/wider_face/` ablegen.
4. Datensatzcheck erneut ausfuehren.
5. Erst danach `step01_training_lab.ipynb` fuer Trainingskommandos nutzen.

## Hinweise

- `step001` importiert bewusst nicht `step00_common.py`, damit der Pfad- und Datensatzcheck auch ohne installierte Trainingsbibliotheken wie `torch` oder `cv2` laufen kann.
- Fehlende WIDER-FACE-Daten erzeugen standardmaessig nur eine Warnung. Fuer harte Fehlerpruefung in `step001`: `STRICT_DATASET_CHECK = True`.
- Automatischer Download ist vorbereitet, aber deaktiviert. Trage stabile URLs ein und setze `DOWNLOAD_DATASET = True`, falls ein Online-Download gewuenscht ist.
