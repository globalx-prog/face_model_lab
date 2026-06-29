# Effizienter und kuehler RCNN-Trainingslauf

Ziel: Faster R-CNN testen, ohne die GPU thermisch zu stark zu belasten. Fuer den finalen Versuch wurde ein hartes Limit von 105 Grad und eine maximale Laufzeit von 8 Stunden gesetzt.

## Empfohlene Einstellung

```bash
cd /home/clemi/projekte/MIM
PATH=/home/clemi/projekte/MIM/.texlive/2026/bin/x86_64-linux:$PATH \
/home/clemi/.venvs/MIM/bin/python face_model_lab/run_with_rocm_guard.py \
  --log model_results/thermal_fasterrcnn_mobile_red20_ep1_105c_8h_20260629.csv \
  --interval 15 \
  --max-junction 105 \
  --max-edge 105 \
  --max-seconds 28800 \
  -- \
  /home/clemi/.venvs/MIM/bin/python face_model_lab/step03_train_torchvision_detector.py \
    --kind fasterrcnn_mobile \
    --epochs 1 \
    --batch 2 \
    --reduction 20 \
    --workers 2 \
    --prefetch-factor 2 \
    --min-size 256 \
    --max-size 384 \
    --lr 0.0001 \
    --save-every 1
```

## Warum diese Parameter?

- `run_with_rocm_guard.py`: protokolliert Temperatur, Power, GPU- und VRAM-Nutzung und beendet den Lauf bei 105 Grad oder nach 8 Stunden.
- `fasterrcnn_mobile`: nutzt Faster R-CNN mit MobileNetV3-Large-FPN-Backbone. Das ist thermisch/zeitlich sinnvoller als ResNet50-FPN, verliert aber Recall.
- `batch=2`: stabilster Wert auf der AMD Radeon PRO W7800 48GB. Groessere Batches erhoehen bei variablen Bildgroessen den Overhead und waren nicht verlaesslich effizienter.
- `reduction=20`: nutzt 644 Trainingsbilder und erzeugt in diesem Setup einen vollstaendigen Checkpoint innerhalb des 8h-Fensters.
- `min-size=256`, `max-size=384`: begrenzt teure grosse Eingaben; im finalen MobileNetV3-FPN-Lauf max. 88 Grad Junction.
- `workers=2`, `prefetch-factor=2`: genug paralleles Laden ohne unnoetig viel CPU-/Speicher-Overhead.
- `amp` nicht setzen: Mixed Precision war hier kein verlaesslicher Vorteil.

## Gemessene Grenzen

| Variante | Beobachtung | Bewertung |
|---|---|---|
| ResNet50-FPN, Default-Resize | max. 86 Grad Junction, aber nach 6 Minuten erst 2/65 Red100-Batches | thermisch ok, viel zu langsam |
| ResNet50-FPN, `256/384` | max. 88 Grad Junction | kuehlste sinnvolle RCNN-Variante, aber fuer Red6/10 noch zu langsam |
| MobileNet-FPN, Default-Resize | 100 Grad Junction | vom Guard abgebrochen |
| MobileNetV3-FPN, `red10/10`, `256/384` | erster Batch deutete auf mehr als 9 Stunden pro Epoche | nicht sinnvoll, weil vor 8h kein Epoch-Checkpoint entsteht |
| MobileNetV3-FPN, `red20/1`, `256/384` | 12:56 Minuten, max. 88 Grad Junction, Mean Loss 0,642 | erfolgreich, aber nur 0,365 Recall und 620,6 ms/Bild |

## Einordnung

Faster R-CNN ist als Two-Stage-Detector in dieser Torchvision/ROCm-Umgebung der Effizienz-Engpass. Der MobileNetV3-Large-FPN-Backbone macht den Trainingslauf kuehl und zeitlich kontrollierbar, aber im direkten Vergleich bleiben FCOS und YOLOv8m ueberlegen: FCOS beim Recall, YOLOv8m bei der Video-Pipeline und Latenz.
