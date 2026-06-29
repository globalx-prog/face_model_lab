# Faster R-CNN im Face Model Lab

Dieses Dokument erklaert Faster R-CNN, die konkrete Umsetzung im Projekt und die Unterschiede zu den anderen getesteten Modelltypen.

## Kurzidee von Faster R-CNN

Faster R-CNN ist ein Two-Stage-Detector. Das Modell arbeitet also in zwei groben Schritten:

1. **Region Proposal Network (RPN)**: Das Modell sucht zuerst Bildbereiche, in denen ein Objekt liegen koennte. Im Projekt ist das Objekt immer `face`.
2. **ROI Head**: Die vorgeschlagenen Regionen werden ausgeschnitten/gebuendelt, klassifiziert und als Bounding Boxes verfeinert.

Der Vorteil dieser Architektur ist eine saubere, oft robuste Lokalisierung. Der Nachteil ist der hohe Overhead: Pro Bild entstehen viele Vorschlagsregionen, und die ROI-Verarbeitung ist bei variablen Bildgroessen und vielen kleinen Gesichtern teuer. Genau das war im Projekt der Engpass.

## Umsetzung im Projekt

Das aktuelle Training laeuft ueber:

```text
face_model_lab/step03_train_torchvision_detector.py
```

Die Modelle werden in `step06_evaluate_models.py` gebaut und dort auch fuer die Evaluation wiederhergestellt:

- `build_fasterrcnn(...)`: `torchvision.models.detection.fasterrcnn_resnet50_fpn`
- `build_fasterrcnn_mobile(...)`: `torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn`

Fuer die Face-Detection wird der Klassifikationskopf ersetzt:

```python
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

`num_classes=2` bedeutet:

- Klasse 0: Hintergrund
- Klasse 1: Gesicht

Die WIDER-FACE-Annotationen werden fuer Torchvision in COCO-artige Annotationen umgewandelt. Das passiert ueber:

```text
step04_train_fasterrcnn.py
```

Wichtige Dataset-Klassen/Funktionen:

- `convert_wider_to_coco(...)`
- `CocoFaceDataset`
- `collate_fn`

## Backbone

Im Projekt wurden zwei RCNN-Backbones genutzt:

| Variante | Backbone | Bewertung im Projekt |
|---|---|---|
| Faster R-CNN ResNet50-FPN | ResNet50 + Feature Pyramid Network | Qualitativ bessere Two-Stage-Baseline, aber sehr langsam im Training |
| Faster R-CNN MobileNetV3-FPN | MobileNetV3-Large + Feature Pyramid Network | deutlich kuehler und kleiner, aber schwaecherer Recall |

Der finale 105-Grad-/8h-Testlauf nutzte **MobileNetV3-Large-FPN**. Das war bewusst ein leichterer Backbone, weil ResNet50-FPN fuer einen vergleichbaren langen Lauf zu langsam war.

## Optimizer und Scheduler

Im aktuellen Torchvision-Trainingsskript wird fuer Faster R-CNN, FCOS und RetinaNet dieselbe Optimizer-/Scheduler-Logik verwendet.

### Optimizer

```python
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr,
    weight_decay=1e-4,
)
```

Die Standard-Lernrate ist:

```text
lr = 0.0001
```

Warum AdamW?

- stabil bei Fine-Tuning vortrainierter Detektoren
- entkoppelte Weight Decay Regularisierung
- weniger manuelles Momentum-Tuning als bei SGD

### Scheduler

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(args.epochs, 1),
)
```

Der Scheduler senkt die Lernrate ueber die Epochen nach einer Cosine-Kurve ab. Bei kurzen Runs ist der Effekt klein, bei laengeren Runs stabilisiert er das Fine-Tuning.

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Das begrenzt starke Gradienten. Bei Detection-Batches mit sehr vielen oder sehr kleinen Gesichtern kann das Training sonst unruhig werden.

### Mixed Precision

Das Skript hat eine `--amp`-Option. Im Projekt wurde AMP fuer Torchvision/ROCm aber nicht als verlaesslicher Gewinn bewertet. Deshalb wurden die stabilen RCNN-Laeufe ohne `--amp` dokumentiert.

## Guard fuer Temperatur und Laufzeit

Der Guard liegt in:

```text
face_model_lab/run_with_rocm_guard.py
```

Er protokolliert ROCm-Messwerte und beendet den Lauf bei Temperatur- oder Zeitlimit.

Wichtige Optionen:

```text
--max-junction 105
--max-edge 105
--max-seconds 28800
--interval 15
--log <csv-datei>
```

Der finale RCNN-MobileNetV3-FPN-Lauf:

- Training: `red20`, 1 Epoche, 644 Bilder
- Batch: 2
- Resize: `min-size=256`, `max-size=384`
- Laufzeit: 12:56 Minuten
- max. Junction: 88 Grad
- Mean Loss: 0,642

Der spaetere starke RCNN-ResNet50-FPN-Vergleichslauf:

- Training: `red6`, 10 Epochen, 2.147 Bilder
- Batch: 2
- Resize: `min-size=640`, `max-size=640`
- Backbone: `ResNet50-FPN`
- Mean Loss: 0,583 -> 0,185
- Ergebnis in der 500-Bilder-Evaluation: Recall 0,603 bei 49,2 ms/Bild

## Vergleich mit den Gewinnern

Finale Evaluation auf 500 Validierungsbildern, 5.015 Ground-Truth-Gesichtern, IoU 0,4 und Confidence 0,25:

| Modell | Typ | Backbone / Kern | Recall | ms/Bild | Einordnung |
|---|---|---|---:|---:|---|
| FCOS red6 ep10 | One-Stage, anchor-free | ResNet50-FPN | 0,673 | 34,3 | bester Recall |
| YOLOv8m red6 ep10 | One-Stage, YOLO | YOLOv8m CSP/Ultralytics | 0,455 | 14,6 | schnellster und beste Video-Pipeline-Anbindung |
| Faster R-CNN ResNet50-FPN red6 ep10 | Two-Stage | ResNet50-FPN | 0,603 | 49,2 | starker Two-Stage-Vergleich, aber langsamer als YOLO/FCOS |

## Vergleich der Modelltypen

### Faster R-CNN

- Two-Stage-Detector
- RPN erzeugt Vorschlagsregionen, ROI Head klassifiziert/verfeinert
- starke klassische Baseline fuer Box-Qualitaet
- im Projekt hoher Overhead durch variable Bildgroessen, RPN und ROI Heads
- ResNet50-FPN war im finalen Vergleich deutlich staerker als MobileNetV3-FPN

### FCOS

- One-Stage-Detector
- anchor-free: keine Anchor-Box-Konfiguration wie bei RetinaNet
- im Projekt bester Recall
- guter Kandidat, wenn Datenschutz-Abdeckung wichtiger ist als Pipeline-Komfort
- weniger direkt in die bestehende Video-/Tracking-Pipeline integriert als YOLO

### YOLOv8m

- One-Stage-Detector
- sehr gute praktische Pipeline-Unterstuetzung durch Ultralytics
- schnellste Inferenz im finalen Vergleich
- Tracking und Videoverarbeitung sind direkt nutzbar
- Recall niedriger als FCOS, aber beste operative Wahl fuer die Blur-Pipeline

### RetinaNet

- One-Stage-Detector mit Anchor Boxes
- nutzt Focal Loss, um viele einfache Negativbeispiele weniger stark zu gewichten
- sinnvoll als Vergleichsmodell fuer dichte Object-Detection
- fiel im Projekt wegen niedrigerem Recall gegen FCOS/YOLO ab

### RT-DETR

- Transformer-basierter End-to-End-Detector
- theoretisch interessant als YOLO-Alternative
- im Projekt nicht der Gewinner der finalen Face-Detection-Strecke
- meist schwerer und weniger direkt passend fuer die einfache Video-Pipeline

## Fazit

Faster R-CNN ist im Projekt vor allem eine Erklaer- und Vergleichsbasis fuer klassische Two-Stage-Detection. Die Architektur ist nachvollziehbar und fuer Lokalisierung stark, aber die praktische Effizienz war schwach. Fuer das konkrete Ziel `Gesichter in Videos anonymisieren` sind die Gewinner klarer:

- **FCOS**, wenn maximaler Recall priorisiert wird.
- **YOLOv8m**, wenn Video-Pipeline, Geschwindigkeit und Bedienbarkeit priorisiert werden.
- **Faster R-CNN**, wenn eine klassische Two-Stage-Baseline erklaert oder verglichen werden soll.
