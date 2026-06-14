# Face Model Lab

Goal: find the strongest self-trained face detector for the video blurring pipeline.

## Recommended Strategy

Start with YOLO fine-tuning. It is usually the best practical choice for this project because it is fast in video, supports tracking through Ultralytics, and lets you control VRAM directly with `batch` and `imgsz`.

Then test RT-DETR as the main non-YOLO challenger. It is an end-to-end transformer detector and can be a better accuracy/speed compromise than classic R-CNN models.

Use Faster R-CNN, RetinaNet, and FCOS as controlled baselines:

- Faster R-CNN: strong two-stage localization baseline, slower for video.
- RetinaNet: one-stage detector with Focal Loss, useful when negatives dominate.
- FCOS: anchor-free one-stage detector, useful to compare against anchor-based RetinaNet.

## Training Methods To Try

- Increase data gradually: smoke sample, 5k images, full WIDER FACE.
- Stochastic augmentation: mosaic, mixup, HSV/color jitter, random scale, random translation.
- Optimizer/schedule: cosine LR, warmup, early stopping, seed sweeps.
- Batch and image size sweeps: for YOLO try `batch=8/16`, `imgsz=640/768/960`.
- Confidence threshold sweeps during evaluation: `0.2`, `0.35`, `0.5`.
- Evaluate on the same fixed validation subset for every model.

Do not optimize only for recall. For video blurring, a few false positives may be acceptable, but missed faces are worse. Track:

- recall
- ms/image
- VRAM
- false positives by visual spot-check

## Commands

Train Faster R-CNN:

```bash
cd /home/clemi/projekte/MIM
source /home/clemi/.venvs/MIM/bin/activate
python face_model_lab/train_fasterrcnn.py --epochs 1 --batch 2 --reduction 10
```

Train YOLO:

```bash
python face_model_lab/train_yolo.py --base face_yolov8m.pt --epochs 1 --batch 8 --imgsz 640 --train-limit 600 --val-limit 120
```

Train YOLO or RT-DETR with the shared Ultralytics trainer:

```bash
python face_model_lab/train_ultralytics_detector.py --family rtdetr --base rtdetr-l.pt --epochs 1 --batch 4 --imgsz 640
python face_model_lab/train_ultralytics_detector.py --family yolo --base face_yolov8m.pt --epochs 1 --batch 8 --imgsz 640
```

Train Torchvision baselines:

```bash
python face_model_lab/train_torchvision_detector.py --kind retinanet --epochs 1 --batch 2 --reduction 10
python face_model_lab/train_torchvision_detector.py --kind fcos --epochs 1 --batch 2 --reduction 10
python face_model_lab/train_torchvision_detector.py --kind fasterrcnn --epochs 1 --batch 2 --reduction 10
```

Evaluate models:

```bash
python face_model_lab/evaluate_models.py \
  --models trained_models/fasterrcnn_resnet50_fpn_rocm_bs2_ep1.pth face_yolov8m.pt trained_models/rtdetr_rtdetrl_widerface_rocm_bs4_ep1.pt \
  --limit 100
```

Blur a video:

```bash
python face_model_lab/blur_video.py \
  --model face_yolov8m.pt \
  --input Videos/Feuerwehr_progressiv.mp4 \
  --output Videos/Feuerwehr_lab_blur.mp4 \
  --max-frames 200
```

## Model Naming

Saved models use:

```text
<model_type>_bs<batch_size>_ep<epochs>.<suffix>
```

Examples:

```text
fasterrcnn_resnet50_fpn_rocm_bs2_ep1.pth
face_yolov8m_widerface_rocm_bs8_ep1.pt
rtdetr_rtdetrl_widerface_rocm_bs4_ep1.pt
retinanet_resnet50_fpn_rocm_bs2_ep1.pth
fcos_resnet50_fpn_rocm_bs2_ep1.pth
```
