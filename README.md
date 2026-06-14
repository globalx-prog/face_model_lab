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

## File Order

Use the lab in this order:

```text
README.md
step00_common.py
step01_training_evaluation_lab.ipynb
step02_train_ultralytics_detector.py
step03_train_torchvision_detector.py
step04_train_fasterrcnn.py
step05_train_yolo_legacy.py
step06_evaluate_models.py
step07_video_blurring_lab.ipynb
step08_blur_video.py
```

`step01_training_evaluation_lab.ipynb` is the main training and evaluation control surface. It calls the numbered Python scripts and shows progress/output. `step07_video_blurring_lab.ipynb` is the video workflow.

## Bigger Training Runs

Open `step01_training_evaluation_lab.ipynb` and change the config cell:

```python
RUN_PRESET = "smoke"   # "smoke", "medium" or "full"
```

Recommended flow:

```python
RUN_PRESET = "medium"
cfg["epochs"] = 5
cfg["ultra_train_limit"] = 5000
cfg["eval_limit"] = 500
```

For a serious full run:

```python
RUN_PRESET = "full"
cfg["epochs"] = 15
cfg["batch"] = 2
cfg["imgsz"] = 768
```

Torchvision models are selected with:

```python
TORCHVISION_KINDS = ["retinanet", "fasterrcnn"]
SAVE_EVERY = 1
```

`SAVE_EVERY = 1` writes a usable model checkpoint after every epoch. If a 15 epoch run reaches epoch 12 and the process stops afterwards, the `*_ep12.pth` checkpoint can still be evaluated and used.

## Progress Display

Training scripts use `tqdm` progress bars. For Torchvision/Faster R-CNN the progress bar also prints live VRAM values from ROCm:

```text
loss=..., alloc=..., res=..., free=..., total=...
```

The same progress output appears in VS Code/Jupyter cell output and in the CLI.

## Batch Size Notes

A larger batch can be slower per epoch on ROCm for detection models because images have different sizes and many boxes, so each batch may trigger more padding, CPU collation, GPU memory traffic, and larger FPN/RPN/ROI workloads. If VRAM pressure increases, PyTorch may reserve more memory and kernels can become less efficient. For this project `batch=2` is a good stable default; improve throughput first with more images/epochs, then test `batch=4` only if thermals and VRAM stay healthy.

## Faster R-CNN Parameters

Current defaults are conservative:

- `lr=1e-4`: stable for fine-tuning pretrained detectors.
- `momentum=0.9`, `weight_decay=5e-4`: normal SGD baseline for Faster R-CNN.
- cosine scheduler: reasonable for longer runs.
- `clip_grad_norm_=1.0`: useful against unstable detection batches.
- `batch=2`: good ROCm default for WIDER FACE and your GPU thermals.

For a stronger Faster R-CNN run, try:

```bash
python face_model_lab/step04_train_fasterrcnn.py --epochs 15 --batch 2 --reduction 1 --lr 0.0001 --save-every 1
```

For experiments, change one thing at a time: `lr=5e-5`, `lr=2e-4`, then compare recall and ms/image.

Train Faster R-CNN:

```bash
cd /home/clemi/projekte/MIM
source /home/clemi/.venvs/MIM/bin/activate
python face_model_lab/step04_train_fasterrcnn.py --epochs 1 --batch 2 --reduction 2000 --save-every 1
```

Train YOLO:

```bash
python face_model_lab/step05_train_yolo_legacy.py --base face_yolov8m.pt --epochs 1 --batch 2 --imgsz 640 --train-limit 20 --val-limit 8
```

Train YOLO or RT-DETR with the shared Ultralytics trainer:

```bash
python face_model_lab/step02_train_ultralytics_detector.py --family rtdetr --base rtdetr-l.pt --epochs 1 --batch 2 --imgsz 640 --train-limit 20 --val-limit 8
python face_model_lab/step02_train_ultralytics_detector.py --family yolo --base face_yolov8m.pt --epochs 1 --batch 2 --imgsz 640 --train-limit 20 --val-limit 8
```

Train Torchvision baselines:

```bash
python face_model_lab/step03_train_torchvision_detector.py --kind retinanet --epochs 1 --batch 2 --reduction 2000 --save-every 1
python face_model_lab/step03_train_torchvision_detector.py --kind fcos --epochs 1 --batch 2 --reduction 2000 --save-every 1
python face_model_lab/step03_train_torchvision_detector.py --kind fasterrcnn --epochs 1 --batch 2 --reduction 2000 --save-every 1
```

Evaluate models:

```bash
python face_model_lab/step06_evaluate_models.py \
  --models trained_models/fasterrcnn_resnet50_fpn_rocm_bs2_ep1.pth face_yolov8m.pt trained_models/rtdetr_rtdetrl_widerface_rocm_bs4_ep1.pt \
  --limit 100
```

Blur a video:

```bash
python face_model_lab/step08_blur_video.py \
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
