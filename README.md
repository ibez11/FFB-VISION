# Janjang Vision

Janjang Vision is an AI-powered project for detecting and counting oil palm fresh fruit bunches (FFB), locally known as "janjang", using computer vision and YOLO object detection. **Now with ripeness classification!**

The system can:
- **Detect** fruit bunch positions (bounding boxes)
- **Classify** ripeness level: **ripe**, **unripe**, or **underripe**
- Automate fruit counting and ripeness assessment in palm oil farms, inspection workflows, and video monitoring.

It supports image detection, video processing, live camera detection, and custom model training using YOLO-compatible datasets.

## Project Overview

This project uses the Ultralytics YOLO framework to identify oil palm fruit bunches in images and video streams. It detects objects with bounding boxes and counts the number of detected bunches based on confidence threshold and optional class filtering.

The main workflow includes:

- detecting fruit bunches in still images
- processing video footage for counting
- monitoring in real time from a webcam
- training a custom YOLO model from a labeled dataset
- exporting annotated result images and videos

## Features

- **Multi-class detection + classification**
  - Detect oil palm fruit bunch positions (bounding box)
  - Classify ripeness: `ripe`, `unripe`, `underripe`
- YOLO-based object detection
- Image counting from JPG/PNG inputs
- Video counting from MP4 or other OpenCV-supported video sources
- Real-time camera detection with ripeness overlay
- Automatic class detection for relevant labels such as `janjang`, `tbs`, `ffb`, `bunch`, and similar names
- Confidence threshold control using `--conf`
- Custom class filtering using `--classes`
- Model training via `dataset.yaml`
- Output saving for processed images and videos with ripeness annotations

## Project Structure

```text
janjang-vision/
├── janjang_counter.py          # Main detection / training script
├── janjang_train_merged.ipynb  # Notebook for merging Roboflow datasets and training
├── janjang_colab.ipynb         # Notebook version for Colab workflow
├── merged_best.pt              # Recommended merged model checkpoint
├── gcstech.pt                  # Model checkpoint from gcstech dataset
├── model_2.pt                  # Additional checkpoint
├── SawitMVC.pt                 # Another trained checkpoint
├── yolo11n.pt                  # Base YOLO11 pre-trained model
├── ffb_check.jpg               # Example validation image
├── foto_uji.png                # Sample test image
├── runs/                       # YOLO training/inference outputs
│   └── detect/
├── README.md                   # Project documentation
├── train_log.txt               # Training logs
├── train_log_v2.txt            # Training logs v2
├── train_log_v2full.txt        # Full training logs
├── __pycache__/                # Python cache files
├── .git/                       # Git metadata
├── .vscode/                    # editor config
└── .gitignore
```

Note: the project currently uses root-level checkpoints and output folders rather than a single `dataset/` folder in the workspace. If you want to retrain, you can create a custom `dataset/` folder or point the training command to a custom `data.yaml` file.

## Requirements

Install the required Python packages:

```bash
pip install ultralytics opencv-python
```

## Download Model from Google Drive

The recommended trained models are stored in Google Drive:

- **merged_best.pt** - Detection model (count only)
  - Google Drive folder: https://drive.google.com/drive/folders/1TpYLqy0NLaXq-XeoPMsomkt-K25bPUtk?hl=ID
  - Use for: Quick counting without ripeness info
  
- **merged_ripeness_best.pt** - Detection + Ripeness Classification model (RECOMMENDED)
  - Same folder as above
  - Use for: Detect + classify ripeness (ripe/unripe/underripe)

Download either file and place in the project root folder.

### Using merged_ripeness_best.pt (with ripeness)

```bash
python janjang_counter.py foto.jpg --model merged_ripeness_best.pt --conf 0.25
```

Output will show:
- Bounding boxes for each janjang
- Ripeness label: `ripe` (green), `unripe` (yellow), `underripe` (red)
- Count by ripeness class

### Using merged_best.pt (detection only)

```bash
python janjang_counter.py foto.jpg --model merged_best.pt --conf 0.25
```

## Download Datasets from Roboflow

This project uses multiple datasets for detection and ripeness classification:

### Detection datasets (3 datasets):
1. **bunchtest** - 237 close-up FFB images
2. **nazwa** - 2,400+ plantation scene images
3. **gcstech** - 2,700+ images with ripeness labels (Decayed, Fully Ripe, Immature, Over Ripe, Partially Ripe)
   - https://universe.roboflow.com/gcstech/oil-palm-fruit-bunch-vlynl/dataset/2

### Ripeness classification dataset (1 dataset):
4. **workspace-alwjv palm-oil-ripeness-classification** - Additional ripeness data
   - https://universe.roboflow.com/workspace-alwjv/palm-oil-ripeness-classification-iqmds

### Training with merged datasets

Use the **[janjang_train_ripeness.ipynb](janjang_train_ripeness.ipynb)** notebook to:
- Download all 4 datasets
- Merge and remap ripeness classes to: `ripe`, `unripe`, `underripe`
- Train a multi-class detection+classification model


## Quick Start

### 1. Run object detection on an image (with ripeness)

Recommended - use ripeness model:

```bash
python janjang_counter.py image.jpg --model merged_ripeness_best.pt --conf 0.25
```

This will output bounding boxes with ripeness labels (ripe/unripe/underripe).

Alternatively, detection-only model:

```bash
python janjang_counter.py image.jpg --model merged_best.pt --conf 0.25
```

### 2. Run detection on a video

```bash
python janjang_counter.py --video input_video.mp4 --model merged_best.pt --save
```

This is the correct video mode for the current script. The `--save` flag saves the annotated output video beside the input file.

### 3. Use the webcam

```bash
python janjang_counter.py --cam --model merged_best.pt
```

This opens the live webcam stream and displays detections in real time. In the current implementation, webcam mode does not save a video file automatically.

### 4. Train a custom model

```bash
python janjang_counter.py --train dataset/dataset.yaml --epochs 100 --imgsz 640
```

## Model and Class Handling

The script automatically tries to detect labels related to fruit bunches. If a custom model uses class names like:

- `janjang`
- `tbs`
- `ffb`
- `bunch`
- `fruit_bunch`

it will identify the object class that matches and count only those relevant detections.

## Dataset

The project includes a YOLO-style dataset structure under the `dataset/` folder. The training configuration is stored in `dataset/dataset.yaml` and includes:

- training image folder
- validation image folder
- number of classes
- class names

This allows the model to be trained for the custom palm fruit bunch detection task.

## Output

The program can generate:

- annotated prediction images with bounding boxes
- saved result files such as `*_hasil.jpg` or `*_hasil.png`
- processed video outputs when `--save` is enabled
- training checkpoints and logs under the `runs/` directory

### Example detection result

This result was generated using the recommended model checkpoint: `merged_best.pt`.

Input image: `ffb_check.jpg`  
Output image: `ffb_check_hasil.png`

![FFB detection result using merged_best.pt](ffb_check_hasil.png)

## Notes

- The default `yolo11n.pt` model is a general pretrained COCO model and is not specialized for palm fruit detection.
- For better field accuracy, you should collect labeled images of actual fruit bunches, annotate them, and retrain the model with your custom dataset.
- The system is useful for agricultural monitoring, harvest estimation, and inspection support.

## Example Use Case

This project is suitable for counting fresh fruit bunches in palm plantations, where the goal is to estimate the number of bunches from field images or surveillance videos. It can assist in:

- field inspection
- fruit counting automation
- harvest planning support
- AI-based agricultural monitoring

## License

This project is intended for research, educational, and practical agricultural vision use. Please ensure compliance with the licensing terms of the dependencies used in the project.
