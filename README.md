# RailVLM — Railway Track Crack & Irregularity Detection

## Overview

RailVLM is a computer vision-based railway track inspection project designed to detect visible cracks and surface irregularities in railway track images and videos.

The system uses a YOLOv8-based object detection pipeline to identify potential rail defects and produce detection results with bounding boxes and confidence scores. RailVLM provides both command-line tools for training, testing, and diagnostics and a FastAPI-based web application for interactive inference and visualization.

The project is intended to demonstrate the application of deep learning and computer vision techniques to automated railway infrastructure inspection.

---

## Features

* Custom-trained YOLOv8-based model for railway defect and crack detection
* Image-based inference
* Video-based inference
* Adjustable confidence threshold for detection
* Detection visualization with bounding boxes and confidence scores
* FastAPI backend for inference services
* Web frontend for interactive image/video analysis
* Command-line support for training and inference
* Dataset preparation and validation utilities
* Environment, GPU, and dataset diagnostic tools
* Training and evaluation output generation

---

## Project Structure

```text
RailVLM/
│
├── backend/                 # FastAPI application and inference API
├── frontend/               # Web interface for image/video analysis
├── models/                 # Model weights and trained models
├── pretrained/             # Pretrained YOLOv8 checkpoints
├── rail/                   # Railway track dataset
├── rail_yolo/              # YOLO dataset configuration
├── utils/                  # Dataset preparation and utility modules
├── runs/detect/            # Training outputs, metrics, plots, and weights
├── docs/                   # Project documentation and supporting materials
│
├── train.py                # Dataset preparation and model training
├── test_inference.py       # Image/directory inference and testing
├── diagnose.py             # Environment, GPU, and dataset diagnostics
├── QUICKSTART.md           # Additional setup and usage instructions
├── statement.md            # Project statement
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Requirements

Before running RailVLM, ensure that the following software is installed:

* Python 3.9 or later
* pip
* Git
* A CUDA-capable GPU is recommended for faster model training and inference
* Sufficient storage for the project, dataset, dependencies, and model weights

The project can also be configured to run on CPU, although training and inference performance may be slower.

---

## Installation and Setup

### 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/kartikmodi2006-ctrl/RailVLM.git
cd RailVLM
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Verify the Environment

Run the diagnostic script:

```bash
python diagnose.py
```

This checks the project environment and provides information about the available computing resources and dataset configuration.

---

## Command-Line Usage

RailVLM can be executed through the command line without requiring the web interface.

### 1. Prepare and Validate the Dataset

For the first run, prepare and validate the dataset using:

```bash
python train.py --prepare --validate
```

### 2. Train the Model

To train the YOLOv8-based detection model:

```bash
python train.py --epochs 100 --model m
```

Additional training configurations and CPU/GPU-specific recommendations are available in `QUICKSTART.md`.

### 3. Run Image Inference

To run detection on a single image:

```bash
python test_inference.py --image path/to/image.jpg
```

### 4. Run Directory Inference

To process multiple images from a directory:

```bash
python test_inference.py --dir path/to/images/
```

### 5. Adjust the Detection Confidence

The confidence threshold can be adjusted using:

```bash
python test_inference.py --conf 0.6
```

The confidence threshold controls the minimum confidence required for a detection to be considered.

---

## Web Application

RailVLM also provides a FastAPI-based web application for interactive inference.

Start the backend using:

```bash
python -m uvicorn backend.main:app --reload
```

After the server starts, open the following address in a web browser:

```text
http://localhost:8000
```

The web interface can be used to provide supported image or video inputs and visualize the resulting detections.

---

## Testing

RailVLM provides command-line inference functionality that can be used to validate the trained model.

For example:

```bash
python test_inference.py --dir rail/test/
```

The project also provides diagnostic functionality:

```bash
python diagnose.py
```

Training outputs, including available evaluation metrics, plots, and model weights, are stored under:

```text
runs/detect/
```

Testing should be performed after the required dependencies, dataset, and model weights have been configured.

---

## Model

RailVLM uses the YOLOv8 object detection framework for railway track defect detection.

The project supports model training using YOLOv8 configurations and stores trained model weights under the project's model/training output directories.

The trained model is used during inference to identify visible railway track defects and return their corresponding detection locations and confidence scores.

---

## Dataset

RailVLM uses a railway track image dataset prepared in a YOLO-compatible format.

The dataset is organized for model training, validation, and testing. Dataset preparation and validation are supported through the project training utilities.

The dataset configuration is maintained within the project under:

```text
rail_yolo/
```

Further dataset-specific information and preparation details are provided in the project documentation.

---

## Results

RailVLM produces object detection results for railway track images and videos.

The system provides:

* Detected defect locations
* Bounding boxes
* Detection confidence scores
* Visualized detection outputs
* Training and evaluation outputs

Quantitative evaluation results such as mAP, precision, and recall should be reported using the actual results generated during model evaluation.

---

## Screenshots

Screenshots demonstrating the RailVLM interface and detection results will be included in the project documentation.

Example materials may include:

* Web application interface
* Input railway track image
* Detected crack/irregularity
* Video inference output
* Command-line inference output
* Training/evaluation results

---

## Tech Stack

### Programming Language

* Python

### Computer Vision & Machine Learning

* PyTorch
* Ultralytics YOLOv8
* OpenCV
* NumPy
* SciPy
* scikit-image

### Backend

* FastAPI
* Uvicorn

### Frontend

* Web-based frontend for interactive inference and result visualization

### Development & Version Control

* Git
* GitHub

---

## Additional Documentation

Additional project information and instructions are available in:

* `QUICKSTART.md` — detailed setup and execution instructions
* `statement.md` — problem statement, project scope, target users, and high-level features
* `TRAINING_IMPROVEMENTS.md` — training-related documentation
* `docs/` — supporting project documentation

---

## Project Objective

The objective of RailVLM is to demonstrate how computer vision and deep learning can be applied to the automated visual inspection of railway tracks by detecting visible cracks and surface irregularities from image and video data.

The project focuses on developing an end-to-end pipeline covering dataset preparation, model training, inference, evaluation, and visualization of railway track defect detections.
