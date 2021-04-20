# COVID-19 Chest X-ray Detection
Binary classification of COVID-19 from chest X-ray images using CNN with Grad-CAM visualization.

## Components
- `bi_classifier.py`: Main CNN model training
- `grad_cam.py`: Gradient visualization
- Images in `images/` folder

## Requirements
- Python 3.9
- Dependencies listed in requirements.txt

## Installation
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python bi_classifier.py
```

2. Generate Grad-CAM visualization:
```bash
python grad_cam.py
```