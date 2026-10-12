# Multi-Scale Deep Learning Architecture for Surface Defect Detection

## Overview
This project implements a multi-scale deep learning architecture for detecting surface defects in manufacturing images. The model uses parallel convolution branches to learn features at different scales and classify images as defective or non-defective.

## Features
- Multi-scale convolutional neural network using PyTorch
- Image classification into Defective and Non-Defective categories
- Training and validation support
- Model checkpoint saving
- Evaluation using a classification report and confusion matrix
- Prediction on individual images

## Project Structure
```text
.
├── dataset/
│   ├── train/
│   │   ├── Defective/
│   │   └── Non-Defective/
│   └── val/
│       ├── Defective/
│       └── Non-Defective/
├── models/
│   ├── __init__.py
│   └── multi_scale_model.py
├── outputs/
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements
- Python 3.10 or compatible version
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- Pillow

## Installation
Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation
Place your images in the appropriate folders:

- `dataset/train/Defective/`
- `dataset/train/Non-Defective/`
- `dataset/val/Defective/`
- `dataset/val/Non-Defective/`

Use labeled images that represent both classes. The dataset folders currently contain placeholders; actual images must be added before training.

## Training
```bash
python train.py
```

## Evaluation
```bash
python evaluate.py
```

## Prediction
```bash
python predict.py path/to/image.jpg
```

Check the scripts for their expected arguments and configuration before running them.

## Disclaimer
Model performance depends on the quality, diversity, and labeling of the dataset.

## License
This project is licensed under the MIT License.
