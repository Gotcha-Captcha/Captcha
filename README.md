# Captcha Recognition System

An automated text-based CAPTCHA recognition algorithm that converts images into text. This project is useful for building automated testing environments and developing accessibility tools.

## Project Overview

- **Application Type**: Image Classification & Computer Vision
- **Objective**: Recognize 5-character text-based CAPTCHAs with noise lines and attached characters.

## Dataset

- **Source**: [Kaggle - Captcha Version 2 Images](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images)
- **Details**: 
    - Approximately 1,000 images.
    - Each image contains 5 characters.
    - Total training samples: ~5,000 characters.

## Model Architecture

### LinearSVC (Support Vector Machine)
The project uses a `LinearSVC` model for its efficiency and strong generalization performance even with relatively small datasets.

### Feature Extraction: HOG (Histogram of Oriented Gradients)
- **Method**: Measures the distribution of gradient directions within localized regions.
- **Why HOG?**: Focuses on the "outline shape" and "structure" of characters rather than pixel values, making it robust against lighting variations and noise.

## Tech Stack

- **Language**: Python 3.x
- **Development Environment**: Jupyter Notebook
- **Machine Learning**: Scikit-learn
- **Experiment Tracking**: MLflow
- **Image Processing**: OpenCV / Scikit-image

## Getting Started

### 1. Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Usage
Open the Jupyter Notebook and execute the cells:
```bash
jupyter notebook captcha.ipynb
```

### 4. Experiment Tracking
Experiments are tracked using MLflow. To view the results:
```bash
mlflow ui --port 5001
```

## Results & Models
The trained models are saved as pickle files:
- `captcha_svm_model.pkl`: The trained LinearSVC model.
- `label_encoder.pkl`: Encoder for character labels.