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

## Project Structure

```text
Captcha/
├── app/                # FastAPI Web Server
│   ├── main.py         # Entry point & Endpoints
│   ├── model_logic.py  # Core ML & Preprocessing
│   ├── templates/      # Dashboard (UI)
│   └── uploads/        # Temp Storage for Analysis
├── models/             # Trained Weights (.pkl)
├── notebooks/          # Research & Experimentation
├── .gitignore
├── README.md
└── requirements.txt
```

## Tech Stack

- **Backend**: FastAPI, Uvicorn, WebSockets
- **Frontend**: Vanilla CSS (Glassmorphism), JavaScript
- **ML**: Scikit-learn (LinearSVC + HOG)
- **Experiment Tracking**: MLflow

## Getting Started

### 1. Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Usage

#### Demo Web Server
To launch the interactive dashboard:
```bash
# Run from the project root
uvicorn app.main:app --reload
```
Visit http://127.0.0.1:8000 to use the premium Analysis Dashboard.

#### Research Notebook
To run experiments:
```bash
cd notebooks
jupyter notebook captcha.ipynb
```

## Results & Models
The models are stored in the `/models` directory using `joblib` (.pkl).