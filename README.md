# ISL PoseGRU: Indian Sign Language Recognition System

A production-ready baseline for **Indian Sign Language (ISL)** recognition using pose keypoints from MediaPipe Holistic and a **GRU** sequence model in PyTorch. This system provides real-time ISL gesture recognition with high accuracy using standard webcam input.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Overview

This project addresses the communication gap between the deaf community and hearing individuals by providing an accessible, real-time Indian Sign Language recognition system that:

- ✅ Recognizes **84 ISL signs** with 82.4% validation accuracy
- ✅ Works in **real-time** (15 FPS) with standard webcam
- ✅ Requires **no specialized hardware**
- ✅ Provides **production-ready pipeline** with rich UI
- ✅ Supports both **offline** and **real-time** inference

## 📋 Table of Contents

- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Quick Start](#-quick-start)
- [Detailed Usage](#-detailed-usage)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Configuration](#-configuration)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Core Functionality
- **Pose-based Recognition**: Uses MediaPipe Holistic for robust keypoint extraction
- **GRU Sequence Modeling**: Bidirectional GRU networks for temporal pattern recognition
- **Real-time Processing**: Live webcam demo with sliding window classification
- **Batch Processing**: Efficient training and evaluation on large datasets

### User Experience
- **Rich Terminal Interface**: Beautiful progress bars and training visualization
- **Configurable Pipeline**: YAML-based configuration for easy experimentation
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and reports
- **Resume Capability**: Smart checkpointing and extraction resumption

### Production Features
- **Docker Support**: Containerized deployment for consistency
- **Modular Design**: Clean separation of concerns for easy maintenance
- **Error Handling**: Robust error recovery and logging
- **Cross-platform**: Works on Windows, Linux, and macOS

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, or macOS 10.15+
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for dataset and models
- **Camera**: Standard USB webcam for real-time demo

### Recommended Setup
- **CPU**: Intel i7 or AMD Ryzen 7 (multi-core for faster processing)
- **GPU**: NVIDIA GTX 1060+ or RTX series (optional, for faster training)
- **RAM**: 32GB for large dataset processing
- **Storage**: SSD for faster I/O operations

### Tested Configuration
This project was developed and tested on:
- **CPU**: Intel Core i7-13700HX
- **GPU**: NVIDIA GeForce RTX 4050 (6GB VRAM)
- **RAM**: 32GB DDR5
- **OS**: Windows 11 24H2

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/ISL-PoseGRU.git
cd ISL-PoseGRU
```

### Step 2: Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n isl-posegru python=3.10
conda activate isl-posegru

# OR using venv
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; import cv2; import mediapipe; print('✅ Installation successful!')"
```

## 📂 Dataset Preparation

### Step 1: Organize Your Data
Create the following directory structure:
```
data/
├── train/
│   ├── class1/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

### Step 2: Split Existing Dataset (Optional)
If you have an unsplit dataset, use our smart splitting tool:

```bash
# Configure split parameters in split_config.yaml
python split_dataset.py --data_root raw_data --output_root data --train_ratio 0.7
```

This will:
- Split your data into 70% train / 30% validation
- Maintain class balance across splits
- Generate detailed split reports
- Preserve original folder structure

### Step 3: Extract Keypoints
Transform videos into pose feature sequences:

```bash
# Extract features for all classes
python extract_keypoints_selective.py --data_root data --out_root features_all

# OR extract specific classes (configure in extract_config.yaml)
python extract_keypoints_selective.py --data_root data --out_root features --first_n 10
```

**Configuration Options** (in `extract_config.yaml`):
- `selection_mode`: "all", "first_n", "specific", or "range"
- `fps`: Target frame rate for processing (default: 15)
- `max_frames`: Maximum frames per video (null for no limit)

## 🏃‍♂️ Quick Start

### 1. Train a Model (10 Classes)
```bash
# Quick training on first 10 classes
python extract_keypoints_selective.py --data_root data --out_root features --first_n 10
python train.py --features_root features --epochs 50 --batch_size 8 --lr 2e-4
```

### 2. Evaluate Performance
```bash
python eval.py --features_root features --checkpoint checkpoints/best_gru.pt
```

### 3. Test Single Video
```bash
python predict.py --video path/to/test.mp4 --checkpoint checkpoints/best_gru.pt
```

### 4. Real-time Demo
```bash
python streaming_demo.py --checkpoint checkpoints/best_gru.pt --window 32 --stride 4
```

## 📖 Detailed Usage

### Training Pipeline

#### For 10 Classes (Quick Experimentation)
```bash
# Extract keypoints
python extract_keypoints_selective.py --data_root data --out_root features --first_n 10

# Train model
python train.py --features_root features --epochs 50 --batch_size 8 --lr 2e-4 --max_len 32

# Evaluate
python eval.py --features_root features --checkpoint checkpoints/best_gru.pt
```

#### For All 84 Classes (Full Dataset)
```bash
# Extract all keypoints
python extract_keypoints_selective.py --data_root data --out_root features_all

# Train with optimized parameters
python train.py --features_root features_all --epochs 90 --batch_size 16 --lr 1e-4 --max_len 32 --dropout 0.4 --hidden 512

# Comprehensive evaluation
python eval.py --features_root features_all --checkpoint checkpoints/best_gru.pt
```

### Configuration Management

#### Extract Configuration (`extract_config.yaml`)
```yaml
# Selection mode
selection_mode: "first_n"  # or "all", "specific", "range"
first_n_classes: 10

# Processing settings
extraction:
  fps: 15.0
  max_frames: null
  overwrite: false
```

#### Training Parameters
```bash
# Key parameters for different scenarios
--epochs 50          # Training epochs (50 for 10 classes, 90 for 84 classes)
--batch_size 8       # Batch size (8 for 10 classes, 16 for 84 classes)
--lr 2e-4           # Learning rate (2e-4 for 10 classes, 1e-4 for 84 classes)
--hidden 256        # Hidden size (256 for 10 classes, 512 for 84 classes)
--dropout 0.3       # Dropout rate (0.3 for 10 classes, 0.4 for 84 classes)
```

### Real-time Demo Features

The streaming demo provides:
- **Live Recognition**: Real-time sign classification
- **Confidence Scores**: Prediction confidence display
- **Sliding Window**: Smooth predictions with overlapping windows
- **Simple Controls**: Press 'q' to quit

```bash
# Standard demo
python streaming_demo.py --checkpoint checkpoints/best_gru.pt

# Custom window settings
python streaming_demo.py --checkpoint checkpoints/best_gru.pt --window 32 --stride 4
```

## 📁 Project Structure

```
ISL-PoseGRU/
├── 📊 Data Processing
│   ├── split_dataset.py           # Smart dataset splitting
│   ├── extract_keypoints_selective.py  # Configurable feature extraction
│   └── merge_class_mappings.py    # Class mapping utilities
├── 🧠 Model & Training
│   ├── common.py                  # Core models and utilities
│   ├── train.py                   # Enhanced training with Rich UI
│   ├── eval.py                    # Comprehensive evaluation
│   └── predict.py                 # Single video inference
├── 🎥 Real-time Demo
│   ├── streaming_demo.py          # Live webcam recognition
│   └── streaming_ctc_demo.py      # Continuous recognition
├── ⚙️ Configuration
│   ├── extract_config.yaml        # Extraction parameters
│   ├── split_config.yaml         # Dataset splitting config
│   └── config.yaml               # Model hyperparameters
├── 📋 Documentation
│   ├── README.md                  # This file
│   ├── flow.md                   # Technical pipeline details
│   ├── speech.md                 # Presentation material
│   └── result_report.md          # Comprehensive results
└── 🔧 Utilities
    ├── requirements.txt           # Dependencies
    ├── Dockerfile                # Container deployment
    └── .gitignore                # Git ignore rules
```

## 🏗️ Model Architecture

### Overview
Our system uses a two-stage architecture:
1. **Feature Extraction**: MediaPipe Holistic → 150-dim pose vectors
2. **Sequence Classification**: Bidirectional GRU → Sign classification

### Technical Specifications
- **Input**: Video frames (RGB, variable FPS)
- **Features**: 150-dimensional pose vectors (33 pose + 21×2 hand landmarks)
- **Sequence Length**: 32 frames (~2 seconds at 15 FPS)
- **Model**: Bidirectional GRU with 512 hidden units, 2 layers
- **Output**: 84 ISL sign classes
- **Parameters**: ~2.1M trainable parameters

### Key Components
```python
# Feature extraction
pose_landmarks: 33 points × 2 coords = 66 features
left_hand: 21 points × 2 coords = 42 features  
right_hand: 21 points × 2 coords = 42 features
total_features = 66 + 42 + 42 = 150 per frame

# GRU architecture
input_dim = 150
hidden_size = 512 (bidirectional = 1024 total)
num_layers = 2
output_classes = 84
```

## ⚙️ Configuration

### Extract Configuration
Edit `extract_config.yaml` to customize feature extraction:

```yaml
# Class selection
selection_mode: "first_n"    # "all", "first_n", "specific", "range"
first_n_classes: 10

# Processing
extraction:
  fps: 15.0                  # Target frame rate
  max_frames: null           # Frame limit (null = no limit)
  overwrite: false           # Skip existing files

# Output
output:
  create_filtered_class_map: true
  verbose: true
```

### Training Configuration
Use command-line arguments or modify defaults in `train.py`:

```bash
# Essential parameters
--features_root features_all  # Feature directory
--epochs 90                   # Training epochs
--batch_size 16              # Batch size
--lr 1e-4                    # Learning rate
--max_len 32                 # Sequence length
--hidden 512                 # Hidden dimensions
--dropout 0.4                # Regularization
```

## 📊 Results

### Performance Metrics
- **Dataset**: 4,243 videos across 84 ISL signs
- **Training**: 2,964 videos (70%)
- **Validation**: 1,279 videos (30%)
- **Training Time**: ~6 hours (RTX 4050)

### Model Performance
- **Training Accuracy**: 94.7%
- **Validation Accuracy**: 82.4%
- **Real-time Performance**: 15 FPS
- **Inference Time**: ~80ms per video

### Class Distribution
The model handles 84 diverse ISL signs including:
- Daily communication: "about", "acting", "afternoon"
- Objects: "camera", "table", "umbrella"
- Actions: "exercise", "throw", "boil"
- Descriptive terms: "warm", "calm", "satisfied"

## 🐛 Troubleshooting

### Common Issues

#### 1. MediaPipe Installation Issues
```bash
# If MediaPipe fails to install
pip install --upgrade pip
pip install mediapipe --no-cache-dir
```

#### 2. CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# For CPU-only training, models automatically fall back to CPU
```

#### 3. Video Processing Errors
- Ensure video files are in supported formats (MP4, AVI, MOV)
- Check file permissions and paths
- Verify sufficient disk space for feature extraction

#### 4. Memory Issues
```bash
# Reduce batch size for limited memory
python train.py --batch_size 8  # Instead of 16

# Or process fewer classes at once
python extract_keypoints_selective.py --first_n 5
```

#### 5. Webcam Access Issues
- Ensure camera permissions are granted
- Try different camera indices: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`
- Check if other applications are using the camera

### Performance Optimization

#### For Faster Training
```bash
# Use GPU if available
export CUDA_VISIBLE_DEVICES=0

# Increase batch size (if memory allows)
python train.py --batch_size 32

# Use multiple workers (careful on Windows)
# Modify DataLoader num_workers in train.py
```

#### For Better Accuracy
```bash
# Increase model capacity
python train.py --hidden 512 --layers 3

# More training epochs
python train.py --epochs 120

# Data augmentation (already included in training)
```

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

### Development Setup
```bash
git clone https://github.com/your-username/ISL-PoseGRU.git
cd ISL-PoseGRU
conda create -n isl-dev python=3.10
conda activate isl-dev
pip install -r requirements.txt
pip install -e .  # Editable install
```

### Code Style
- Follow PEP 8 conventions
- Use meaningful variable names
- Add docstrings for functions
- Include type hints where appropriate

### Testing
```bash
# Run basic functionality tests
python -c "from common import GRUClassifier; print('✅ Model import successful')"
python extract_keypoints_selective.py --help
python train.py --help
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe Team** for excellent pose detection capabilities
- **PyTorch Team** for the robust deep learning framework
- **Rich Library** for beautiful terminal interfaces
- **ISL Community** for inspiration and guidance

## 📞 Support

If you encounter issues or have questions:

1. **Check Troubleshooting** section above
2. **Search existing issues** in the repository
3. **Create a new issue** with detailed description
4. **Include system information** and error logs

---

**Made with ❤️ for the deaf and hard-of-hearing community**

*This project aims to break down communication barriers and create a more inclusive world through accessible technology.*