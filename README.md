# Brain Tumor Detection Using Deep Learning Models

## Overview

This project presents a deep learning-based system for automatic brain tumor classification from MRI scans. It benchmarks several state-of-the-art Convolutional Neural Network (CNN) architectures and proposes a custom feature-level ensemble model that combines VGG19 and ResNet50 to improve classification performance.

The objective of this project is to assist medical diagnosis by providing a fast, accurate, and automated solution for identifying different types of brain tumors from MRI images.

---

## Features

* Automatic brain tumor classification from MRI images
* Transfer learning using pre-trained CNN models
* Custom feature-level ensemble architecture
* Data preprocessing and augmentation pipeline
* Performance evaluation using multiple metrics
* Confusion matrix and training visualization
* Modular and reproducible implementation

---

## Problem Statement

Brain tumor diagnosis using MRI scans requires significant expertise and can be time-consuming. This project aims to automate the classification process using deep learning models, reducing diagnostic time while maintaining high accuracy.

The model classifies MRI scans into four categories:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

---

## Model Architecture

The proposed system follows the workflow below:

```text
Brain MRI Images
        │
        ▼
Image Preprocessing
(Resize, Normalize, Data Augmentation)
        │
        ▼
Feature Extraction
(VGG19 + ResNet50)
        │
        ▼
Feature Concatenation
        │
        ▼
Global Average Pooling
        │
        ▼
Dense Layers + Dropout
        │
        ▼
Softmax Classifier
        │
        ▼
Brain Tumor Classification
```

---

## Models Evaluated

The following deep learning architectures were implemented and compared:

* VGG16
* VGG19
* ResNet50
* InceptionV3
* DenseNet121
* MobileNetV2
* Custom VGG19 + ResNet50 Feature Fusion Ensemble

---

## Dataset

The project uses the Brain Tumor MRI Dataset containing four classes:

| Class      | Description      |
| ---------- | ---------------- |
| Glioma     | Glioma Tumor     |
| Meningioma | Meningioma Tumor |
| Pituitary  | Pituitary Tumor  |
| No Tumor   | Healthy MRI      |

### Dataset Split

| Dataset    | Images |
| ---------- | -----: |
| Training   |  5,645 |
| Validation |    564 |
| Testing    |  1,311 |
| Total      |  7,520 |

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* OpenCV
* Scikit-learn
* Matplotlib
* Seaborn
* Google Colab
* Git
* GitHub

---

## Data Preprocessing

The preprocessing pipeline includes:

* Image resizing (224 × 224)
* Pixel normalization
* Random rotation
* Horizontal flipping
* Width and height shifting
* Zoom augmentation
* Brightness adjustment
* Shear transformation

These techniques improve model generalization and reduce overfitting.

---

## Training Configuration

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Optimizer        | Adam                     |
| Learning Rate    | 0.0001 – 0.01            |
| Batch Size       | 64                       |
| Epochs           | 5–10                     |
| Loss Function    | Categorical Crossentropy |
| Early Stopping   | Enabled                  |
| Model Checkpoint | Enabled                  |

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Training Loss
* Validation Loss

---

## Results

| Model                     | Test Accuracy |
| ------------------------- | ------------: |
| MobileNetV2               |        89.15% |
| ResNet50                  |        90.85% |
| VGG19                     |        91.00% |
| DenseNet121               |        91.40% |
| InceptionV3               |        92.22% |
| VGG19 + ResNet50 Ensemble |    **93.85%** |

The proposed feature-level ensemble model achieved the highest classification accuracy by combining hierarchical features from VGG19 with residual features from ResNet50.

---

## Project Structure

```text
Brain-Tumor-Detection/
│
├── dataset/
│
├── notebooks/
│   ├── Data_Preprocessing.ipynb
│   ├── VGG16.ipynb
│   ├── VGG19.ipynb
│   ├── ResNet50.ipynb
│   ├── InceptionV3.ipynb
│   ├── DenseNet121.ipynb
│   ├── MobileNetV2.ipynb
│   └── Ensemble_Model.ipynb
│
├── models/
│
├── outputs/
│   ├── confusion_matrix/
│   ├── training_curves/
│   └── predictions/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Brain-Tumor-Detection.git
```

Navigate to the project directory:

```bash
cd Brain-Tumor-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## Requirements

```text
tensorflow
keras
numpy
pandas
opencv-python
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Applications

* Medical Image Analysis
* Computer-Aided Diagnosis (CAD)
* Clinical Decision Support Systems
* AI-Assisted Radiology
* Healthcare Automation
* Medical Research

---

## Future Work

* Vision Transformer (ViT) implementation
* Brain tumor segmentation using U-Net
* Explainable AI with Grad-CAM
* DICOM image support
* TensorFlow Lite deployment
* ONNX model conversion
* Web-based clinical application
* Multi-modal medical image analysis

---

## Author

**Aniket Mishra**

Bachelor of Technology (Computer Science and Engineering)

Areas of Interest:

* Artificial Intelligence
* Deep Learning
* Computer Vision
* Medical Image Analysis
* Machine Learning

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

* TensorFlow
* Keras
* OpenCV
* Scikit-learn
* Kaggle Brain Tumor MRI Dataset
* Google Colaboratory

---

If you find this project useful, consider starring the repository and contributing through issues or pull requests.
