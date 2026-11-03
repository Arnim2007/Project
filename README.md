# Physical Activity Pattern Recognition (HAR)

A Machine Learning project for Human Activity Recognition (HAR) using accelerometer and gyroscope sensor measurements. The system classifies human physical movements into six distinct activities using **K-Nearest Neighbors (KNN)** and **Random Forest** algorithms.

---

## 📌 Project Overview

Smartphones and wearable sensors capture continuous triaxial linear acceleration and angular velocity signals. This project processes statistical features extracted from these sensor readings to recognize human activities in real-time.

### Target Activities
The model recognizes 6 physical activities:
1. `STANDING`
2. `SITTING`
3. `LAYING`
4. `WALKING`
5. `WALKING_DOWNSTAIRS`
6. `WALKING_UPSTAIRS`

---

## 📊 Dataset Description

The dataset is located in `Datasets/HAR_simple_dataset.csv`:
- **Total Samples:** 2,947 records
- **Features:** 66 motion sensor feature columns derived from accelerometer and gyroscope measurements (frequency and time-domain signals including 3-axial mean and standard deviation).
- **Target Column:** `Activity` (ground truth label for each movement pattern).
- **Subject Column:** `subject` (volunteer ID) is excluded during training to ensure the model generalizes across different individuals rather than learning person-specific biases.
- **Split Ratio:** 80% Training (2,357 samples) and 20% Testing (590 samples) with stratified sampling.

---

## ⚙️ Models & Methodology

Per project specifications, two classification models were trained and evaluated:

### 1. K-Nearest Neighbors (KNN)
- **Configuration:** $k=5$ with distance-based weighting (`weights='distance'`).
- **Feature Scaling:** Embedded in a Scikit-Learn `Pipeline` with `StandardScaler` to ensure Euclidean distances are not distorted by feature variance and to eliminate data leakage.
- **Test Accuracy:** **92.88%**

### 2. Random Forest Classifier
- **Configuration:** 100 Decision Trees (`n_estimators=100`, `random_state=42`).
- **Characteristics:** Bagging ensemble algorithm robust against non-linear interactions, multi-collinearity, and noise.
- **Test Accuracy:** **97.63%** *(Selected as the primary production model)*

---

## 📈 Performance Comparison

| Metric | K-Nearest Neighbors (KNN) | Random Forest Classifier (Best) |
| :--- | :---: | :---: |
| **Accuracy** | **92.88%** | **97.63%** |
| **Macro Precision** | 93.15% | **97.68%** |
| **Macro Recall** | 92.68% | **97.55%** |
| **Macro F1-Score** | 92.82% | **97.59%** |

### Per-Class Performance (Random Forest)
| Activity | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `LAYING` | 1.0000 | 1.0000 | 1.0000 | 108 |
| `SITTING` | 0.9495 | 0.9592 | 0.9543 | 98 |
| `STANDING` | 0.9623 | 0.9533 | 0.9577 | 107 |
| `WALKING` | 0.9802 | 1.0000 | 0.9900 | 99 |
| `WALKING_DOWNSTAIRS` | 1.0000 | 0.9405 | 0.9693 | 84 |
| `WALKING_UPSTAIRS` | 0.9691 | 1.0000 | 0.9843 | 94 |

---

## 📁 Repository Structure

```text
Project/
├── Datasets/
│   └── HAR_simple_dataset.csv    # Sensor dataset
├── myvenv/                       # Python virtual environment
├── instructions.txt              # Project instructions & specifications
├── train_model.py                # Standalone script to train, evaluate & save models
├── test_predict.py               # Example script demonstrating model inference
├── app.py                        # Streamlit Web UI application
├── model.pkl                     # Default production model (Best Model: Random Forest)
├── best_model.pkl                # Alias for the best performing model
├── random_forest_model.pkl       # Trained Random Forest model file
├── knn_model.pkl                 # Trained KNN pipeline file (with StandardScaler)
└── README.md                     # Project documentation
```

### Explanation of `.pkl` Model Files
- **`model.pkl` & `best_model.pkl`:** Contain the best-performing model (Random Forest, 97.63% accuracy). Saved under standard names for easy integration into web applications or APIs.
- **`random_forest_model.pkl`:** The standalone Random Forest model pipeline.
- **`knn_model.pkl`:** The standalone KNN model pipeline including its fitted `StandardScaler`.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Dependencies
Ensure Python 3.10+ is installed with the following packages:
- `pandas`
- `numpy`
- `scikit-learn`

If using the included virtual environment:
```powershell
.\myvenv\Scripts\activate
```

### 2. Training the Models
To train both models, print evaluation reports, and export the `.pkl` files:
```powershell
python train_model.py
```

### 3. Running Sample Predictions
To test loading the saved model and predicting sample activities:
```powershell
python test_predict.py
```

### 4. Running the Streamlit Web Application
To launch the interactive Web UI for testing and evaluating the models:
```powershell
streamlit run app.py
```
Or with the virtual environment python:
```powershell
.\myvenv\Scripts\streamlit.exe run app.py
```

#### Web UI Features:
- **🎯 Interactive Activity Prediction**: Test single records directly from the dataset, pick random samples, compare against ground truth, view confidence distributions across all 6 classes, and tweak sensor features with real-time sliders.
- **📁 Batch CSV Testing & Evaluation**: Upload custom CSV datasets or run benchmark test splits, compute accuracy and classification metrics, view confusion matrices, and download full prediction results.
- **📊 Model Comparison & Analytics**: Side-by-side performance comparison between Random Forest (97.63%) and KNN (92.88%), per-class metrics, and top 15 feature importances.
- **📖 Activity & Sensor Guide**: Comprehensive reference for all 6 physical activities, sensor coordinate systems, Butterworth filter pipeline, and frequency-domain transformations.

### 5. Using the Model in Python
```python
import pickle
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load new sensor readings (66 feature columns)
new_data = pd.read_csv('Datasets/HAR_simple_dataset.csv').drop(columns=['Activity', 'subject']).iloc[:5]

# Predict activity
predictions = model.predict(new_data)
print("Predicted Activities:", predictions)
```

