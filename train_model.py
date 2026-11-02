"""
Physical Activity Pattern Recognition
Training Script for KNN and Random Forest Classifiers

This script loads the Human Activity Recognition (HAR) dataset from the
'Datasets' folder, preprocesses the sensor measurements, trains both
K-Nearest Neighbors (KNN) and Random Forest classifiers as per instructions,
evaluates and compares their performance, and exports the trained models to .pkl files.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def load_data(filepath: str):
    """
    Load dataset and separate features and target label.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    print(f"[*] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[*] Dataset successfully loaded with shape: {df.shape}")

    # Inspect target distribution
    print("\n[*] Activity Class Distribution:")
    activity_counts = df['Activity'].value_counts()
    for activity, count in activity_counts.items():
        print(f"    - {activity:20s}: {count:5d} samples")

    # Drop non-feature columns
    # 'Activity' is the target; 'subject' is the participant ID (not a sensor reading)
    drop_cols = ['Activity']
    if 'subject' in df.columns:
        drop_cols.append('subject')

    X = df.drop(columns=drop_cols)
    y = df['Activity']
    feature_names = list(X.columns)

    print(f"\n[*] Number of feature columns: {X.shape[1]}")
    print(f"[*] Total samples: {X.shape[0]}")

    return X, y, feature_names


def train_knn(X_train, y_train, n_neighbors: int = 5):
    """
    Train a K-Nearest Neighbors classifier wrapped in a StandardScaler pipeline.
    Scaling prevents features with larger magnitudes from dominating Euclidean distance.
    """
    print(f"\n[*] Training K-Nearest Neighbors (k={n_neighbors})...")
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance'))
    ])
    knn_pipeline.fit(X_train, y_train)
    print("    KNN training complete.")
    return knn_pipeline


def train_random_forest(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Train a Random Forest classifier.
    Random Forest is an ensemble of decision trees, robust to noise and multi-class classification.
    """
    print(f"\n[*] Training Random Forest Classifier (trees={n_estimators})...")
    rf_pipeline = Pipeline([
        ('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)
    print("    Random Forest training complete.")
    return rf_pipeline


def evaluate_model(name: str, model, X_test, y_test):
    """
    Evaluate model on test dataset and print comprehensive metrics.
    """
    print(f"\n{'=' * 60}")
    print(f" Evaluation Report: {name}")
    print(f"{'=' * 60}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%\n")

    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    return acc, y_pred


def save_model(model, filepath: str):
    """
    Save the trained model / pipeline to a .pkl file using pickle.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"[+] Model successfully saved to: {filepath}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'Datasets', 'HAR_simple_dataset.csv')

    # 1. Load Data
    X, y, feature_names = load_data(dataset_path)

    # 2. Train / Test Split
    print("\n[*] Splitting dataset into 80% train and 20% test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train size: {X_train.shape[0]} samples")
    print(f"    Test size : {X_test.shape[0]} samples")

    # 3. Train KNN Model
    knn_model = train_knn(X_train, y_train, n_neighbors=5)
    knn_acc, _ = evaluate_model("K-Nearest Neighbors (KNN)", knn_model, X_test, y_test)

    # 4. Train Random Forest Model
    rf_model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
    rf_acc, _ = evaluate_model("Random Forest Classifier", rf_model, X_test, y_test)

    # 5. Performance Comparison
    print(f"\n{'=' * 60}")
    print(" Performance Comparison")
    print(f"{'=' * 60}")
    print(f"1. K-Nearest Neighbors (KNN)  : {knn_acc * 100:.2f}%")
    print(f"2. Random Forest Classifier   : {rf_acc * 100:.2f}%")

    if rf_acc >= knn_acc:
        best_name = "Random Forest Classifier"
        best_model = rf_model
        best_acc = rf_acc
    else:
        best_name = "K-Nearest Neighbors (KNN)"
        best_model = knn_model
        best_acc = knn_acc

    print(f"\n[*] Best Model: {best_name} with {best_acc * 100:.2f}% accuracy")

    # 6. Save Models to .pkl Files
    print(f"\n{'=' * 60}")
    print(" Saving Models to .pkl")
    print(f"{'=' * 60}")

    best_model_path = os.path.join(base_dir, 'best_model.pkl')
    default_model_path = os.path.join(base_dir, 'model.pkl')
    rf_model_path = os.path.join(base_dir, 'random_forest_model.pkl')
    knn_model_path = os.path.join(base_dir, 'knn_model.pkl')

    save_model(best_model, best_model_path)
    save_model(best_model, default_model_path)
    save_model(rf_model, rf_model_path)
    save_model(knn_model, knn_model_path)

    # 7. Verification: Test loading the saved .pkl file and predicting
    print(f"\n{'=' * 60}")
    print(" Verification: Loading Saved .pkl Model")
    print(f"{'=' * 60}")
    with open(best_model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    sample_test_X = X_test.iloc[:5]
    sample_actual_y = y_test.iloc[:5].values
    sample_pred_y = loaded_model.predict(sample_test_X)

    print("Sample inference on 5 test instances:")
    for idx, (actual, pred) in enumerate(zip(sample_actual_y, sample_pred_y), 1):
        status = "CORRECT" if actual == pred else "INCORRECT"
        print(f"  Sample {idx}: Ground Truth = {actual:20s} | Prediction = {pred:20s} [{status}]")

    print("\n[SUCCESS] Training, evaluation, serialization, and verification completed successfully!")


if __name__ == '__main__':
    main()
