"""
Example inference script demonstrating how to load and use the saved .pkl model.
"""

import os
import pickle
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.pkl')
    dataset_path = os.path.join(base_dir, 'Datasets', 'HAR_simple_dataset.csv')

    print(f"Loading trained model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load a few sample records
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=['Activity', 'subject'])
    y = df['Activity']

    # Select random samples
    sample_X = X.sample(n=5, random_state=42)
    sample_y = y.loc[sample_X.index]

    # Predict
    predictions = model.predict(sample_X)

    print("\nInference Results:")
    for actual, pred in zip(sample_y, predictions):
        match = "MATCH" if actual == pred else "MISMATCH"
        print(f"Actual: {actual:20s} | Predicted: {pred:20s} [{match}]")

if __name__ == '__main__':
    main()

