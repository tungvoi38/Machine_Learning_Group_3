# Contents of /project_name/project_name/src/models/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def train_model(data, target_column):
    """Train a Random Forest model."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)

if __name__ == "__main__":
    # Example usage
    data = load_data('path/to/your/dataset.csv')
    model, X_test, y_test = train_model(data, target_column='target')
    save_model(model, 'results/models/trained_model.pkl')