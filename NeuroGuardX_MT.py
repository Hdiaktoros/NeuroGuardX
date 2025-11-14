from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np

def train_deep_learning_model(features, labels, dataset_type):
    """
    Trains a deep learning model appropriate for the dataset type.
    """
    if features is None or labels is None or features.empty:
        print(f"Skipping deep learning model training for {dataset_type} due to missing data.")
        return None

    # Smart model selection based on data characteristics
    if has_temporal_patterns(features):
        model = initialize_rnn(features.shape)
    else:
        model = initialize_dense_nn(features.shape)

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    print(f"Starting deep learning model training for {dataset_type}...")
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), verbose=1)

    print(f"Deep learning model training complete for {dataset_type}.")
    return model

def train_traditional_ml_models(features, labels, dataset_type):
    """
    Trains a suite of traditional machine learning models.
    """
    if features is None or labels is None or features.empty:
        print(f"Skipping traditional ML model training for {dataset_type} due to missing data.")
        return {}

    models = {
        "SVC": SVC(probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    trained_models = {}
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    print(f"Starting traditional ML model training for {dataset_type}...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        print(f"  - {name} validation accuracy: {score:.4f}")
        trained_models[name] = model

    print(f"Traditional ML model training complete for {dataset_type}.")
    return trained_models

# Placeholder/Helper functions for model initialization
def has_temporal_patterns(features):
    # Simple heuristic: if features are in a sequence (e.g., multiple timesteps), it's temporal.
    # This is a placeholder; a more robust check would be needed for real-world data.
    return len(features.shape) > 2

def initialize_rnn(shape):
    model = Sequential([
        Input(shape=(shape[1], shape[2]) if len(shape) > 2 else (shape[1], 1)),
        LSTM(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def initialize_dense_nn(shape):
    model = Sequential([
        Input(shape=(shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Main function for model training
def train_models(osn_features, osn_labels, network_features, network_labels):
    """
    Main function to orchestrate model training for both OSN and Network datasets.
    """
    # Train Deep Learning Models
    osn_dl_model = train_deep_learning_model(osn_features, osn_labels, "OSN")
    network_dl_model = train_deep_learning_model(network_features, network_labels, "Network")

    # Train Traditional Machine Learning Models
    osn_ml_models = train_traditional_ml_models(osn_features, osn_labels, "OSN")
    network_ml_models = train_traditional_ml_models(network_features, network_labels, "Network")

    return {
        "OSN_DL": osn_dl_model,
        "Network_DL": network_dl_model,
        "OSN_ML": osn_ml_models,
        "Network_ML": network_ml_models
    }
