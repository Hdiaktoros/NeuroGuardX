from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Function to reduce dimensionality on numeric features
def reduce_dimensionality(features, dataset_type):
    numeric_features = features.select_dtypes(include=np.number)
    non_numeric_features = features.select_dtypes(exclude=np.number)

    if numeric_features.empty:
        return features # No numeric features to reduce

    if dataset_type == "OSN" and numeric_features.shape[1] > 50:
        reducer = TSNE(n_components=2, perplexity=min(30, len(numeric_features)-1), random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)

    reduced_numeric_data = reducer.fit_transform(numeric_features)
    reduced_df = pd.DataFrame(reduced_numeric_data, index=numeric_features.index)

    return pd.concat([reduced_df, non_numeric_features], axis=1)


# Function to find important features using RFE
def select_important_features(features, labels):
    numeric_features = features.select_dtypes(include=np.number)
    non_numeric_features = features.select_dtypes(exclude=np.number)

    if numeric_features.empty:
        return features # No numeric features to select from

    estimator = RandomForestClassifier(random_state=42)
    # Select top 5 features, or all if less than 5
    n_features_to_select = min(5, numeric_features.shape[1])
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector = selector.fit(numeric_features, labels)

    selected_numeric_features = numeric_features.loc[:, selector.support_]

    return pd.concat([selected_numeric_features, non_numeric_features], axis=1)


# Function to engineer features (placeholder)
def engineer_features(features, dataset_type):
    if dataset_type == "OSN":
        engineered_features = derive_OSN_metrics(features)
    else:
        engineered_features = derive_Network_metrics(features)
    return engineered_features

# Placeholder functions for deriving metrics
def derive_OSN_metrics(data):
    # Replace with actual logic, e.g., creating interaction terms
    return data

def derive_Network_metrics(data):
    # Replace with actual logic, e.g., calculating packet ratios
    return data

# Main function for feature selection for a single dataset
def feature_selection(dataset, dataset_type):
    """
    Performs feature selection on a given dataset.
    Assumes the last column is the label.
    """
    if dataset is None or dataset.empty:
        print(f"Dataset for {dataset_type} is empty. Skipping feature selection.")
        return None, None

    # Assume the last column is the label
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # --- Pipeline for feature processing ---
    # 1. Reduce dimensionality
    X_reduced = reduce_dimensionality(X, dataset_type)

    # 2. Select important features
    X_important = select_important_features(X_reduced, y)

    # 3. Engineer new features
    X_engineered = engineer_features(X_important, dataset_type)

    print(f"Feature selection complete for {dataset_type} dataset.")

    return X_engineered, y
