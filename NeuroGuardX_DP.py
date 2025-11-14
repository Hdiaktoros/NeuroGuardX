import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import requests
import os

# Function to collect data
def collect_data(dataset_path):
    """
    Collects data from a given path, which can be a URL or a local CSV file.
    """
    try:
        if dataset_path.startswith(('http://', 'https://')):
            # It's a URL, read it as a CSV
            df = pd.read_csv(dataset_path)
            print(f"Successfully loaded data from URL: {dataset_path}")
        elif os.path.exists(dataset_path):
            # It's a local file, read it as a CSV
            df = pd.read_csv(dataset_path)
            print(f"Successfully loaded data from local file: {dataset_path}")
        else:
            print(f"Error: The file or URL specified does not exist: {dataset_path}")
            return None

        # Anonymize data if needed (placeholder)
        if violates_privacy(df):
            df = anonymize(df)

        return df

    except Exception as e:
        print(f"An error occurred while collecting data from {dataset_path}: {e}")
        return None


# Function to clean data
def clean_data(df):
    # Handle missing data using mean imputation for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Handle outliers using Z-score for numeric columns
    # numeric_df = df.select_dtypes(include=np.number)
    # if not numeric_df.empty:
    #     z_scores = np.abs(stats.zscore(numeric_df))
    #     # Keep rows where all z-scores are less than 5
    #     df = df[(z_scores < 5).all(axis=1)]

    return df


# Function to transform data
def transform_data(data, method):
    if data.empty:
        print("Warning: The dataset is empty after cleaning. Cannot transform data.")
        return data

    numeric_data = data.select_dtypes(include=np.number)
    categorical_data = data.select_dtypes(exclude=np.number)

    # Scale numerical data
    if not numeric_data.empty:
        if method == "Z-score":
            scaler = StandardScaler()
            scaled_numeric_data = scaler.fit_transform(numeric_data)
            scaled_df = pd.DataFrame(scaled_numeric_data, columns=numeric_data.columns, index=numeric_data.index)
        elif method == "Min-Max":
            scaler = MinMaxScaler()
            scaled_numeric_data = scaler.fit_transform(numeric_data)
            scaled_df = pd.DataFrame(scaled_numeric_data, columns=numeric_data.columns, index=numeric_data.index)
        else:
            scaled_df = numeric_data
    else:
        scaled_df = pd.DataFrame()

    # One-hot encode categorical data
    if not categorical_data.empty:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_categorical_data = encoder.fit_transform(categorical_data)
        encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_data.columns), index=categorical_data.index)
    else:
        encoded_df = pd.DataFrame()

    return pd.concat([scaled_df, encoded_df], axis=1)


# Main function to preprocess data
def preprocess_data(OSN_dataset_path, Network_dataset_path):
    OSN_data = collect_data(OSN_dataset_path)
    Network_data = collect_data(Network_dataset_path)

    if OSN_data is None or Network_data is None:
        print("One or both datasets could not be loaded. Exiting preprocessing.")
        return None, None

    OSN_cleaned = clean_data(OSN_data)
    Network_cleaned = clean_data(Network_data)

    O_transformed = transform_data(OSN_cleaned, "Z-score")
    N_transformed = transform_data(Network_cleaned, "Min-Max")

    return O_transformed, N_transformed

# Placeholder functions for privacy checks and anonymization
def violates_privacy(data):
    return False  # Replace with actual logic

def anonymize(data):
    return data  # Replace with actual logic
