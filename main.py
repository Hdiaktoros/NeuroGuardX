import argparse
import pandas as pd
from NeuroGuardX_DP import preprocess_data
from NeuroGuardX_FS import feature_selection
from NeuroGuardX_MT import train_models
from NeuroGuardX_EAII import explain_model

def main(args):
    """
    Main function to run the NeuroGuardX pipeline.
    """
    print("Starting NeuroGuardX Pipeline...")

    # 1. Data Preprocessing
    print("\n--- Step 1: Data Preprocessing ---")
    osn_preprocessed, network_preprocessed = preprocess_data(args.osn_data, args.network_data)

    # 2. Feature Selection
    print("\n--- Step 2: Feature Selection ---")
    osn_features, osn_labels = feature_selection(osn_preprocessed, "OSN")
    network_features, network_labels = feature_selection(network_preprocessed, "Network")

    # 3. Model Training
    print("\n--- Step 3: Model Training ---")
    all_trained_models = train_models(osn_features, osn_labels, network_features, network_labels)

    # 4. Explainable AI
    print("\n--- Step 4: Explainable AI ---")

    # Select the model to explain based on user's choice
    # For simplicity, we'll explain the OSN models here. A more complex CLI could allow choosing.
    model_to_explain = None
    features_to_explain = None
    model_category = '' # 'deep_learning' or 'traditional'

    if args.model_type.upper() == 'DL':
        model_to_explain = all_trained_models.get("OSN_DL")
        features_to_explain = osn_features
        model_category = 'deep_learning'
    elif args.model_type.upper() in all_trained_models.get("OSN_ML", {}):
        model_to_explain = all_trained_models["OSN_ML"][args.model_type.upper()]
        features_to_explain = osn_features
        model_category = 'traditional'
    else:
        print(f"Model type '{args.model_type}' not found for OSN dataset. Defaulting to RandomForest.")
        model_to_explain = all_trained_models.get("OSN_ML", {}).get("RandomForest")
        features_to_explain = osn_features
        model_category = 'traditional'

    if model_to_explain and features_to_explain is not None and not features_to_explain.empty:
        # Use a smaller subset of the data for SHAP explanation to speed up the process
        explanation_data = features_to_explain.sample(n=100, random_state=42) if len(features_to_explain) > 100 else features_to_explain
        explanation = explain_model(model_to_explain, explanation_data, args.xai_method, model_category)
        if explanation is not None:
            print("\nGenerated Explanation (Top 5 features):")
            # We'll average the absolute importance scores to rank features
            feature_importance = explanation.abs().mean().sort_values(ascending=False)
            print(feature_importance.head(5))
    else:
        print("Could not generate explanation due to missing model or features.")


    print("\nNeuroGuardX Pipeline Finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NeuroGuardX: An Explainable AI pipeline for Intrusion Detection.")

    parser.add_argument('--osn_data', type=str, required=True, help='Path or URL to the OSN dataset CSV file.')
    parser.add_argument('--network_data', type=str, required=True, help='Path or URL to the Network dataset CSV file.')
    parser.add_argument('--model_type', type=str, default='RandomForest',
                        help='The model to explain (e.g., "DL", "SVC", "RandomForest"). Default is RandomForest.')
    parser.add_argument('--xai_method', type=str, default='SHAP',
                        help='The XAI method to use ("LRP" or "SHAP"). Default is SHAP.')

    args = parser.parse_args()
    main(args)
