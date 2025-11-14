import numpy as np
import shap
import pandas as pd
from tensorflow.keras.models import Model

def lrp_rule(layer, R, a):
    """
    Apply the LRP rule for a given layer.
    This is a simplified epsilon-LRP rule.
    """
    # Get weights and biases
    w = layer.get_weights()[0]
    b = layer.get_weights()[1] if len(layer.get_weights()) > 1 else 0

    # Add a small epsilon to the denominator to avoid division by zero
    epsilon = 1e-7

    # Compute the relevance scores for the previous layer
    z = np.dot(a, w) + b
    s = R / (z + epsilon)
    c = np.dot(s, w.T)

    return a * c

def lrp_explain(model, data):
    """
    Perform Layer-wise Relevance Propagation (LRP) for a dense neural network.
    """
    # Get the activations of each layer
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(data)

    # Initialize relevance scores at the output
    R = model.predict(data)

    # Propagate relevance scores backwards through the layers
    for i in range(len(model.layers) - 1, 0, -1):
        R = lrp_rule(model.layers[i], R, activations[i-1])

    return R

def shap_explain(model, data, model_type='deep_learning'):
    """
    Perform SHAP analysis on a model.
    """
    if model_type == 'deep_learning':
        # For deep learning models, GradientExplainer is a good choice
        explainer = shap.GradientExplainer(model, data)
    else:
        # For tree-based models, TreeExplainer is efficient
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, data)
        else:
            explainer = shap.KernelExplainer(model.predict, data)

    shap_values = explainer.shap_values(data)

    # If the shap_values is a list of arrays (multi-class), take the first one.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return shap_values

def explain_model(model, features, method, model_type):
    """
    Main function to generate explanations for a trained model.
    """
    if model is None:
        print(f"Skipping explanation for an invalid model.")
        return None

    print(f"Generating explanations using {method} for a {model_type} model...")

    if method.upper() == 'LRP':
        if model_type != 'deep_learning':
            print("Warning: LRP is only implemented for deep learning models. Skipping.")
            return None
        explanation = lrp_explain(model, features)

    elif method.upper() == 'SHAP':
        explanation = shap_explain(model, features, model_type)

    else:
        raise ValueError(f"Unknown explanation method: {method}")

    # Convert the explanation to a DataFrame for easier interpretation
    explanation_df = pd.DataFrame(explanation, columns=features.columns)

    print("Explanation generation complete.")
    return explanation_df
