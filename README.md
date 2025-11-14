# NeuroGuardX: Explainable AI for Intrusion Detection and Prevention

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Pipeline Overview](#pipeline-overview)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgments](#acknowledgments)

## Introduction

This repository contains the code for NeuroGuardX, a research project focused on integrating Explainable AI (XAI) techniques into Intrusion Detection Systems (IDS) for Online Social Networks (OSN) and Network Systems. The goal of NeuroGuardX is to make intrusion detection models more interpretable, transparent, and trustworthy.

## Installation

### Prerequisites

- Python 3.x
- `pip` (Python package installer)

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/Explainable-AI-for-IDP.git
    cd Explainable-AI-for-IDP
    ```

2.  **Install the required packages:**

    ```bash
    pip install pandas numpy scikit-learn tensorflow shap
    ```

## Pipeline Overview

The NeuroGuardX pipeline is an end-to-end workflow that takes raw data and produces trained models with corresponding explanations. The pipeline is orchestrated by the `main.py` script and consists of the following modules:

-   **`NeuroGuardX_DP.py` (Data Preprocessing):** This module handles data loading, cleaning, and transformation. It can ingest data from both local CSV files and URLs, and it automatically performs one-hot encoding for categorical features.

-   **`NeuroGuardX_FS.py` (Feature Selection):** This module performs feature selection to identify the most relevant features for model training.

-   **`NeuroGuardX_MT.py` (Model Training):** This module trains a suite of machine learning models, including both deep learning and traditional models.

-   **`NeuroGuardX_EAII.py` (Explainable AI Integration):** This module integrates Layer-wise Relevance Propagation (LRP) and SHapley Additive exPlanations (SHAP) to generate explanations for the trained models.

## Usage

To run the entire pipeline, use the `main.py` script with the following command-line arguments:

```bash
python main.py --osn_data <path_or_url_to_osn_data> --network_data <path_or_url_to_network_data>
```

### Command-line Arguments

-   `--osn_data`: Path or URL to the Online Social Network (OSN) dataset (CSV format).
-   `--network_data`: Path or URL to the Network dataset (CSV format).
-   `--model_type` (optional): The model to explain (e.g., "DL", "SVC", "RandomForest"). Defaults to `RandomForest`.
-   `--xai_method` (optional): The XAI method to use ("LRP" or "SHAP"). Defaults to `SHAP`.

### Example

```bash
python main.py --osn_data Train_data.csv --network_data Train_data.csv --model_type RandomForest --xai_method SHAP
```

## Contributing

If you would like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the MIT License.

## Acknowledgments

- This research is supported by [funding sources will be place here].
- Special thanks to [names to be placed here] for their invaluable contributions to this research.
