# Multi-Modal Neural Network Project

This repository contains the implementation of a multi-modal neural network designed for real estate price estimation, as part of the challenge hosted by Institut Louis Bachelier (ILB). The goal is to predict French housing prices by combining hierarchical tabular data with images of the assets. This approach aims to assess whether including photos improves prediction accuracy compared to using tabular data alone.

## Table of Contents

- [Challenge Context](#challenge-context)
- [Performance](#performance)
- [Approach](#approach)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Running Pipelines](#running-pipelines)
  - [Experiment Tracking](#experiment-tracking)

## Challenge Context

This project was developed for the [ILB DataLab Real Estate Price Challenge](https://challengedata.ens.fr/participants/challenges/68/). The challenge involves estimating housing prices based on:

1. **Tabular data**: Features such as location, surface area, number of rooms, etc.
2. **Image data**: Between 1 to 6 photos of each property.

The aim is to explore whether incorporating images alongside tabular data leads to improved price predictions. Emphasis is placed on interpretability to understand the contribution of different features.

## Performance

As of the latest submission, the best solution achieved **5th place** out of 182 participants in the ILB Challenge leaderboard.

## Approach

This project uses a multi-modal approach to leverage both tabular and image data for real estate predictions. Key components:

1. **TabNet**: For tabular data modeling.
2. **Image Processing**: Pretrained image models for extracting features.
3. **Multi-modal Neural Network**: Combines features from tabular and image data. (see [lightning_model.py](models/lightning_model.py) and ``pipe_mmnn`` pipeline)
3. **Comparison to regular ML approach**: Combines features from tabular and image data. (see `pipe_ilb` pipeline and [train_ilb.py](pipelines/pipe_ilb/train_ilb.py))

### Multimodal End-to-End Approach

This project uses a custom **multimodal neural network model** implemented in PyTorch Lightning to predict real estate prices. The model integrates both **tabular data** (e.g., property attributes) and **image data** (e.g., property photos) into a unified framework, leveraging the strengths of each data modality to achieve improved prediction accuracy. 

The benefits of this approach include: 
- Combination structured (tabular) and unstructured (image) data for improved predictive accuracy.
- Use of attention mechanisms / transformer blocks to prioritize relevant images from a varying number for each property.
- Leveraging TabNet for interpretability and MobileNet for robust image feature extraction.
- Scaling efficiently to diverse datasets with modular pipelines.

Below is an overview of the approach:

#### Key Components

1. **Tabular Data Processing**
   - The **TabNet** model is used to process tabular data. TabNet utilizes attention mechanisms to focus on the most relevant features for prediction.
   - If a pretrained TabNet model is available, it can be directly integrated into the framework. Otherwise, a new TabNet model is instantiated.

2. **Image Data Processing**
   - Images are processed using a pre-trained or custom convolutional model. The default configuration uses a **MobileNet** backbone.
   - The model extracts mid-level features from the images, which are then projected into a lower-dimensional space for integration with tabular features.
   - For each property, multiple images (up to six) are supported. 

3. **Attention Mechanism for Image Features**
   - The image features are further refined using an attention mechanism to focus on the most critical aspects of the images. Options include:
     - **Transformer-based attention**
     - **Gated attention mechanism** (default)

4. **Feature Fusion**
   - Features from the tabular and image data are concatenated into a single vector.
   - The fused features are passed through fully connected layers to generate the final price prediction.

5. **Loss Function and Metrics**
   - The model supports both **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)** loss functions.
   - An additional evaluation metric, **Mean Absolute Percentage Error (MAPE)**, is logged during training and validation.

6. **Training and Optimization**
   - The model is optimized using the Adam optimizer, with configurable learning rates and weight decay.
   - Gradient clipping is applied to prevent instability during training.
   - A learning rate scheduler (either `ReduceLROnPlateau` or `OneCycleLR`) is used to dynamically adjust the learning rate based on validation performance.

For further details about the implementation, refer to the file [lightning_model.py](models/lightning_model.py).


### Comparison to regular ML approach (ILB pipeline)

The ILB pipeline is a comparative model leveraging CatBoost with tabular features and optional integration of image-derived features. This pipeline is designed to assess performance relative to the multimodal approach.

#### Overview of the ILB Pipeline

1. **Tabular Data Enrichment and Preprocessing**  
   - Tabular data is loaded and enriched with engineered features, including categorical and continuous attributes.
   - Preprocessing steps include scaling, quantile transformations, and optional clipping of outliers.

2. **Image Feature Integration (Optional)**  
   - Image features are pre-computed as the output of the last layer of a pre-trained CNN (e.g., ResNet, EfficientNet, MobileNet). These are averaged across all images per property.
   - Features are retrieved as artifacts and merged into the tabular datasets. 
   - Dimensionality reduction can be applied to image features:
     - **PCA**: Principal Component Analysis for linear feature reduction.
     - **UMAP**: Non-linear dimensionality reduction for capturing complex relationships.

3. **Pseudo-Labeling for Data Augmentation**  
   - A pseudo-labeling step enriches the training dataset by predicting labels for test samples and integrating them into training.

4. **Modeling with CatBoost**  
   - The primary model is **CatBoostRegressor**, configured for regression tasks and fine-tuned with cross-validation.
   - Targets are log-transformed during training to better handle skewed price distributions and are inversely transformed during predictions.

5. **Cross-Validation Ensemble**  
   - The pipeline uses a custom [cross-validation ensemble](https://github.com/pathoumieu/crossval-ensemble) to stabilize predictions and reduce variance.

6. **Evaluation Metrics**  
   - Model performance is evaluated using **Mean Absolute Percentage Error (MAPE)** on training and validation datasets.

7. **Output Generation**  
   - Final predictions on the test dataset are saved and logged as artifacts in Weights & Biases (wandb) for analysis and submission.

#### Key Features

- **Modularity**: Easy integration of additional features such as image data or engineered attributes.
- **Advanced Dimensionality Reduction**: PCA and UMAP allow for customizable feature compression.
- **CatBoost Optimization**: Leveraging the best practices for boosting algorithms, including early stopping and robust parameterization.
- **Pseudo-Labeling**: A semi-supervised learning technique to boost model performance with additional labeled data.

This script complements the multimodal approach, providing a robust baseline model for comparison. You can find the source code for this pipeline in the provided script file [train_ilb.py](pipelines/pipe_ilb/train_ilb.py).


## Project Structure

```
.
├── Makefile                   # Automation of build, push, and run processes
├── data                       # Contains dataset files (to be downloaded by the user)
├── models                     # Model definitions and utilities
│   ├── __init__.py
│   ├── base_model.py          # Base class for models
│   ├── data_loader.py         # Data loader utilities
│   ├── lightning_model.py     # PyTorch Lightning model definition
│   └── model_utils.py         # Helper functions for models
├── notebooks                  # Jupyter notebooks for profiling and analysis
│   └── profiling.html
├── pipelines                  # Training pipelines for various models
│   ├── pipe_gpu_mmnn          # GPU-based multi-modal neural network pipeline
│   │   └── Dockerfile
│   ├── pipe_ilb               # ILB training pipeline
│   │   ├── Dockerfile
│   │   ├── config.yml
│   │   ├── requirements.txt
│   │   ├── sweep.yml
│   │   └── train_ilb.py
│   ├── pipe_mmnn              # Multi-modal neural network pipeline
│   │   ├── Dockerfile
│   │   ├── config.yml
│   │   ├── requirements.txt
│   │   ├── sweep_mmnn.yml
│   │   └── train_mmnn.py
│   ├── pipe_tabnet           # TabNet training pipeline
│   │   ├── Dockerfile
│   │   ├── config.yml
│   │   ├── requirements.txt
│   │   ├── sweep_tabnet.yml
│   │   └── train_tabnet.py
│   └── pipe_tabnet_pretraining # Pretraining for TabNet
│       ├── Dockerfile
│       ├── config.yml
│       └── train_tabnet_pretraining.py
├── preprocess                 # Data preprocessing scripts
│   ├── __init__.py
│   ├── preprocess.py
│   └── pseudo_labels.py
└── scripts                    # Miscellaneous utility scripts
    ├── __init__.py
    ├── config_image_features.yml
    ├── explo_image_size.py
    ├── extract_image_features.py
    └── resize_images.py
```

## Prerequisites

- **Docker**: Ensure Docker is installed and running on your machine.
- **Python**: Python 3.10 or higher is recommended.
- **Weights & Biases (W&B)**: For experiment tracking, set up a W&B account and obtain an API key.
- **Data Directory**: Create a `data` directory in the root folder and download the dataset files into this directory.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pathoumieu/ilb
   cd multi-modal-nn
   ```

2. Create a `data` directory at the root and put the downloaded files from the challenge in it.

3. Set up environment variables:
   Create a `.env` file in the root directory:
   ```bash
   WANDB_API_KEY=your_wandb_api_key
   WANDB_PROJECT=your_project_name
   WANDB_ENTITY=your_entity_name
   ```
   You can use the `.env.dist` file template for creation.

## Usage

### Running Pipelines

The `Makefile` provides commands to build, push, and run Docker containers for each pipeline.

#### Build and Run All Pipelines
```bash
make all
```

#### Build, Push, and Run Specific Pipeline
For example, to run the `pipe_tabnet` pipeline:
```bash
make tabnet
```

### Experiment Tracking
The project integrates with W&B for logging metrics, losses, and visualizations. Ensure your environment variables are correctly set up in the `.env` file.

To run a sweep for hyperparameter optimization (example for TabNet):
```bash
wandb sweep pipelines/pipe_tabnet/sweep_tabnet.yml
wandb agent <sweep_id>
```