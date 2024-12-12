# Multi-Modal Neural Network Project

This repository contains the implementation of a multi-modal neural network designed for real estate data processing. It includes data preprocessing, training pipelines for various models, and evaluation tools. The project leverages Docker for reproducibility and scalability and integrates with Weights & Biases (W&B) for experiment tracking.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Running Pipelines](#running-pipelines)
  - [Experiment Tracking](#experiment-tracking)
- [Approach](#approach)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
.
├── Makefile                   # Automation of build, push, and run processes
├── data                       # Contains dataset files (CSV format)
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

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-modal-nn.git
   cd multi-modal-nn
   ```

2. Install required Python dependencies (optional if not using Docker):
   ```bash
   pip install -r pipelines/pipe_ilb/requirements.txt
   pip install -r pipelines/pipe_tabnet/requirements.txt
   # Repeat for other pipelines as needed
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory:
   ```bash
   WANDB_API_KEY=your_wandb_api_key
   WANDB_PROJECT=your_project_name
   WANDB_ENTITY=your_entity_name
   ```

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
make build-tabnet
make push-tabnet
make run-tabnet
```

#### Run Pipeline with Custom Environment Variables
```bash
make job-tabnet
```

### Experiment Tracking
The project integrates with W&B for logging metrics, losses, and visualizations. Ensure your environment variables are correctly set up in the `.env` file.

To run a sweep for hyperparameter optimization (example for TabNet):
```bash
wandb sweep pipelines/pipe_tabnet/sweep_tabnet.yml
wandb agent <sweep_id>
```

## Approach

This project uses a multi-modal approach to leverage both tabular and image data for real estate predictions. Key components:

1. **TabNet**: For tabular data modeling.
2. **Image Processing**: Pretrained image models for extracting features.
3. **Multi-modal Neural Network**: Combines features from tabular and image data.

### Steps:

1. **Preprocessing**: Includes resizing images, extracting features, and generating pseudo-labels.
2. **Model Training**: 
   - Train individual pipelines like TabNet or ILB.
   - Combine modalities in a unified multi-modal neural network.
3. **Evaluation**: Logs metrics and results to W&B for analysis.

## Contributing

Contributions are welcome! Please submit a pull request or create an issue if you encounter any problems or have suggestions for improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
