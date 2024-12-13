import os
import sys
import yaml
import argparse
import wandb
import numpy as np
import pandas as pd
import torch
from torch.nn import L1Loss
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE

sys.path.append(os.getcwd())
from preprocess.preprocess import CAT_COLS, CONT_COLS, preprocess_for_nn  # noqa
from models.model_utils import WandbCallback, tabnet_mape  # noqa


def get_embedding_dimensions(cat_cols, categorical_dims, config):
    """
    Calculates the embedding dimensions for categorical features based on
    a scaling factor or explicit dimensions provided in the configuration.

    Parameters
    ----------
    cat_cols : list of str
        List of categorical column names to calculate embedding dimensions for.
    categorical_dims : dict
        A dictionary where keys are categorical feature names and values are the number
        of unique categories for each feature.
    config : dict
        Configuration dictionary containing:
        - `embed_scale` (float): A scaling factor to determine embedding dimensions based
        on the number of unique categories.
        - Explicit embedding dimensions (e.g., 'feature_embed_dim') for specific features.

    Returns
    -------
    list of int
        A list of embedding dimensions corresponding to the input categorical columns.

    Notes
    -----
    - If `embed_scale` is greater than 0, the embedding dimension is calculated as:
      max(1, int(num_categories / embed_scale)).
    - If explicit embedding dimensions are provided in the configuration, they override the scaling factor.

    Example
    -------
    >>> cat_cols = ['feature_a', 'feature_b']
    >>> categorical_dims = {'feature_a': 100, 'feature_b': 50}
    >>> config = {'embed_scale': 10}
    >>> get_embedding_dimensions(cat_cols, categorical_dims, config)
    [10, 5]
    """
    # Check if scaling factor `embed_scale` is provided in the configuration
    if config.get('embed_scale') > 0:
        # Calculate embedding dimensions as num_categories / embed_scale, ensuring a minimum dimension of 1
        return [max(1, int(categorical_dims[f] / config.get('embed_scale'))) for f in cat_cols]

    # Extract explicit embedding dimensions for specific features from the configuration
    embed_dims = {
        key.split('_embed_dim')[0]: value for key, value in config.items() if key.endswith('embed_dim')
    }

    # Return the embedding dimensions for the categorical columns based on the explicit configuration
    return [embed_dims[f] for f in cat_cols]


if __name__ == "__main__":
    # Get paths for configuration and data files from environment variables or default to local directories
    config_file_dir = os.environ.get("CONFIG_FILE_DIR", f"{os.getcwd()}/pipelines/pipe_tabnet")
    data_file_dir = os.environ.get("DATA_FILE_DIR", f"{os.getcwd()}/data")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run TabNet Regression Training and Evaluation')
    parser.add_argument(
        '--config-path',
        type=str,
        help='Path to the configuration file',
        default=f'{config_file_dir}/config.yml'
        )
    parser.add_argument('--run-name', type=str, help='Optional custom name for the run', default=None)
    args = vars(parser.parse_args())

    # Load configuration file
    with open(args['config_path']) as file_:
        default_config = yaml.load(file_, Loader=yaml.FullLoader)

    # Prepare W&B configuration by merging relevant sections of the YAML config
    wandb_config = {key: value for key, value in default_config.items() if key not in ['model_params', 'embed_dims']}
    wandb_config.update(default_config['model_params'])
    wandb_config.update(default_config['embed_dims'])

    # Login to W&B and initialize the run
    wandb.login()
    run = wandb.init(
        project='ILB',
        tags=default_config['tags'],
        name=args['run_name'],
        config=wandb_config
    )
    run_name = run.name
    config = run.config

    # Load the data
    X_train = pd.read_csv(f"{data_file_dir}/X_train_J01Z4CN.csv")
    y_train = pd.read_csv(f"{data_file_dir}/y_train_OXxrJt1.csv")
    X_test = pd.read_csv(f"{data_file_dir}/X_test_BEhvxAN.csv")
    y_random = pd.read_csv(f"{data_file_dir}/y_random_MhJDhKK.csv")

    # Preprocess the data
    # The preprocessing step includes splitting the data, transforming features, and calculating categorical dimensions.
    X_train, y_train, X_valid, y_valid, X_test, categorical_dims = preprocess_for_nn(
        X_train,
        y_train,
        X_test,
        valid_size=config.get('valid_size'),
        quantile_transform=config.get('quantile_transform'),
        n_quantiles=config.get('n_quantiles'),
        clip_rooms=config.get('clip_rooms')
    )

    # Prepare categorical feature information
    cat_idxs = list(range(len(CAT_COLS)))  # Indices of categorical columns
    cat_dims = [categorical_dims[f] for f in CAT_COLS]  # Number of unique categories per feature
    cat_emb_dim = get_embedding_dimensions(CAT_COLS, categorical_dims, config)  # Embedding dimensions

    cols = CAT_COLS + CONT_COLS

    # Configure TabNet model parameters
    model_params = {key: config.get(key) for key in ['gamma', 'n_steps', 'n_a', 'n_shared', 'clip_value']}
    model_params.update({
        'optimizer_params': {'lr': config.get('lr'), 'weight_decay': config.get('weight_decay')},
        'n_d': config.get('n_a'),  # Set number of decision units equal to attention units
        'n_independent': config.get('n_shared')  # Set independent layers equal to shared layers
    })

    # Configure learning rate scheduler
    if config.get('cycle_lr'):
        scheduler_fn = torch.optim.lr_scheduler.OneCycleLR
        scheduler_params = {
            "is_batch_level": True,
            "max_lr": config.get('max_lr'),
            "steps_per_epoch": int(X_train.shape[0] / config.get('batch_size')) + 1,
            "epochs": config.get('max_epochs'),
            "anneal_strategy": "cos"
        }
    else:
        scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler_params = {
            "mode": 'min',
            "factor": config.get('lr_factor'),
            "patience": config.get('lr_patience'),
        }

    # Initialize the TabNetRegressor model
    clf = TabNetRegressor(
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        cat_idxs=cat_idxs,
        optimizer_fn=torch.optim.Adam,
        scheduler_fn=scheduler_fn,
        scheduler_params=scheduler_params,
        **model_params
    )

    # Log-transform target for better handling of price distributions
    target_train = np.log(y_train.price.values.reshape(-1, 1))
    target_valid = np.log(y_valid.price.values.reshape(-1, 1))

    # Train the model
    clf.fit(
        X_train=X_train[cols].values, y_train=target_train,
        eval_set=[(X_train[cols].values, target_train), (X_valid[cols].values, target_valid)],
        loss_fn=L1Loss(),
        eval_name=['train', 'valid'],
        eval_metric=["mae", tabnet_mape],
        max_epochs=config.get('max_epochs'),
        patience=config.get('patience'),
        batch_size=config.get('batch_size'),
        virtual_batch_size=config.get('virtual_batch_size'),
        num_workers=8,
        drop_last=False,
        callbacks=[WandbCallback()]
    )

    # Make predictions and inverse the log transformation
    y_pred_train = np.exp(clf.predict(X_train[cols].values))
    y_pred_valid = np.exp(clf.predict(X_valid[cols].values))
    y_pred_test = np.exp(clf.predict(X_test[cols].values))

    # Save predictions and artifacts
    y_random['price'] = y_pred_test
    y_random.to_csv(f'{data_file_dir}/submission.csv', index=False)
    artifact = wandb.Artifact(name="submission", type="test predictions")
    artifact.add_file(local_path=f'{data_file_dir}/submission.csv')
    run.log_artifact(artifact)

    # Save the trained model
    saving_path_name = f"{data_file_dir}/tabnet_model.pt"
    clf.save_model(saving_path_name)
    artifact = wandb.Artifact(name="tabnet_model", type="model")
    artifact.add_file(local_path=f'{data_file_dir}/tabnet_model.pt.zip')
    run.log_artifact(artifact)

    # Log metrics
    run.log(
        {
            'MAPE_train': MAPE(y_train.price, y_pred_train),
            'MAPE_valid': MAPE(y_valid.price, y_pred_valid)
        }
    )

    # Finish the W&B run
    run.finish()
