import os
import sys
import yaml
import argparse
import wandb
import numpy as np
import pandas as pd
from torchvision import transforms
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# Import custom modules
sys.path.append(os.getcwd())
from preprocess.pseudo_labels import add_pseudo_labels  # noqa
from preprocess.preprocess import CAT_COLS, CONT_COLS, preprocess_for_nn, resize_with_padding  # noqa
from models.data_loader import get_dataloader  # noqa
from models.model_utils import load_trained_tabnet  # noqa
from models.lightning_model import RealEstateModel  # noqa


if __name__ == "__main__":
    """
    Main script for training a multi-modal neural network (MMNN) on real estate data.
    The script performs the following steps:
    1. Parse command-line arguments and load configurations.
    2. Load tabular and image data.
    3. Preprocess data for training.
    4. Train the model with PyTorch Lightning.
    5. Save predictions and the trained model to WandB.
    """

    # 1. Configuration and environment setup
    # Define directories for config files and data
    config_file_dir = os.environ.get("CONFIG_FILE_DIR", f"{os.getcwd()}/pipelines/pipe_mmnn")
    data_file_dir = os.environ.get("DATA_FILE_DIR", f"{os.getcwd()}/data")

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config-path', type=str, help='', default=f'{config_file_dir}/config.yml')
    parser.add_argument('--run-name', type=str, help='', default=None)

    args = vars(parser.parse_args())

    # Load configuration from the YAML file
    with open(args['config_path']) as file_:
        # Load YAML file to a Python dictionary
        default_config = yaml.load(file_, Loader=yaml.FullLoader)

    # Prepare WandB configuration by merging necessary parameters
    wandb_config = {key: value for key, value in default_config.items() if key != 'model_params'}
    wandb_config.update(default_config['model_params'])
    wandb_config.update(default_config['embed_dims'])

    # Initialize WandB logging
    wandb.login()
    run = wandb.init(
        project='ILB',  # WandB project name
        tags=default_config['tags'],  # Tags for the run
        name=args['run_name'],  # Custom run name if provided
        config=wandb_config  # Log configuration to WandB
    )
    run_name = run.name
    config = run.config  # Access configuration parameters

    # 2. Load data
    # Load tabular data
    if config.get('debug'):
        # Use a smaller dataset for debugging
        X_train = pd.read_csv(f"{data_file_dir}/X_train_J01Z4CN.csv").sample(frac=0.01)
        y_train = pd.read_csv(f"{data_file_dir}/y_train_OXxrJt1.csv").iloc[X_train.index]
        X_test = pd.read_csv(f"{data_file_dir}/X_test_BEhvxAN.csv").sample(frac=0.01)
        y_random = pd.read_csv(f"{data_file_dir}/y_random_MhJDhKK.csv").iloc[X_test.index]
    else:
        # Load full dataset for training and testing
        X_train = pd.read_csv(f"{data_file_dir}/X_train_J01Z4CN.csv")
        y_train = pd.read_csv(f"{data_file_dir}/y_train_OXxrJt1.csv")
        X_test = pd.read_csv(f"{data_file_dir}/X_test_BEhvxAN.csv")
        y_random = pd.read_csv(f"{data_file_dir}/y_random_MhJDhKK.csv")

    # 3. Preprocess data
    # Preprocess tabular data
    X_train, y_train, X_valid, y_valid, X_test, categorical_dims = preprocess_for_nn(
        X_train,
        y_train,
        X_test,
        valid_size=config.get('valid_size'),
        quantile_transform=config.get('quantile_transform'),
        n_quantiles=config.get('n_quantiles'),
        clip_rooms=config.get('clip_rooms')
        )

    # Prepare categorical columns
    cat_idxs = list(range(len(CAT_COLS)))
    cat_dims = [categorical_dims[f] for f in CAT_COLS]
    embed_dims = {key.split('_embed_dim')[0]: value for key, value in config.items() if key.endswith('embed_dim')}
    cat_emb_dim = [embed_dims[f] for f in CAT_COLS]
    cols = CAT_COLS + CONT_COLS

    # Add pseudo labels to the dataset
    X_train, y_train = add_pseudo_labels(X_train, X_test, y_train, run, config)

    # Create image transformation pipeline
    if config.get('target_size') > 0:
        im_size = (config.get('target_size'), config.get('target_size'))
        transform = transforms.Lambda(lambda x: resize_with_padding(x, config.get('target_size')))
    else:
        im_size = config.get('im_size')
        transform = transforms.Resize(config.get('im_size'))

    # Apply optional data augmentation for images
    if config.get('image_aug'):
        color_jitter_transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        transform = transforms.Compose([
            transform,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply([color_jitter_transform], p=0.5),
        ])

    # Finalize the transformation pipeline
    transform = transforms.Compose([transform, transforms.ToTensor()])
    if config.get('norm_image'):
        transform = transforms.Compose([
            transform,  # Add the previous transform here
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Add normalization transform
        ])

    # Apply log transformation to the target variable
    target_train = y_train.price.apply(np.log)
    target_valid = y_valid.price.apply(np.log)

    # Create PyTorch DataLoaders for training, validation, and testing
    dataloader_train = get_dataloader(
        X_train, cols, target_train, True, 'train', im_size=im_size,
        transform=transform, batch_size=config.get('batch_size'), num_workers=config.get('num_workers')
        )
    dataloader_valid = get_dataloader(
        X_valid, cols, target_valid, False, 'train', im_size=im_size,
        transform=transform, batch_size=config.get('batch_size'), num_workers=config.get('num_workers')
        )
    dataloader_test = get_dataloader(
        X_test, cols, X_test.id_annonce, False, 'test', im_size=im_size,
        transform=transform, batch_size=config.get('predict_batch_size'), num_workers=config.get('num_workers')
        )

    # Determine the device to use for training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. Train the model with lightning
    # Load a pre-trained TabNet model if specified in the configuration
    if config.get('frozen_pretrained_tabnet'):
        pretrained_tabnet = load_trained_tabnet(run, version=config.get('tabnet_version'), freeze=True, device=device)
    elif config.get('embed_frozen_pretrained_tabnet'):
        pretrained_tabnet = load_trained_tabnet(
            run, version=config.get('tabnet_version'),
            freeze=False,
            freeze_embed=True,
            device=device
            )
    else:
        pretrained_tabnet = None

    # Configure the learning rate scheduler if cyclic learning rate is enabled
    scheduler_params = {
        "max_lr": config.get('max_lr'),
        "steps_per_epoch": int(X_train.shape[0] / config.get('batch_size')) + 1,
        "epochs": config.get('max_epochs'),
        "anneal_strategy": "cos"
    } if config.get('cycle_lr') else {}

    # Define the model parameters
    required_keys = [
        'image_model_name', 'hidden_size', 'last_layer_size', 'n_heads', 'n_layers',
        'transformer_attention', 'lr', 'weight_decay', 'lr_factor', 'lr_patience',
        'pretrain', 'mid_level_layer', 'dropout', 'loss_name', 'batch_norm', 'clip_grad'
    ]
    model_params = {key: config.get(key) for key in required_keys}

    # Add additional keys that aren't directly from config
    model_params.update({
        'tabular_input_size': len(X_train.columns) - 1,
        'im_size': im_size,
        'cat_idxs': cat_idxs,
        'cat_dims': cat_dims,
        'cat_emb_dim': cat_emb_dim,
        'pretrained_tabnet': pretrained_tabnet,
        'scheduler_params': scheduler_params,
    })

    # Initialize the model
    model = RealEstateModel(**model_params)

    # Set up PyTorch Lightning Trainer with callbacks
    wandb_logger = WandbLogger(log_model="all")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="valid_mape", mode="min")
    early_stopping = EarlyStopping(monitor="valid_mape", mode="min", patience=config.get('patience'))

    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=config.get('max_epochs'),
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=wandb_logger,
        callbacks=[early_stopping, checkpoint_callback]
    )

    # Train the model
    trainer.fit(model, dataloader_train, val_dataloaders=dataloader_valid)

    # Reload the best model weights
    print('Reloading best weights...')
    best_model = RealEstateModel.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            **model_params
            )

    # y_pred_train = np.exp(clf.predict(X_train[cols].values))
    y_pred_valid = np.exp(np.concatenate(trainer.predict(best_model, dataloaders=dataloader_valid))).flatten()

    # 5. Save outputs
    if config.get('save'):
        predictions = trainer.predict(best_model, dataloaders=dataloader_test)
        y_random['price'] = np.exp(np.concatenate(predictions)).flatten()
        y_random.to_csv(f'{data_file_dir}/submission.csv', index=False)

        artifact = wandb.Artifact(name="submission", type="test predictions")
        artifact.add_file(local_path=f'{data_file_dir}/submission.csv')
        run.log_artifact(artifact)

        artifact = wandb.Artifact(name="mmnn_model", type="model")
        artifact.add_file(local_path=checkpoint_callback.best_model_path)
        run.log_artifact(artifact)

    best_checkpoint = checkpoint_callback.best_model_path
    best_epoch = int(best_checkpoint.split('-')[0].split("=")[1])

    run.log(
        {
            # 'MAPE_train': MAPE(y_train.price, y_pred_train),
            'MAPE_valid': MAPE(y_valid.price, y_pred_valid),
            'best_epoch': best_epoch
        }
    )

    run.finish()
