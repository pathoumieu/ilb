import os, yaml, argparse, sys
import wandb
import numpy as np
import pandas as pd
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.getcwd())
from utils import CAT_COLS, CONT_COLS, preprocess
from utils_torch import RealEstateModel, get_dataloader, load_trained_tabnet


if __name__ == "__main__":

    cfd = os.environ.get("CONFIG_FILE_DIR", f"{os.getcwd()}/pipe_mmnn")
    dfd = os.environ.get("DATA_FILE_DIR", f"{os.getcwd()}/data")

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config-path', type=str, help='', default=f'{cfd}/config.yml')
    parser.add_argument('--run-name', type=str, help='', default=None)

    args = vars(parser.parse_args())

    # Get config and params
    with open(args['config_path']) as file_:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        default_config = yaml.load(file_, Loader=yaml.FullLoader)

    wandb_config = {key: value for key, value in default_config.items() if key != 'model_params'}

    wandb_config.update(default_config['model_params'])

    wandb.login()

    run = wandb.init(
        # set the wandb project where this run will be logged
        project='ILB',
        tags=default_config['tags'],
        name=args['run_name'],
        # track hyperparameters and run metadata
        config=wandb_config
    )
    run_name = run.name
    config = run.config

    # Load the tabular data
    if config.get('debug'):
        X_train = pd.read_csv(f"{dfd}/X_train_J01Z4CN.csv").sample(frac=0.01)
        y_train = pd.read_csv(f"{dfd}/y_train_OXxrJt1.csv").iloc[X_train.index]
        X_test = pd.read_csv(f"{dfd}/X_test_BEhvxAN.csv").sample(frac=0.01)
        y_random = pd.read_csv(f"{dfd}/y_random_MhJDhKK.csv").iloc[X_test.index]
    else:
        X_train = pd.read_csv(f"{dfd}/X_train_J01Z4CN.csv")
        y_train = pd.read_csv(f"{dfd}/y_train_OXxrJt1.csv")
        X_test = pd.read_csv(f"{dfd}/X_test_BEhvxAN.csv")
        y_random = pd.read_csv(f"{dfd}/y_random_MhJDhKK.csv")

    # Preprocess
    X_train, y_train, X_valid, y_valid, X_test, categorical_dims = preprocess(
        X_train,
        y_train,
        X_test,
        valid_size=config.get('valid_size'),
        quantile_transform=config.get('quantile_transform'),
        n_quantiles=config.get('n_quantiles'),
        clip_rooms=config.get('clip_rooms')
        )
    cat_idxs = list(range(len(CAT_COLS)))
    cat_dims = [categorical_dims[f] for f in CAT_COLS]
    cat_emb_dim = [max(1, int(categorical_dims[f] / config.get('embed_scale'))) for f in CAT_COLS]
    cols = CAT_COLS + CONT_COLS

    # Assuming you have X_train, y_train, and the image folder directory
    # Create transforms for image processing
    transform = transforms.Compose([
        transforms.Resize(config.get('im_size')),
        transforms.ToTensor()
    ])

    target_train = y_train.price.apply(np.log)
    target_valid = y_valid.price.apply(np.log)

    # Create datasets and dataloaders
    dataloader_train = get_dataloader(
        X_train, cols, target_train, True, 'train', im_size=config.get('im_size'),
        transform=transform, batch_size=config.get('batch_size'), num_workers=config.get('num_workers')
        )
    dataloader_valid = get_dataloader(
        X_valid, cols, target_valid, False, 'train', im_size=config.get('im_size'),
        transform=transform, batch_size=config.get('batch_size'), num_workers=config.get('num_workers')
        )
    dataloader_test = get_dataloader(
        X_test, cols, X_test.id_annonce, False, 'test', im_size=config.get('im_size'),
        transform=transform, batch_size=config.get('predict_batch_size'), num_workers=config.get('num_workers')
        )
    # Create the model
    if config.get('frozen_pretrained_tabnet'):
        pretrained_tabnet = load_trained_tabnet(run, freeze=True)
    elif config.get('embed_frozen_pretrained_tabnet'):
        pretrained_tabnet = load_trained_tabnet(run, freeze=False, freeze_embed=True)
    else:
        pretrained_tabnet = None

    model = RealEstateModel(
        tabular_input_size=len(X_train.columns) - 1,
        im_size=config.get('im_size'),
        hidden_size=config.get('hidden_size'),
        lr=config.get('lr'),
        weight_decay=config.get('weight_decay'),
        lr_factor=config.get('lr_factor'),
        lr_patience=config.get('lr_patience'),
        pretrain=config.get('pretrain'),
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        pretrained_tabnet=pretrained_tabnet
        )

    # Initialize a trainer
    wandb_logger = WandbLogger(log_model="all")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="valid_mae", mode="min")

    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs'),
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="valid_mae", mode="min", patience=config.get('patience')),
            checkpoint_callback
            ]
    )

    # Train the model
    trainer.fit(model, dataloader_train, val_dataloaders=dataloader_valid)

    print('Reloading best weights...')
    best_model = RealEstateModel.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            tabular_input_size=len(X_train.columns) - 1,
            im_size=config.get('im_size'),
            hidden_size=config.get('hidden_size'),
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            pretrained_tabnet=pretrained_tabnet
            )

    # y_pred_train = np.exp(clf.predict(X_train[cols].values))
    y_pred_valid = np.exp(np.concatenate(trainer.predict(best_model, dataloaders=dataloader_valid))).flatten()

    if config.get('save'):
        predictions = trainer.predict(best_model, dataloaders=dataloader_test)
        y_random['price'] = np.exp(np.concatenate(predictions)).flatten()
        y_random.to_csv(f'{dfd}/submission.csv', index=False)

        artifact = wandb.Artifact(name="submission", type="test predictions")
        artifact.add_file(local_path=f'{dfd}/submission.csv')
        run.log_artifact(artifact)

        artifact = wandb.Artifact(name="mmnn_model", type="model")
        artifact.add_file(local_path=checkpoint_callback.best_model_path)
        run.log_artifact(artifact)

    run.log(
        {
            # 'MAPE_train': MAPE(y_train.price, y_pred_train),
            'MAPE_valid': MAPE(y_valid.price, y_pred_valid)
        }
    )

    run.finish()
