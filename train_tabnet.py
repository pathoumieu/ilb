import os, yaml, argparse
import wandb
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from utils import CAT_COLS, CONT_COLS, preprocess
from utils_torch import WandbCallback


if __name__ == "__main__":

    cfd = os.environ.get("CONFIG_FILE_DIR", os.getcwd())
    dfd = os.environ.get("DATA_FILE_DIR", f"{os.getcwd()}/data")

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config-path', type=str, help='', default=f'{cfd}/config_tabnet.yml')
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
    print(config.get('n_quantiles'))
    cat_idxs = list(range(len(CAT_COLS)))
    cat_dims = [categorical_dims[f] for f in CAT_COLS]
    cat_emb_dim = [max(1, int(categorical_dims[f] / config.get('embed_scale'))) for f in CAT_COLS]
    cols = CAT_COLS + CONT_COLS

    # Train
    max_epochs = config.get('max_epochs')
    model_params = {
        'optimizer_params': {'lr': config.get('lr'), 'weight_decay': config.get('lr')},
        'gamma': config.get('gamma'),
        'n_steps': config.get('n_steps'),
        'n_a': config.get('n_a'),
        'n_d': config.get('n_a'),
        'n_shared': config.get('n_shared'),
        'n_independent': config.get('n_shared')
        }

    clf = TabNetRegressor(
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        cat_idxs=cat_idxs,
        optimizer_fn=torch.optim.Adam,
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params={
            "mode": 'min', # max because default eval metric for binary is AUC
            "factor": config.get('lr_factor'),
            "patience": config.get('lr_patience')
            },
        **model_params
        )

    target_train = np.log(y_train.price.values.reshape(-1, 1))
    target_valid = np.log(y_valid.price.values.reshape(-1, 1))

    clf.fit(
        X_train=X_train[cols].values, y_train=target_train,
        eval_set=[(X_train[cols].values, target_train), (X_valid[cols].values, target_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['mae'],
        max_epochs=config.get('max_epochs'),
        patience=config.get('patience'),
        batch_size=config.get('batch_size'),
        virtual_batch_size=config.get('virtual_batch_size'),
        num_workers=0,
        drop_last=False,
        callbacks=[WandbCallback()]
    )

    y_pred_train = np.exp(clf.predict(X_train[cols].values))
    y_pred_valid = np.exp(clf.predict(X_valid[cols].values))
    y_pred_test = np.exp(clf.predict(X_test[cols].values))

    y_random['price'] = y_pred_test
    y_random.to_csv(f'{dfd}/submission.csv', index=False)
    artifact = wandb.Artifact(name="submission", type="test predictions")
    artifact.add_file(local_path=f'{dfd}/submission.csv')
    run.log_artifact(artifact)

    saving_path_name = f"{dfd}/tabnet_model.pt"
    saved_filepath = clf.save_model(saving_path_name)
    artifact = wandb.Artifact(name="tabnet_model", type="model")
    artifact.add_file(local_path=f'{dfd}/tabnet_model.pt.zip')
    run.log_artifact(artifact)

    run.log(
        {
            'MAPE_train': MAPE(y_train.price, y_pred_train),
            'MAPE_valid': MAPE(y_valid.price, y_pred_valid)
        }
    )

    run.finish()
