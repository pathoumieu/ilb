import os, yaml, argparse, sys
import wandb
import numpy as np
import pandas as pd
from torchvision import transforms
import pytorch_lightning as pl

sys.path.append(os.getcwd())
from utils import CAT_COLS, CONT_COLS, preprocess
from utils_torch import RealEstateModel, get_dataloader

IMG_SIZE = (128, 128)
BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 256
MAX_EPOCHS = 1
HIDDEN_SIZE = 8
DEBUG = True


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
    if DEBUG:
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
    X_train, y_train, X_valid, y_valid, X_test, _ = preprocess(
        X_train,
        y_train,
        X_test,
        valid_size=config.get('valid_size'),
        quantile_transform=config.get('quantile_transform'),
        n_quantiles=config.get('n_quantiles'),
        clip_rooms=config.get('clip_rooms')
        )
    cols = CAT_COLS + CONT_COLS

    # Assuming you have X_train, y_train, and the image folder directory
    # Create transforms for image processing
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    target_train = y_train.price.apply(np.log)
    target_valid = y_valid.price.apply(np.log)

    # Create datasets and dataloaders
    dataloader_train = get_dataloader(
        X_train, target_train, True, 'train',
        transform=transform, batch_size=BATCH_SIZE
        )
    dataloader_valid = get_dataloader(
        X_valid, target_valid, False, 'train',
        transform=transform, batch_size=BATCH_SIZE
        )
    dataloader_test = get_dataloader(
        X_test, X_test.id_annonce, False, 'test',
        transform=transform, batch_size=PREDICT_BATCH_SIZE
        )

    # Create the model
    model = RealEstateModel(
        tabular_input_size=len(X_train.columns) - 1,
        im_size=IMG_SIZE,
        hidden_size=HIDDEN_SIZE
        )

    # Initialize a trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train the model
    # max_epochs = config.get('max_epochs')
    # model_params = {
    #     'optimizer_params': {'lr': config.get('lr')},
    #     'gamma': config.get('gamma'),
    #     'n_steps': config.get('n_steps'),
    #     'n_a': config.get('n_a'),
    #     'n_d': config.get('n_a'),
    #     'n_shared': config.get('n_shared'),
    #     'n_independent': config.get('n_shared')
    #     }
    trainer.fit(model, dataloader_train, val_dataloaders=dataloader_valid)

    predictions = trainer.predict(model, dataloaders=dataloader_test)
    y_random['price'] = np.exp(np.concatenate(predictions)).flatten()
    y_random.to_csv(f'{dfd}/submission.csv', index=False)

    # y_pred_train = np.exp(clf.predict(X_train[cols].values))
    # y_pred_valid = np.exp(clf.predict(X_valid[cols].values))

    artifact = wandb.Artifact(name="submission", type="test predictions")
    artifact.add_file(local_path=f'{dfd}/submission.csv')
    run.log_artifact(artifact)

    # saving_path_name = f"{dfd}/tabnet_model.pt"
    # saved_filepath = clf.save_model(saving_path_name)
    # artifact = wandb.Artifact(name="tabnet_model", type="model")
    # artifact.add_file(local_path=f'{dfd}/tabnet_model.pt.zip')
    # run.log_artifact(artifact)

    # run.log(
    #     {
    #         'MAPE_train': MAPE(y_train.price, y_pred_train),
    #         'MAPE_valid': MAPE(y_valid.price, y_pred_valid)
    #     }
    # )

    run.finish()
