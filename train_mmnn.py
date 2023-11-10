import os, yaml, argparse
import wandb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from utils import CAT_COLS, CONT_COLS, RealEstateDataset, RealEstateModel

IMG_SIZE = (128, 128)
BATCH_SIZE = 512
MAX_EPOCHS = 1


if __name__ == "__main__":

    cfd = os.environ.get("CONFIG_FILE_DIR", os.getcwd())
    dfd = os.environ.get("DATA_FILE_DIR", f"{os.getcwd()}/data")

    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--config-path', type=str, help='', default=f'{cfd}/config_tabnet.yml')
    # parser.add_argument('--run-name', type=str, help='', default=None)

    # args = vars(parser.parse_args())

    # # Get config and params
    # with open(args['config_path']) as file_:
    #     # The FullLoader parameter handles the conversion from YAML
    #     # scalar values to Python the dictionary format
    #     default_config = yaml.load(file_, Loader=yaml.FullLoader)

    # wandb_config = {'valid_size': default_config['valid_size']}
    # wandb_config.update(default_config['model_params'])

    # wandb.login()

    # run = wandb.init(
    #     # set the wandb project where this run will be logged
    #     project='ILB',
    #     tags=default_config['tags'],
    #     name=args['run_name'],
    #     # track hyperparameters and run metadata
    #     config=wandb_config
    # )
    # run_name = run.name
    # config = run.config

    # Load the tabular data
    X_train = pd.read_csv(f"{dfd}/X_train_J01Z4CN.csv").sample(frac=0.1)
    y_train = pd.read_csv(f"{dfd}/y_train_OXxrJt1.csv").iloc[X_train.index]
    X_test = pd.read_csv(f"{dfd}/X_test_BEhvxAN.csv")
    y_random = pd.read_csv(f"{dfd}/y_random_MhJDhKK.csv")

    # Preprocess
    datasets = [X_train, X_test]
    for dataset in datasets:
        dataset['department'] = X_train.postal_code.apply(lambda x: str(x).zfill(5)[:2])
        dataset[CONT_COLS] = dataset[CONT_COLS].fillna(0.0).astype(float)
        dataset[CAT_COLS] = dataset[CAT_COLS].fillna('-1').astype(str)

    valid_size = 0.2
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=0)

    categorical_dims =  {}
    for cat_col in CAT_COLS:

        unknown = X_train[cat_col].nunique()

        oe = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=unknown,
            encoded_missing_value=unknown,
            dtype=int
        )

        X_train[cat_col] = oe.fit_transform(X_train[cat_col].values.reshape(-1, 1))
        X_valid[cat_col] = oe.transform(X_valid[cat_col].values.reshape(-1, 1))
        X_test[cat_col] = oe.transform(X_test[cat_col].values.reshape(-1, 1))
        categorical_dims[cat_col] = len(oe.categories_[0]) + 1

    for cont_col in CONT_COLS:
        std = StandardScaler()
        X_train[cont_col] = std.fit_transform(X_train[cont_col].values.reshape(-1, 1))
        X_valid[cont_col] = std.transform(X_valid[cont_col].values.reshape(-1, 1))
        X_test[cont_col] = std.transform(X_test[cont_col].values.reshape(-1, 1))

    cat_idxs = list(range(len(CAT_COLS)))
    cat_dims = [categorical_dims[f] for f in CAT_COLS]
    cat_emb_dim = [int(np.sqrt(categorical_dims[f])) for f in CAT_COLS]
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
    dataset_train = RealEstateDataset(
        tabular_data=X_train,
        target=target_train,
        image_dir='data/reduced_images_ILB/reduced_images/train',
        transform=transform
        )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True
        )

    dataset_valid = RealEstateDataset(
        tabular_data=X_valid,
        target=target_train,
        image_dir='data/reduced_images_ILB/reduced_images/train',
        transform=transform
        )
    dataloader_valid = DataLoader(
        dataset_valid,
        batch_size=BATCH_SIZE,
        shuffle=True
        )

    dataset_test = RealEstateDataset(
        tabular_data=X_test,
        target=X_test.id_annonce,
        image_dir='data/reduced_images_ILB/reduced_images/test',
        transform=transform
        )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False
        )

    # Create the model
    model = RealEstateModel(
        tabular_input_size=len(X_train.columns) - 1,
        hidden_size=8
        )

    # Initialize a trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train the model
    trainer.fit(model, dataloader_train, val_dataloaders=dataloader_valid)

    predictions = trainer.predict(model, dataloaders=dataloader_test)
    print(predictions)
    # # Train
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

    # clf = TabNetRegressor(
    #     cat_dims=cat_dims,
    #     cat_emb_dim=cat_emb_dim,
    #     cat_idxs=cat_idxs,
    #     optimizer_fn=torch.optim.Adam,
    #     **model_params
    #     )

    # target_train = np.log(y_train.price.values.reshape(-1, 1))
    # target_valid = np.log(y_valid.price.values.reshape(-1, 1))

    # clf.fit(
    #     X_train=X_train[cols].values, y_train=target_train,
    #     eval_set=[(X_train[cols].values, target_train), (X_valid[cols].values, target_valid)],
    #     eval_name=['train', 'valid'],
    #     eval_metric=['mae'],
    #     max_epochs=config.get('max_epochs'),
    #     patience=config.get('patience'),
    #     batch_size=config.get('batch_size'),
    #     virtual_batch_size=config.get('virtual_batch_size'),
    #     num_workers=0,
    #     drop_last=False,
    #     callbacks=[WandbCallback()]
    # )

    # y_pred_train = np.exp(clf.predict(X_train[cols].values))
    # y_pred_valid = np.exp(clf.predict(X_valid[cols].values))
    # y_pred_test = np.exp(clf.predict(X_test[cols].values))

    # y_random['price'] = y_pred_test
    # y_random.to_csv(f'{dfd}/submission.csv', index=False)
    # artifact = wandb.Artifact(name="submission", type="test predictions")
    # artifact.add_file(local_path=f'{dfd}/submission.csv')
    # run.log_artifact(artifact)

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

    # run.finish()