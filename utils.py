import os
import wandb
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_tabnet.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn_pandas import gen_features, DataFrameMapper


CAT_COLS = [
    'property_type',
    'city',
    'postal_code',
    'energy_performance_category',
    'ghg_category',
    'exposition',
    'has_a_balcony',
    'has_a_cellar',
    'has_a_garage',
    'has_air_conditioning',
    'last_floor',
    'upper_floors',
    'department'
]

CONT_COLS = [
    'approximate_latitude',
    'approximate_longitude',
    'size',
    'floor',
    'land_size',
    'energy_performance_value',
    'ghg_value',
    'nb_rooms',
    'nb_bedrooms',
    'nb_bathrooms',
    'nb_parking_places',
    'nb_boxes',
    'nb_photos',
    'nb_terraces',
]


def create_preprocessor(cont_cols, cat_cols):
    cat_cols_list = [[cat_col] for cat_col in cat_cols]
    cont_cols_list = [[cont_col] for cont_col in cont_cols]

    gen_numeric = gen_features(
        columns=cont_cols_list,
        classes=[
            {
                "class": SimpleImputer,
                "strategy": "constant",
                "fill_value": 0.0
            },
            {
                "class": StandardScaler
            }
        ]
    )

    gen_categories = gen_features(
        columns=cat_cols_list,
        classes=[
            {
                "class": SimpleImputer,
                "strategy": "constant",
                "fill_value": -1
            },
            {
                "class": OrdinalEncoder,
                "handle_unknown": 'use_encoded_value',
                "unknown_value": -1,
                "encoded_missing_value": -1,
                "dtype": int
            }
        ]
    )

    # DataFrameMapper construction
    preprocess_mapper = DataFrameMapper(
        [*gen_numeric, *gen_categories],
        input_df=True,
        df_out=True
    )

    return preprocess_mapper


def prepare_datasets(X_train, X_test):

    datasets = [X_train, X_test]
    for dataset in datasets:
        dataset['department'] = dataset['postal_code'].apply(lambda x: str(x).zfill(5)[:2])
        dataset[['energy_performance_value', 'ghg_value']] = dataset[
                ['energy_performance_value', 'ghg_value']
                ].fillna(-1.0).astype(float)

        cont_cols_prep = [col for col in CONT_COLS if col not in ['energy_performance_value', 'ghg_value']]
        dataset[cont_cols_prep] = dataset[cont_cols_prep].fillna(0.0).astype(float)

        dataset[CAT_COLS] = dataset[CAT_COLS].fillna('-1').astype(str)

    return X_train, X_test


def preprocess(X_train, X_test, valid_size=0.2, random_state=0):

    X_train, X_test = prepare_datasets(X_train, X_test)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=valid_size,
        random_state=random_state
        )

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

    return X_train, y_train, X_valid, y_valid, X_test, categorical_dims


class WandbCallback(Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every SuperModule.

    Parameters
    ---------
    trainer : DeepRecoModel
        Model class to train
    verbose : int
        Print results every verbose iteration

    """
    def __post_init__(self):
        super().__init__()
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        wandb.log({"train_mae": logs['train_mae']})
        wandb.log({"valid_mae": logs['valid_mae']})
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class RealEstateDataset(Dataset):
    def __init__(self, tabular_data, target, image_dir, transform=None, max_images=6):
        self.tabular = tabular_data
        self.target = target
        self.image_dir = image_dir
        self.transform = transform
        self.max_images = max_images

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        tabular_features = self.tabular.drop(columns=['id_annonce']).iloc[idx].values.tolist()
        targets = self.target.iloc[idx].tolist()

        image_folder = os.path.join(self.image_dir, 'ann_' + str(int(self.tabular.iloc[idx]['id_annonce'])))
        image_paths = [os.path.join(image_folder, image) for image in os.listdir(image_folder)]

        images = []
        for i in range(self.max_images):
            if i < len(image_paths):
                image = Image.open(image_paths[i]).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            else:
                # Pad with zeros if fewer images exist
                images.append(torch.zeros((3, 128, 128)))

        tabular_features = torch.tensor(tabular_features, dtype=torch.float)
        images = torch.stack(images)
        targets = torch.tensor(targets, dtype=torch.float)

        return tabular_features, images, targets


class RealEstateTestDataset(Dataset):
    def __init__(self, tabular_data, image_dir, transform=None, max_images=6):
        self.tabular = tabular_data
        self.image_dir = image_dir
        self.transform = transform
        self.max_images = max_images

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        tabular_features = self.tabular.drop(columns=['id_annonce']).iloc[idx].values.tolist()

        image_folder = os.path.join(self.image_dir, 'ann_' + str(int(self.tabular.iloc[idx]['id_annonce'])))
        image_paths = [os.path.join(image_folder, image) for image in os.listdir(image_folder)]

        images = []
        for i in range(self.max_images):
            if i < len(image_paths):
                image = Image.open(image_paths[i]).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            else:
                # Pad with zeros if fewer images exist
                images.append(torch.zeros((3, 128, 128)))

        tabular_features = torch.tensor(tabular_features, dtype=torch.float)
        images = torch.stack(images)

        return tabular_features, images


class RealEstateModel(pl.LightningModule):
    def __init__(self, tabular_input_size, hidden_size=64, max_images=6):
        super(RealEstateModel, self).__init__()
        self.max_images = max_images

        # Tabular model
        self.tabular_model = nn.Sequential(
            nn.Linear(tabular_input_size, hidden_size),
            nn.ReLU()
        )

        # Image model with modified structure
        self.image_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 64 * 64, hidden_size)  # Adjusted to match with the hidden size
        )

        # Combined model
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_size + hidden_size * self.max_images, 32),  # Update input size for concatenated output
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, tabular_data, image_data):
        tabular_output = self.tabular_model(tabular_data)
        # Process image data (multiple images)
        image_output = []
        for i in range(6):
            single_image_output = self.image_model(image_data[:, i])  # Process each image
            image_output.append(single_image_output)

        # Concatenate the outputs of all processed images
        image_output = torch.cat(image_output, dim=1)

        # Combine tabular and image outputs
        combined = torch.cat((tabular_output, image_output), dim=1)

        # Final prediction
        output = self.fc_combined(combined).squeeze(1)
        return output

    def training_step(self, batch, batch_idx):
        tabular_data, image_data, targets = batch
        outputs = self(tabular_data, image_data)
        # loss = nn.MSELoss()(outputs, targets)
        # Calculate Mean Absolute Error (MAE)
        loss = nn.L1Loss()(outputs, targets)

        # Log both loss and MAE
        # self.log('train_loss', loss)  # Log loss
        self.log('train_mae', loss)    # Log MAE
        return loss

    def predict_step(self, batch, batch_idx):
        tabular_data, image_data, _ = batch
        return self(tabular_data, image_data)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class InputLogger(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        # Log the input to the training step
        self.log("train_input", batch["x"])

    def validation_step(self, batch, batch_idx):
        # Log the input to the validation step
        self.log("valid_input", batch["x"])


def get_dataloader(X, y, shuffle, dir, transform, batch_size):
        dataset = RealEstateDataset(
            tabular_data=X,
            target=y,
            image_dir=f'data/reduced_images_ILB/reduced_images/{dir}',
            transform=transform
            )
        dataloader = (
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle
                )
            )
        return dataloader