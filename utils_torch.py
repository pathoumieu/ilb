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

NUM_WORKERS = 8


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
    def __init__(self, tabular_input_size, im_size, hidden_size=64, max_images=6):
        super(RealEstateModel, self).__init__()
        self.max_images = max_images
        self.im_size = im_size
        self.hidden_size = hidden_size

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

        # Reshape image_data to (batch_size * max_images, 3, 64, 64)
        image_data = image_data.view(-1, 3, *self.im_size)

        # Process all images in one pass
        image_output = self.image_model(image_data)

        # Reshape image_output to (batch_size, max_images * hidden_size)
        image_output = image_output.view(-1, self.max_images * self.hidden_size)

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
        self.log('train_mae', loss, prog_bar=True)    # Log MAE
        return loss

    def validation_step(self, batch, batch_idx):
        tabular_data, image_data, targets = batch
        outputs = self(tabular_data, image_data)
        loss = nn.L1Loss()(outputs, targets)
        self.log('valid_mae', loss, prog_bar=True, on_step=False, on_epoch=True)    # Log MAE

    def predict_step(self, batch, batch_idx):
        tabular_data, image_data, _ = batch
        return self(tabular_data, image_data)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


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
                shuffle=shuffle,
                num_workers=NUM_WORKERS
                )
            )
        return dataloader