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
from torchvision import models
from pytorch_tabnet.tab_network import TabNet


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
    def __init__(self, tabular_data, cols, target, image_dir, im_size, transform=None, max_images=6):
        self.tabular = tabular_data
        self.target = target
        self.image_dir = image_dir
        self.im_size = im_size
        self.transform = transform
        self.max_images = max_images
        self.cols = cols

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        tabular_features = self.tabular[self.cols].iloc[idx].values.tolist()
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
                images.append(torch.zeros((3, *self.im_size)))

        tabular_features = torch.tensor(tabular_features, dtype=torch.float)
        images = torch.stack(images)
        targets = torch.tensor(targets, dtype=torch.float)

        return tabular_features, images, targets


class RealEstateTestDataset(Dataset):
    def __init__(self, tabular_data, cols, image_dir, im_size, transform=None, max_images=6):
        self.tabular = tabular_data
        self.image_dir = image_dir
        self.transform = transform
        self.max_images = max_images
        self.im_size = im_size
        self.cols = cols

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        tabular_features = self.tabular[self.cols].iloc[idx].values.tolist()

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
                images.append(torch.zeros(3, *self.im_size))

        tabular_features = torch.tensor(tabular_features, dtype=torch.float)
        images = torch.stack(images)

        return tabular_features, images


class ModifiedMobileNet(nn.Module):
    def __init__(self, freeze=True, mid_level_layer=7, hidden_size=8, im_size=(64, 64)):
        super(ModifiedMobileNet, self).__init__()
        # Load the pre-trained MobileNet model
        mobilenet = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")

        # Extract layers up to the mid-level layer
        self.features = nn.Sequential(*list(mobilenet.features.children())[:mid_level_layer])

        # Freeze the layers if specified
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        # Flatten the output
        self.flatten = nn.Flatten()

        # Get the number of output channels at the specified mid_level_layer
        dummy_input = torch.randn(1, 3, *im_size)  # Assuming input size of 224x224
        mid_level_output = self.flatten(self.features(dummy_input))
        num_channels = mid_level_output.size(1)

        # Additional trainable fully connected layer
        self.additional_fc = nn.Linear(num_channels, hidden_size)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.additional_fc(x)
        return x


class RealEstateModel(pl.LightningModule):
    def __init__(
            self,
            tabular_input_size,
            im_size,
            cat_idxs,
            cat_dims,
            cat_emb_dim,
            hidden_size=64,
            max_images=6,
            lr=1e-3,
            weight_decay=0.00001,
            lr_factor=0.1,
            lr_patience=50,
            pretrain=True,
            mid_level_layer=8
            ):
        super(RealEstateModel, self).__init__()
        self.max_images = max_images
        self.im_size = im_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.pretrain = pretrain
        self.mid_level_layer = mid_level_layer

        # # Tabular model
        # self.tabular_model = nn.Sequential(
        #     nn.Linear(tabular_input_size, hidden_size),
        #     nn.ReLU()
        # )
        # TabNet model for tabular data
        self.tabular_model = TabNet(
            input_dim=tabular_input_size,
            output_dim=self.hidden_size,
            n_d=8,  # Specify input size for TabNet
            n_a=8,
            n_steps=1,  # Number of steps in the attention mechanism
            gamma=1.3,  # Regularization parameter
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            group_attention_matrix=torch.eye(tabular_input_size)
        )

        # Image model with modified structure
        if pretrain is not None:
            assert type(pretrain) is bool
            self.image_model = ModifiedMobileNet(
                freeze=pretrain,
                mid_level_layer=mid_level_layer,
                hidden_size=hidden_size,
                im_size=im_size,
                )
        else:
            self.image_model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(int(16 * im_size[0] * im_size[1] / 4), hidden_size)  # Adjusted to match with the hidden size
            )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.values_layer = nn.Linear(hidden_size, hidden_size)

        # Combined model
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, 32),  # Update input size for concatenated output
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, tabular_data, image_data):
        tabular_output, _ = self.tabular_model(tabular_data)

        # Reshape image_data to (batch_size * max_images, 3, 64, 64)
        image_data = image_data.view(-1, 3, *self.im_size)

        # Process all images in one pass
        image_output = self.image_model(image_data)

        # Reshape image_output to (batch_size, max_images, hidden_size)
        image_output = image_output.view(-1, self.max_images, self.hidden_size)
        image_values = self.values_layer(image_output)

        # Apply attention mechanism
        attention_weights = F.softmax(self.attention(image_output), dim=1)
        attended_image_features = torch.sum(attention_weights * image_values, dim=1)

        # Combine tabular and image outputs
        combined = torch.cat((tabular_output, attended_image_features), dim=1)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
        )
        scheduler = {
            "scheduler": sched,
            "monitor": "valid_mae",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


def get_dataloader(X, cols, y, shuffle, dir, transform, batch_size, im_size, num_workers=8):
        dataset = RealEstateDataset(
            tabular_data=X,
            target=y,
            cols=cols,
            im_size=im_size,
            image_dir=f'data/reduced_images_ILB/reduced_images/{dir}',
            transform=transform
            )
        dataloader = (
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
                )
            )
        return dataloader