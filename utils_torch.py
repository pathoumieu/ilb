import os
import wandb
import zipfile
import json
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.metrics import Metric


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
        if not os.path.exists(image_folder):
            image_folder = os.path.join('data/reduced_images_ILB/reduced_images/test', 'ann_' + str(int(self.tabular.iloc[idx]['id_annonce'])))

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


class RealEstateTestDataset(RealEstateDataset):
    def __init__(self, tabular_data, cols, image_dir, im_size, transform=None, max_images=6):
        self.tabular = tabular_data
        self.image_dir = image_dir
        self.transform = transform
        self.max_images = max_images
        self.im_size = im_size
        self.cols = cols

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
        # Capture the current learning rate
        learning_rate = self.trainer._optimizer.param_groups[0]['lr']
        # Log the learning rate to Weights & Biases
        wandb.log({"epoch": epoch, "learning_rate": learning_rate})
        pass

    def on_epoch_end(self, epoch, logs=None):
        wandb.log({
            "train_mae": logs['train_mae'],
            "train_mape": logs['train_mape'],
            "valid_mae": logs['valid_mae'],
            "valid_mape": logs['valid_mape']
            })
        pass

    def on_batch_end(self, batch, logs=None):
        pass


def load_trained_tabnet(run, version="v50", freeze=True, freeze_embed=True, device="cuda"):
    artifact = run.use_artifact(f"tabnet_model:{version}")

    datadir = artifact.download()

    with zipfile.ZipFile(datadir + '/tabnet_model.pt.zip', 'r') as zip_ref:
        zip_ref.extractall(datadir)

    with open(datadir + '/model_params.json', 'r') as f:
        model_params = json.load(f)

    keys_to_pop = [
        'clip_value',
        'device_name',
        'grouped_features',
        'lambda_sparse',
        'n_shared_decoder',
        'n_indep_decoder',
        'optimizer_params',
        'scheduler_params',
        'seed',
        'verbose',
    ]

    for key in keys_to_pop:
        model_params['init_params'].pop(key)

    model = TabNet(
        **model_params['init_params'],
        group_attention_matrix=torch.eye(model_params['init_params']['input_dim']).to(device)
        )

    model.load_state_dict(torch.load(datadir + '/network.pt'))

    # Freeze the weights
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    if freeze_embed and not freeze:
        # Freeze specific layers, for example, embedding layers
        for layer in model.embedder.embeddings.children():
            if isinstance(layer, nn.Embedding):
                layer.weight.requires_grad = False

    return model.to(device)


def MAPE(predictions, targets):
    absolute_percentage_errors = torch.abs((targets - predictions) / targets)
    mape = torch.mean(absolute_percentage_errors)
    return mape.item()


class tabnet_mape(Metric):
    def __init__(self):
        self._name = "mape" # write an understandable name here
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute AUC of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            AUC of predictions vs targets.
        """
        return mean_absolute_percentage_error(y_true, y_score)
