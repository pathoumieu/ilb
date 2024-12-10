import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class RealEstateDataset(Dataset):
    """
    Custom Dataset for loading tabular and image data for real estate listings.

    Parameters
    ----------
    tabular_data : pandas.DataFrame
        DataFrame containing the tabular data (features).
    cols : list
        List of column names to be used as features from the tabular data.
    target : pandas.Series
        The target values for the dataset.
    image_dir : str
        Directory where the images are stored.
    im_size : tuple
        Desired image size (height, width).
    transform : callable, optional
        A function/transform to apply to the images.
    max_images : int, optional
        Maximum number of images per real estate listing (default is 6).
    """

    def __init__(self, tabular_data, cols, target, image_dir, im_size, transform=None, max_images=6):
        self.tabular = tabular_data  # Tabular data (features)
        self.target = target  # Target values (labels)
        self.image_dir = image_dir  # Directory where images are stored
        self.im_size = im_size  # Desired image size (height, width)
        self.transform = transform  # Optional transform function for images
        self.max_images = max_images  # Maximum number of images per listing
        self.cols = cols  # Columns to use from the tabular data

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.tabular)

    def __getitem__(self, idx):
        """
        Retrieves one sample (tabular data and images) by index.

        Returns
        -------
        tabular_features : torch.Tensor
            Tabular features for the given sample.
        images : torch.Tensor
            Stack of images for the given sample, padded if necessary.
        targets : torch.Tensor
            Target values for the given sample.
        """

        # Get tabular features and target for the given sample
        tabular_features = self.tabular[self.cols].iloc[idx].values.tolist()
        targets = self.target.iloc[idx].tolist()

        # Construct the image folder path for the given listing
        image_folder = os.path.join(self.image_dir, f'ann_{int(self.tabular.iloc[idx]["id_annonce"])}')

        # If the folder doesn't exist, fall back to the test dataset folder
        if not os.path.exists(image_folder):
            image_folder = os.path.join(
                'data/reduced_images_ILB/reduced_images/test',
                f'ann_{int(self.tabular.iloc[idx]["id_annonce"])}'
            )

        # Get image paths from the folder
        image_paths = [os.path.join(image_folder, image) for image in os.listdir(image_folder)]

        # Load and process images
        images = []
        for i in range(self.max_images):
            if i < len(image_paths):
                # Open image, convert to RGB, and apply transformations (if any)
                image = Image.open(image_paths[i]).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            else:
                # Pad with zeros (black images) if fewer than max_images
                images.append(torch.zeros((3, *self.im_size)))

        # Convert tabular features and targets to tensors
        tabular_features = torch.tensor(tabular_features, dtype=torch.float)
        images = torch.stack(images)  # Stack all images into a tensor
        targets = torch.tensor(targets, dtype=torch.float)

        return tabular_features, images, targets


def get_dataloader(X, cols, y, shuffle, dir, transform, batch_size, im_size, num_workers=8):
    """
    Creates a DataLoader for the RealEstateDataset.

    Parameters
    ----------
    X : pandas.DataFrame
        The tabular data (features).
    cols : list
        List of column names to be used as features.
    y : pandas.Series
        The target values for the dataset.
    shuffle : bool
        Whether to shuffle the dataset.
    dir : str
        Subdirectory of the image data.
    transform : callable, optional
        A function/transform to apply to the images.
    batch_size : int
        The batch size for the DataLoader.
    im_size : tuple
        Desired image size (height, width).
    num_workers : int, optional
        Number of workers to load data in parallel (default is 8).

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader instance for the given dataset.
    """

    # Initialize the RealEstateDataset with the provided parameters
    dataset = RealEstateDataset(
        tabular_data=X,
        target=y,
        cols=cols,
        im_size=im_size,
        image_dir=f'data/reduced_images_ILB/reduced_images/{dir}',
        transform=transform
    )

    # Create and return a DataLoader for the dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader
