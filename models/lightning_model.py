import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_tabnet.tab_network import TabNet

from models.base_model import TabNetEncoder, ImageTransformer, ImageModel, GatedImageAttention
from utils_torch import MAPE


class RealEstateModel(pl.LightningModule):
    """
    RealEstateModel combines tabular data and image data to predict real estate prices.

    It uses a TabNet model for tabular data and a custom image model for image data,
    with optional attention mechanisms to combine the features from both modalities.

    Args:
        tabular_input_size (int): Number of features in the tabular data.
        im_size (tuple): Size of input images (height, width).
        cat_idxs (list): List of indices for categorical columns in the tabular data.
        cat_dims (list): List of number of unique values for each categorical column.
        cat_emb_dim (list): List of embedding dimensions for each categorical column.
        hidden_size (int): Size of hidden layers for the model.
        max_images (int): Maximum number of images to consider per sample.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        lr_factor (float): Factor by which the learning rate will be reduced.
        lr_patience (int): Number of epochs with no improvement before reducing the learning rate.
        pretrain (bool or None): Whether to pretrain the image model. If None, a custom image model is used.
        mid_level_layer (int): The layer of the image model from which features will be extracted.
        pretrained_tabnet (TabNet or None): A pretrained TabNet model for tabular data, if available.
        dropout (float): Dropout probability to be applied to the layers.
        loss_name (str): Loss function to use, either 'mae' or 'mse'.
        batch_norm (bool): Whether to use batch normalization in the image model.
        clip_grad (float): Value for gradient clipping.
        last_layer_size (int): Size of the final hidden layer before output.
        n_heads (int): Number of attention heads in the attention mechanism.
        n_layers (int): Number of layers in the attention mechanism.
        image_model_name (str): Name of the pre-trained image model (e.g., 'mobilenet').
        transformer_attention (bool): Whether to use a Transformer-based attention mechanism.
        scheduler_params (dict): Parameters for the learning rate scheduler.
    """

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
            mid_level_layer=8,
            pretrained_tabnet=None,
            dropout=0.9,
            loss_name='mae',
            batch_norm=True,
            clip_grad=1.0,
            last_layer_size=32,
            n_heads=1,
            n_layers=1,
            image_model_name='mobilenet',
            transformer_attention=False,
            scheduler_params={}
            ):
        super(RealEstateModel, self).__init__()

        # Store hyperparameters for later use
        self.max_images = max_images
        self.im_size = im_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.pretrain = pretrain
        self.mid_level_layer = mid_level_layer
        self.dropout = dropout
        self.loss_name = loss_name
        self.batch_norm = batch_norm
        self.clip_grad = clip_grad
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.transformer_attention = transformer_attention

        # Select device (GPU/CPU)
        self.device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define loss function based on user input
        if self.loss_name == 'mae':
            self.loss_func = nn.L1Loss()  # Mean Absolute Error
        elif self.loss_name == 'mse':
            self.loss_func = nn.MSELoss()  # Mean Squared Error
        else:
            raise ValueError('Wrong loss name')

        # Initialize the TabNet model for tabular data
        if pretrained_tabnet is None:
            # If no pretrained TabNet is provided, create a new TabNet model
            self.tabular_model = TabNet(
                input_dim=tabular_input_size,
                output_dim=self.hidden_size,
                n_d=64,  # Input size for TabNet
                n_a=64,  # Attention size
                n_shared=4,  # Number of shared layers
                n_independent=4,  # Number of independent layers
                n_steps=3,  # Number of attention mechanism steps
                gamma=1.3,  # Regularization parameter
                cat_idxs=cat_idxs,  # Indices of categorical columns
                cat_dims=cat_dims,  # Dimensions of categorical features
                cat_emb_dim=cat_emb_dim,  # Embedding dimension of categorical features
                group_attention_matrix=torch.eye(tabular_input_size).to(self.device_name)  # Attention matrix
            ).to(self.device_name)
            out_size = hidden_size
        else:
            # If a pretrained TabNet model is provided, use it
            self.tabular_model = TabNetEncoder(pretrained_tabnet).to(self.device_name)
            out_size = self.tabular_model.out_features

        # Initialize image model (either pre-trained or custom)
        if pretrain is not None:
            assert type(pretrain) is bool  # Ensure pretrain is a boolean
            self.image_model = ImageModel(
                image_model_name=image_model_name,
                freeze=pretrain,
                mid_level_layer=mid_level_layer,
                hidden_size=hidden_size,
                im_size=im_size,
                dropout=dropout,
                batch_norm=batch_norm
            ).to(self.device_name)
        else:
            # Custom image model (if no pretraining)
            self.image_model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(int(16 * im_size[0] * im_size[1] / 4), hidden_size)  # Adjusted to match hidden size
            ).to(self.device_name)

        # Initialize the attention mechanism (either Transformer or Gated)
        if self.transformer_attention:
            self.attention = ImageTransformer(
                self.n_heads, self.hidden_size, self.n_layers, self.dropout
            ).to(self.device_name)
        else:
            self.attention = GatedImageAttention(
                self.hidden_size, self.n_heads, self.dropout
            ).to(self.device_name)

        # Combined fully connected layer for the output
        self.fc_combined = nn.Sequential(
            nn.Linear(out_size + hidden_size, last_layer_size),  # Concatenate tabular and image outputs
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(last_layer_size, 1).to(self.device_name)  # Final output layer
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if len(scheduler_params) == 0:
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.lr_factor,
                patience=self.lr_patience,
            )
            self.scheduler_interval = "epoch"
        else:
            self.sched = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                **scheduler_params
            )
            self.scheduler_interval = "step"

    def forward(self, tabular_data, image_data):
        """
        Forward pass for the model. Takes in both tabular and image data.

        Args:
            tabular_data (Tensor): Tabular input data.
            image_data (Tensor): Image input data.

        Returns:
            Tensor: Model output (real estate price prediction).
        """
        # Move input data to the correct device
        tabular_data = tabular_data.to(self.device_name)
        image_data = image_data.to(self.device_name)

        # Get tabular data output from TabNet model
        tabular_output = self.tabular_model(tabular_data)

        # Reshape image data to handle multiple images per sample
        image_data = image_data.view(-1, 3, *self.im_size)

        # Process images through the image model
        image_output = self.image_model(image_data)

        # Reshape the image output to match the expected dimensions
        image_output = image_output.view(-1, self.max_images, self.hidden_size)

        # Apply attention mechanism on the image features
        attended_image_features = self.attention(image_output)

        # Combine tabular and image features
        combined = torch.cat((tabular_output, attended_image_features), dim=1)

        # Final prediction from the combined features
        output = self.fc_combined(combined).squeeze(1)
        return output

    def training_step(self, batch, batch_idx):

        tabular_data, image_data, targets = batch
        outputs = self(tabular_data, image_data)

        # Calculate MAE and MAPE
        loss = self.loss_func(outputs, targets)
        mape = MAPE(outputs, targets)

        # Clip gradients
        if self.clip_grad is not None:
            self.clip_gradients(
                self.optimizer, gradient_clip_val=self.clip_grad, gradient_clip_algorithm="norm"
                )

        # Log metrics
        self.log(f'train_{self.loss_name}', loss, prog_bar=True)
        self.log('train_mape', mape, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):

        tabular_data, image_data, targets = batch
        outputs = self(tabular_data, image_data)

        # Calculate MAE and MAPE
        loss = self.loss_func(outputs, targets)
        mape = MAPE(outputs, targets)

        # Log metrics
        self.log(f'valid_{self.loss_name}', loss, prog_bar=True, on_step=False, on_epoch=True)    # Log metric
        self.log('valid_mape', mape, prog_bar=False, on_step=False, on_epoch=True)
        self.log("learning_rate", self.optimizer.param_groups[0]['lr'], on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        tabular_data, image_data, _ = batch
        return self(tabular_data, image_data)

    def configure_optimizers(self):
        scheduler = {
            "scheduler": self.sched,
            "monitor": "valid_mape",
            "interval": self.scheduler_interval,
            "frequency": 1,
        }
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": scheduler,
        }
