import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_tabnet.tab_network import TabNet
from dataclasses import dataclass

from models.base_model import TabNetEncoder, ImageTransformer, ImageModel, GatedImageAttention
from utils_torch import MAPE


@dataclass
class ModelConfig:
    """
    Model configuration class to store all hyperparameters for the model.

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
    tabular_input_size: int
    im_size: tuple
    cat_idxs: list
    cat_dims: list
    cat_emb_dim: list
    hidden_size: int = 64
    max_images: int = 6
    lr: float = 1e-3
    weight_decay: float = 0.00001
    lr_factor: float = 0.1
    lr_patience: int = 50
    pretrain: bool = True
    mid_level_layer: int = 8
    pretrained_tabnet: object = None  # Assume TabNet or None
    dropout: float = 0.9
    loss_name: str = 'mae'
    batch_norm: bool = True
    clip_grad: float = 1.0
    last_layer_size: int = 32
    n_heads: int = 1
    n_layers: int = 1
    image_model_name: str = 'mobilenet'
    transformer_attention: bool = False
    scheduler_params: dict = {}


class RealEstateModel(pl.LightningModule):
    """
    RealEstateModel combines tabular data and image data to predict real estate prices.

    It uses a TabNet model for tabular data and a custom image model for image data,
    with optional attention mechanisms to combine the features from both modalities.

    The model works in the following way:

    1. **Tabular Data Processing**:
        - The input tabular data is passed through the TabNet model.
        - TabNet uses a series of attention-based mechanisms to process the tabular data.
          It extracts relevant features from the tabular input and outputs a set of learned
          features that represent the tabular information.
        - If no pretrained TabNet model is provided, a new TabNet model is created using the provided
          configuration parameters.

    2. **Image Data Processing**:
        - The input image data is passed through an image model.
        - If a pre-trained image model is specified (`pretrain`), it will use the pre-trained model,
          freezing the layers if indicated. Otherwise, a custom image model is constructed using
          convolutional layers to process the image data.
        - The image data is first processed through convolutional layers, followed by a flattening layer
          and a linear layer that outputs features of a size equal to `hidden_size`.

    3. **Attention Mechanism**:
        - If the `transformer_attention` flag is set to `True`, an attention mechanism (either a Transformer
          or a gated attention) is applied to the image features.
        - The attention mechanism is responsible for learning which parts of the image features are most
          important, helping the model focus on key aspects of the image data for prediction.

    4. **Combining Tabular and Image Features**:
        - The features from the tabular data and the processed image features are combined (concatenated)
          into a single feature vector.
        - These combined features are then passed through a fully connected layer to produce the final
          prediction.

    5. **Final Prediction**:
        - The model produces a final output (real estate price prediction) after combining the tabular and
          image features. This is done through a fully connected layer that outputs a single scalar value,
          representing the predicted real estate price.

    6. **Training**:
        - During training, the model calculates the loss based on the selected loss function (`mae` or `mse`),
          and uses this loss to update the model weights via backpropagation.
        - The optimizer used for training is Adam, and the learning rate and weight decay are set according
          to the configuration.
        - The model also computes Mean Absolute Percentage Error (MAPE) as an additional evaluation metric.
        - If specified, gradient clipping is applied during training to prevent the model from diverging due
          to excessively large gradients.

    7. **Learning Rate Scheduling**:
        - The learning rate is adjusted during training using a scheduler. If no custom scheduler parameters
          are provided, a `ReduceLROnPlateau` scheduler is used, which reduces the learning rate when the
          validation loss stops improving.
        - Alternatively, if custom scheduler parameters are provided, a `OneCycleLR` scheduler is used for
          more aggressive learning rate adjustments during training.
    """

    def __init__(self, config: ModelConfig):
        super(RealEstateModel, self).__init__()

        # Directly unpack the config dataclass and update model's attributes
        self.__dict__.update(config.__dict__)

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
        if self.pretrained_tabnet is None:
            # If no pretrained TabNet is provided, create a new TabNet model
            self.tabular_model = TabNet(
                input_dim=self.tabular_input_size,
                output_dim=self.hidden_size,
                n_d=64,  # Input size for TabNet
                n_a=64,  # Attention size
                n_shared=4,  # Number of shared layers
                n_independent=4,  # Number of independent layers
                n_steps=3,  # Number of attention mechanism steps
                gamma=1.3,  # Regularization parameter
                cat_idxs=self.cat_idxs,  # Indices of categorical columns
                cat_dims=self.cat_dims,  # Dimensions of categorical features
                cat_emb_dim=self.cat_emb_dim,  # Embedding dimension of categorical features
                # Attention matrix
                group_attention_matrix=torch.eye(self.tabular_input_size).to(self.device_name)
            ).to(self.device_name)
            out_size = self.hidden_size
        else:
            # If a pretrained TabNet model is provided, use it
            self.tabular_model = TabNetEncoder(self.pretrained_tabnet).to(self.device_name)
            out_size = self.tabular_model.out_features

        # Initialize image model (either pre-trained or custom)
        if self.pretrain is not None:
            if isinstance(self.pretrain, bool):  # Ensure that pretrain is a boolean
                self.image_model = ImageModel(
                    image_model_name=self.image_model_name,
                    freeze=self.pretrain,
                    mid_level_layer=self.mid_level_layer,
                    hidden_size=self.hidden_size,
                    im_size=self.im_size,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                ).to(self.device_name)
            else:
                raise ValueError(
                    "`pretrain` should be a bool value indicating whether to freeze the image model layers."
                    )
        else:
            # Custom image model (if no pretraining)
            self.image_model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                # Adjusted to match hidden size
                nn.Linear(int(16 * self.im_size[0] * self.im_size[1] / 4), self.hidden_size)
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
            # Concatenate tabular and image outputs
            nn.Linear(out_size + self.hidden_size, self.last_layer_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.last_layer_size, 1).to(self.device_name)  # Final output layer
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if len(self.scheduler_params) == 0:
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
                **self.scheduler_params
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
