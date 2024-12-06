import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from pytorch_tabnet.tab_network import TabNet

from utils_torch import MAPE


class ImageModel(nn.Module):
    def __init__(
            self,
            image_model_name='tf_efficientnet_b0_ns',
            freeze=True,
            mid_level_layer=7,
            hidden_size=8,
            im_size=(64, 64),
            dropout=0.1,
            batch_norm=True
            ):
        """
        Initializes the ImageModel class, which allows selecting from
        pre-trained image models (e.g., EfficientNet, MobileNet) and customizing
        the architecture for feature extraction and classification.

        Parameters:
        - image_model_name (str): The name of the pre-trained image model (e.g., 'tf_efficientnet_b0_ns',
          'mobilenet').
        - freeze (bool): If True, freeze the layers of the pre-trained model.
        - mid_level_layer (int): Specifies up to which layer the pre-trained model should be used.
        - hidden_size (int): Size of the hidden layer after feature extraction.
        - im_size (tuple): Input image size, e.g., (224, 224) or (64, 64).
        - dropout (float): Dropout rate for regularization.
        - batch_norm (bool): If True, applies batch normalization after the hidden layer.
        """
        super(ImageModel, self).__init__()

        # Flag for applying batch normalization
        self.batch_norm = batch_norm

        # Handle EfficientNet model selection
        if 'efficientnet' in image_model_name:
            # Load pre-trained EfficientNet-B0 NoisyStudent model from timm library
            efficientnet = timm.create_model(image_model_name, pretrained=True)

            # Extract layers up to the mid-level layer specified
            blocks = list(efficientnet.blocks.children())[:mid_level_layer]
            self.features = nn.Sequential(efficientnet.conv_stem, efficientnet.bn1, *blocks)

        # Handle MobileNet model selection
        elif image_model_name == 'mobilenet':
            # Load pre-trained MobileNet-V2 model from torchvision
            mobilenet = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")

            # Extract layers up to the mid-level layer specified
            self.features = nn.Sequential(*list(mobilenet.features.children())[:mid_level_layer])
        else:
            raise ValueError('Wrong image model name. Choose from "efficientnet" or "mobilenet".')

        # Freeze the layers if freeze flag is set to True
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        # Flatten the output from convolutional layers
        self.flatten = nn.Flatten()

        # BatchNorm applied to the fully connected layer after feature extraction
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Get the number of output channels at the specified mid-level layer
        dummy_input = torch.randn(1, 3, *im_size)  # Assuming input size of (3, 224, 224)
        mid_level_output = self.flatten(self.features(dummy_input))
        num_channels = mid_level_output.size(1)

        # Additional fully connected layer after feature extraction
        self.additional_fc = nn.Linear(num_channels, hidden_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the model.

        Parameters:
        - x (torch.Tensor): Input tensor with shape (batch_size, 3, height, width).

        Returns:
        - torch.Tensor: The output tensor after the forward pass with shape (batch_size, hidden_size).
        """
        # Pass through the feature extractor (EfficientNet or MobileNet up to mid-level layer)
        x = self.features(x)

        # Flatten the features into a 1D vector for each image
        x = self.flatten(x)

        # Pass through the additional fully connected layer
        x = self.additional_fc(x)

        # Apply batch normalization if enabled
        if self.batch_norm:
            x = self.batch_norm(x)

        # Apply ReLU activation
        x = F.relu(x)

        # Apply dropout for regularization
        x = self.dropout(x)

        return x


class TabNetEncoder(nn.Module):
    """
    This class defines a TabNet encoder that extracts features from tabular data.
    It takes a pre-trained TabNet model and removes the final output layer to extract
    intermediate features.

    Parameters:
    - tabnet_model (nn.Module): A pre-trained TabNet model that is used for feature extraction.
      The model should have an embedder and a tabnet encoder part.
    """

    def __init__(self, tabnet_model):
        """
        Initializes the TabNetEncoder class by extracting and configuring the embedding
        and TabNet layers from a pre-trained TabNet model.

        Parameters:
        - tabnet_model (nn.Module): The pre-trained TabNet model used to initialize the encoder.
        """
        super(TabNetEncoder, self).__init__()

        # Set the device to CUDA if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model to the appropriate device (GPU or CPU)
        tabnet_model = tabnet_model.to(self.device)

        # Extract the embedder and tabnet layers from the pre-trained TabNet model
        self.embedder = tabnet_model.embedder
        self.tabnet = tabnet_model.tabnet

        # Get the number of output features (e.g., the number of decision steps in TabNet)
        self.out_features = tabnet_model.n_d

        # Remove the final mapping layer as we are interested in the intermediate features
        self.tabnet.final_mapping = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the TabNet encoder. This method performs the following:
        1. Passes the input through the embedding layers.
        2. Passes the output through the TabNet encoder layers.
        """
        # Forward pass through the embedding layers
        x = self.embedder(x)

        # Forward pass through the TabNet encoder layers
        x = self.tabnet(x)

        return x


class SimpleEncoder(nn.Module):
    """
    A simple neural network encoder designed for mixed data types (numerical and categorical).
    This encoder performs the following steps:
    1. Embeds categorical features using embedding layers.
    2. Concatenates the embedded categorical features with numerical features.
    3. Passes the combined features through a series of fully connected layers with batch normalization
       and dropout.

    Parameters:
    - input_size (int): The total number of input features (numerical + categorical).
    - cat_idxs (list of int): The indices of the categorical columns in the input.
    - cat_dims (list of int): The number of unique values (cardinality) for each categorical feature.
    - cat_emb_dims (list of int): The embedding dimensions for each categorical feature.
    - encoder_hidden_size (int, optional): The size of the hidden layers in the encoder. Default is 512.
    - output_size (int, optional): The size of the output layer. Default is 256.
    - dropout_prob (float, optional): The dropout probability to prevent overfitting. Default is 0.1.
    """

    def __init__(
            self,
            input_size,
            cat_idxs,
            cat_dims,
            cat_emb_dims,
            encoder_hidden_size=512,
            output_size=256,
            dropout_prob=0.1
            ):
        super(SimpleEncoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.cat_idxs = cat_idxs

        # Embedding layers for categorical variables, one embedding for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dims[i], cat_emb_dims[i]) for i in cat_idxs
        ])

        # Fully connected layers for encoding, with batch normalization
        self.fc1 = nn.Linear(input_size - len(self.cat_idxs) + sum(cat_emb_dims), encoder_hidden_size)
        self.bn1 = nn.BatchNorm1d(encoder_hidden_size)

        self.fc2 = nn.Linear(encoder_hidden_size, encoder_hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(encoder_hidden_size // 2)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_prob)

        # Final output layer
        self.output_layer = nn.Linear(encoder_hidden_size // 2, output_size)

    def forward(self, x):
        """
        Forward pass through the SimpleEncoder. This method:
        1. Splits the input tensor into categorical and numerical components.
        2. Embeds categorical variables using the predefined embedding layers.
        3. Concatenates the embedded categorical features with the numerical features.
        4. Passes the combined features through the encoder's fully connected layers with
           batch normalization and dropout.
        5. Produces the final output.
        """
        # Extract categorical and numerical features from the input tensor
        x_cat = x[:, self.cat_idxs].long()  # Categorical features
        num_idxs = [i for i in range(self.input_size) if i not in self.cat_idxs]  # Numerical features indices
        x_num = x[:, num_idxs]  # Numerical features

        # Embed the categorical features
        cat_embedded = [
            embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        cat_embedded = torch.cat(cat_embedded, 1)  # Concatenate all the embeddings

        # Concatenate the embedded categorical features with the numerical features
        x = torch.cat([cat_embedded, x_num], 1)

        # Pass through the fully connected layers with batch normalization and dropout
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))  # First layer with batch norm and dropout
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))  # Second layer with batch norm and dropout

        # Output layer
        x = self.output_layer(x)

        return x


class SimpleAttentionModule(nn.Module):
    """
    A simple attention mechanism module that computes weighted attention on a sequence of features.
    This module performs the following steps:
    1. Uses a learned attention mechanism to calculate the attention weights.
    2. Applies these weights to the input sequence (e.g., image sequence) to compute
       a weighted sum of features.

    Parameters:
    - hidden_size (int): The size of the hidden layers in the attention mechanism.
    - dropout (float): The dropout probability to prevent overfitting.
    """

    def __init__(self, hidden_size, dropout):
        super().__init__()

        # Attention mechanism: learnable weights to compute attention over input features
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # First linear layer
            nn.Tanh(),  # Activation function to introduce non-linearity
            nn.Linear(hidden_size, 1),  # Output layer to produce attention scores
            nn.Dropout(dropout)  # Dropout to prevent overfitting
        )

        # Values layer: transform the input sequence before applying attention
        self.values_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # Linear transformation for values
            nn.Dropout(dropout)  # Dropout for regularization
        )

    def forward(self, image_sequence):
        """
        Forward pass through the SimpleAttentionModule. This method:
        1. Computes attention weights based on the input sequence.
        2. Applies these weights to the input sequence (image sequence) to calculate
           a weighted sum of features.

        Parameters:
        - image_sequence (torch.Tensor): A tensor representing a sequence of image features
          (batch_size x seq_len x hidden_size).

        Returns:
        - attended_image_features (torch.Tensor): A tensor representing the weighted sum of features
          after applying attention.
        """
        # Apply the values layer to the input sequence
        image_values_sequence = self.values_layer(image_sequence)

        # Compute attention weights using the attention mechanism
        attention_weights = F.softmax(self.attention(image_sequence), dim=1)

        # Compute the weighted sum of image features based on the attention weights
        attended_image_features = torch.sum(attention_weights * image_values_sequence, dim=1)

        return attended_image_features


class GatedImageAttention(nn.Module):
    """
    Gated Image Attention Mechanism.

    This module computes gated attention over a sequence of image features.
    It uses attention weights and gating values to selectively focus on relevant features of the image
    sequence and produces a weighted sum of these features based on the computed gates and attention weights.

    Parameters:
    - n_image_features (int): The number of features for each image in the sequence.
    - n_heads (int): The number of attention heads for the attention mechanism.
    - dropout (float): Dropout probability to regularize the model and prevent overfitting.
    """

    def __init__(self, n_image_features, n_heads, dropout):
        super(GatedImageAttention, self).__init__()

        self.n_heads = n_heads
        self.n_image_features = n_image_features

        # Linear layer to compute attention weights for each image feature
        self.attention_weights = nn.Linear(n_image_features, n_heads)

        # Linear layer to compute gating values for each image feature
        self.gating_layer = nn.Linear(n_image_features, n_heads)

        # Gating activation function
        self.gating_activation = nn.Sigmoid()

        # Flatten layer to reshape the output
        self.flatten = nn.Flatten()

        # Output layer to transform the gated representation
        self.linear_out = nn.Linear(n_heads * n_image_features, n_image_features)

        # Dropout layer to regularize the model
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features_sequence):
        """
        Forward pass through the GatedImageAttention module. This method computes the attention weights,
        applies gating, and computes the final weighted sum of image features.

        Parameters:
        - image_features_sequence (torch.Tensor): A tensor representing a sequence of image features
                                                  (batch_size x seq_len x n_image_features).

        Returns:
        - gated_image_representation (torch.Tensor): The gated image representation, which is a weighted sum
                                                     of the input image features after applying attention and
                                                     gating.
        """
        # Compute attention weights
        attention_weights = self.attention_weights(image_features_sequence)
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize attention weights across sequence
        attention_weights = self.dropout(attention_weights)  # Apply dropout to attention weights

        # Compute gating values using sigmoid activation
        gating_values = self.gating_activation(self.gating_layer(image_features_sequence))
        gating_values = self.dropout(gating_values)  # Apply dropout to gating values

        # Apply gated attention: element-wise multiplication of attention weights and gating values
        gated_image_representation = (attention_weights * gating_values).unsqueeze(2) *\
            image_features_sequence.unsqueeze(-1)
        gated_image_representation = gated_image_representation.sum(dim=1)  # Sum over the sequence length

        gated_image_representation = self.flatten(gated_image_representation)  # Flatten the output for linear

        # If multiple attention heads, apply the linear output layer
        if self.n_heads > 1:
            return self.linear_out(gated_image_representation)

        return gated_image_representation  # Return the final gated representation


class ImageTransformer(nn.Module):
    """
    Transformer-based Encoder for Image Feature Processing.

    This module processes image feature sequences using a Transformer encoder, followed by mean pooling and
    a feedforward neural network. It is designed to extract features from a sequence of image representations
    or patches, aggregating the sequence through mean pooling before passing it through a fully connected
    layer.

    Parameters:
    - num_heads (int): The number of attention heads for the Transformer encoder.
    - hidden_size (int): The size of the hidden layers in the Transformer encoder and feedforward network.
    - num_layers (int): The number of layers in the Transformer encoder.
    - dropout (float): Dropout probability for regularization in the feedforward neural network.
    """

    def __init__(self, num_heads, hidden_size, num_layers, dropout):
        super(ImageTransformer, self).__init__()

        # Transformer Encoder Layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,  # Hidden size of the Transformer
                nhead=num_heads,  # Number of attention heads
                dim_feedforward=hidden_size * 2,  # Feedforward layer size (usually larger than d_model)
            ),
            num_layers=num_layers,  # Number of layers in the Transformer Encoder
        )

        # Adaptive Mean Pooling Layer to aggregate the sequence
        self.mean_pooling = nn.AdaptiveAvgPool1d(1)

        # Feedforward Neural Network (after the Transformer encoder and pooling)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # Fully connected layer
            nn.ReLU(),  # Activation function
            nn.Dropout(dropout)  # Dropout for regularization
        )

    def forward(self, x):
        """
        Forward pass through the ImageTransformer module.

        This method processes the input sequence of image features through the Transformer encoder,
        applies mean pooling to aggregate the sequence, and then passes the result through a feedforward
        neural network for final output.

        Parameters:
        - x (torch.Tensor): The input sequence of image features, expected shape
          (batch_size, seq_len, hidden_size).

        Returns:
        - output (torch.Tensor): The output of the feedforward network, representing the processed image
          features.
        """
        # Apply the Transformer Encoder
        x = self.transformer_encoder(x)

        # Apply Mean Pooling to aggregate features across the sequence length dimension
        x = self.mean_pooling(x.permute(0, 2, 1)).squeeze()

        # Pass the pooled output through the Feedforward Neural Network
        output = self.feedforward(x)

        return output


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

        self.device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.loss_name == 'mae':
            self.loss_func = nn.L1Loss()
        elif self.loss_name == 'mse':
            self.loss_func = nn.MSELoss()
        else:
            raise ValueError('Wrong loss name')

        # TabNet model for tabular data
        if pretrained_tabnet is None:
            self.tabular_model = TabNet(
                input_dim=tabular_input_size,
                output_dim=self.hidden_size,
                n_d=64,  # Specify input size for TabNet
                n_a=64,
                n_shared=4,
                n_independent=4,
                n_steps=3,  # Number of steps in the attention mechanism
                gamma=1.3,  # Regularization parameter
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=cat_emb_dim,
                group_attention_matrix=torch.eye(tabular_input_size).to(self.device_name)
            ).to(self.device_name)
            out_size = hidden_size
        else:
            self.tabular_model = TabNetEncoder(pretrained_tabnet).to(self.device_name)
            out_size = self.tabular_model.out_features

        # Image model with modified structure
        if pretrain is not None:
            assert type(pretrain) is bool
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
            self.image_model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(int(16 * im_size[0] * im_size[1] / 4), hidden_size)  # Adjusted to match
                                                                               # with the hidden size
            ).to(self.device_name)

        # Attention mechanism
        # self.attention = SimpleAttentionModule(self.hidden_size, self.dropout)
        if self.transformer_attention:
            self.attention = ImageTransformer(
                self.n_heads, self.hidden_size, self.n_layers, self.dropout
                ).to(self.device_name)
        else:
            self.attention = GatedImageAttention(
                self.hidden_size, self.n_heads, self.dropout
                ).to(self.device_name)

        # Combined model
        self.fc_combined = nn.Sequential(
            nn.Linear(out_size + hidden_size, last_layer_size),  # Update input size for concatenated output
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(last_layer_size, 1).to(self.device_name)
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
        # Move input data to the same device as the model
        tabular_data = tabular_data.to(self.device_name)
        image_data = image_data.to(self.device_name)
        tabular_output = self.tabular_model(tabular_data)

        # Reshape image_data to (batch_size * max_images, 3, 64, 64)
        image_data = image_data.view(-1, 3, *self.im_size)

        # Process all images in one pass
        image_output = self.image_model(image_data)

        # Reshape image_output to (batch_size, max_images, hidden_size)
        image_output = image_output.view(-1, self.max_images, self.hidden_size)

        # Apply attention mechanism
        attended_image_features = self.attention(image_output)

        # Combine tabular and image outputs
        combined = torch.cat((tabular_output, attended_image_features), dim=1)

        # Final prediction
        output = self.fc_combined(combined).squeeze(1)
        return output

    def training_step(self, batch, batch_idx):
        tabular_data, image_data, targets = batch
        outputs = self(tabular_data, image_data)
        # Calculate Mean Absolute Error (MAE)

        loss = self.loss_func(outputs, targets)
        mape = MAPE(outputs, targets)
        # clip gradients
        if self.clip_grad is not None:
            self.clip_gradients(
                self.optimizer, gradient_clip_val=self.clip_grad, gradient_clip_algorithm="norm"
                )
        self.log(f'train_{self.loss_name}', loss, prog_bar=True)    # Log MAE/MSE
        self.log('train_mape', mape, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        tabular_data, image_data, targets = batch
        outputs = self(tabular_data, image_data)
        loss = self.loss_func(outputs, targets)
        mape = MAPE(outputs, targets)
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
