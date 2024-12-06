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
        super(ImageModel, self).__init__()
        self.batch_norm = batch_norm
        if 'efficientnet' in image_model_name:
            # Load the pre-trained EfficientNet-B0 NoisyStudent model
            efficientnet = timm.create_model(image_model_name, pretrained=True)

            # Extract layers up to the mid-level layer
            blocks = list(efficientnet.blocks.children())[:mid_level_layer]
            self.features = nn.Sequential(efficientnet.conv_stem, efficientnet.bn1, *blocks)

        elif image_model_name == 'mobilenet':
            # Load the pre-trained MobileNet model
            mobilenet = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")

            # Extract layers up to the mid-level layer
            self.features = nn.Sequential(*list(mobilenet.features.children())[:mid_level_layer])
        else:
            raise ValueError('Wrong image model name')

        # Freeze the layers if specified
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        # Flatten the output
        self.flatten = nn.Flatten()

        # BatchNorm after the linear layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Get the number of output channels at the specified mid_level_layer
        dummy_input = torch.randn(1, 3, *im_size)  # Assuming input size of 224x224
        mid_level_output = self.flatten(self.features(dummy_input))
        num_channels = mid_level_output.size(1)

        # Additional trainable fully connected layer
        self.additional_fc = nn.Linear(num_channels, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.additional_fc(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class TabNetEncoder(nn.Module):
    def __init__(self, tabnet_model):
        super(TabNetEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tabnet_model = tabnet_model.to(self.device)
        self.embedder = tabnet_model.embedder
        self.tabnet = tabnet_model.tabnet
        self.out_features = tabnet_model.n_d

        # Remove the final mapping layer
        self.tabnet.final_mapping = nn.Identity()

    def forward(self, x):
        # Forward pass through the embedding layers
        x = self.embedder(x)

        # Forward pass through the TabNet encoder layers
        x = self.tabnet(x)

        return x


class SimpleEncoder(nn.Module):
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

        # Embedding layers for categorical variables
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dims[i], cat_emb_dims[i]) for i in cat_idxs
        ])

        # Fully connected layers
        self.fc1 = nn.Linear(input_size - len(self.cat_idxs) + sum(cat_emb_dims), encoder_hidden_size)
        self.bn1 = nn.BatchNorm1d(encoder_hidden_size)

        self.fc2 = nn.Linear(encoder_hidden_size, encoder_hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(encoder_hidden_size // 2)

        self.dropout = nn.Dropout(dropout_prob)

        # Output layer
        self.output_layer = nn.Linear(encoder_hidden_size // 2, output_size)

    def forward(self, x):
        x_cat = x[:, self.cat_idxs].long()
        num_idxs = [i for i in range(self.input_size) if i not in self.cat_idxs]
        x_num = x[:, num_idxs]

        # Embed categorical variables
        cat_embedded = [
            embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        cat_embedded = torch.cat(cat_embedded, 1)

        # Concatenate categorical and numerical features
        x = torch.cat([cat_embedded, x_num], 1)

        # Fully connected layers with batch normalization and dropout
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))

        # Output layer
        x = self.output_layer(x)

        return x


class SimpleAttentionModule(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            # nn.BatchNorm1d(1),
            nn.Dropout(dropout)
        )
        self.values_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout)
            )

    def forward(self, image_sequence):
        # Apply linear layer for values
        image_values_sequence = self.values_layer(image_sequence)

        # Apply attention mechanism
        attention_weights = F.softmax(self.attention(image_sequence), dim=1)
        attended_image_features = torch.sum(attention_weights * image_values_sequence, dim=1)

        return attended_image_features


class GatedImageAttention(nn.Module):
    def __init__(self, n_image_features, n_heads, dropout):
        super(GatedImageAttention, self).__init__()

        self.n_heads = n_heads
        self.n_image_features = n_image_features

        self.attention_weights = nn.Linear(n_image_features, n_heads)
        self.gating_layer = nn.Linear(n_image_features, n_heads)
        self.gating_activation = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.linear_out = nn.Linear(n_heads * n_image_features, n_image_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features_sequence):
        # Calculate attention weights
        attention_weights = self.attention_weights(image_features_sequence)
        attention_weights = F.softmax(attention_weights, dim=1)
        attention_weights = self.dropout(attention_weights)

        # Calculate gating values
        gating_values = self.gating_activation(self.gating_layer(image_features_sequence))
        gating_values = self.dropout(gating_values)

        # Apply gated attention
        gated_image_representation = (attention_weights * gating_values).unsqueeze(2) *\
            image_features_sequence.unsqueeze(-1)
        gated_image_representation = gated_image_representation.sum(dim=1)

        gated_image_representation = self.flatten(gated_image_representation)

        if self.n_heads > 1:
            return self.linear_out(gated_image_representation)

        return gated_image_representation


class ImageTransformer(nn.Module):
    def __init__(self, num_heads, hidden_size, num_layers, dropout):
        super(ImageTransformer, self).__init__()

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 2,
            ),
            num_layers=num_layers,
        )

        # Aggregation Layer (Mean Pooling)
        self.mean_pooling = nn.AdaptiveAvgPool1d(1)

        # Feedforward Neural Network
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Aggregation Layer (Mean Pooling)
        x = self.mean_pooling(x.permute(0, 2, 1)).squeeze()

        # Feedforward Neural Network
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
