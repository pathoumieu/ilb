import wandb
import zipfile
import json
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.metrics import Metric


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


def MAPE(predictions, targets):
    absolute_percentage_errors = torch.abs((targets - predictions) / targets)
    mape = torch.mean(absolute_percentage_errors)
    return mape.item()


class tabnet_mape(Metric):
    def __init__(self):
        self._name = "mape"  # write an understandable name here
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


def load_trained_tabnet(run, version="v50", freeze=True, freeze_embed=True, device="cuda"):
    """
    Loads a pre-trained TabNet model from an artifact, extracts the model parameters,
    and optionally freezes the model weights or specific layers.

    Parameters
    ----------
    run : run object
        A run object from an experiment tracking library (e.g., `wandb`), used to fetch the model artifact.
    version : str, optional
        The version of the model artifact to load (default is "v50").
    freeze : bool, optional
        Whether to freeze all the weights in the model (default is True).
    freeze_embed : bool, optional
        If True and `freeze` is also True, it will freeze the embedding layers specifically (default is True).
    device : str, optional
        The device to load the model on, can be "cuda" or "cpu" (default is "cuda").

    Returns
    -------
    model : TabNet
        The pre-trained TabNet model loaded on the specified device.

    Notes
    -----
    This function assumes that the artifact consists of a zipped model file (`tabnet_model.pt.zip`),
    and a JSON file (`model_params.json`) that contains the model parameters.
    """

    # Fetch the artifact using the experiment run and download it
    artifact = run.use_artifact(f"tabnet_model:{version}")
    datadir = artifact.download()

    # Extract the model zip file (tabnet_model.pt.zip) into the downloaded directory
    with zipfile.ZipFile(datadir + '/tabnet_model.pt.zip', 'r') as zip_ref:
        zip_ref.extractall(datadir)

    # Load model parameters from the JSON file
    with open(datadir + '/model_params.json', 'r') as f:
        model_params = json.load(f)

    # List of keys to remove from the model parameters, as they are not needed for reloading
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

    # Remove unnecessary keys from the parameters
    for key in keys_to_pop:
        model_params['init_params'].pop(key)

    # Initialize the TabNet model with the remaining parameters
    model = TabNet(
        **model_params['init_params'],
        group_attention_matrix=torch.eye(model_params['init_params']['input_dim']).to(device)
    )

    # Load the model weights from the checkpoint file
    model.load_state_dict(torch.load(datadir + '/network.pt'))

    # Freeze all parameters (i.e., prevent them from updating during training)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Optionally, freeze the embedding layers' weights if `freeze_embed` is True
    if freeze_embed and not freeze:
        for layer in model.embedder.embeddings.children():
            if isinstance(layer, nn.Embedding):
                layer.weight.requires_grad = False

    # Return the model on the specified device (either 'cuda' or 'cpu')
    return model.to(device)
