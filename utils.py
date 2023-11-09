import wandb
from pytorch_tabnet.callbacks import Callback


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
