import sys
import wandb
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from catboost import CatBoostRegressor
from sklearn.base import TransformerMixin, BaseEstimator
from pytorch_tabnet.tab_model import TabNetRegressor
# import yaml
# import argparse
# from icecream import ic
from wandb.catboost import WandbCallback
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn_pandas import gen_features, DataFrameMapper
from crossval_ensemble.custom_pipeline import CustomRegressionPipeline
from crossval_ensemble.custom_transformed_target_regressor import CustomTransformedTargetRegressor
from pytorch_tabnet.augmentations import RegressionSMOTE
from icecream import ic


# wandb.login()
# run = wandb.init(
#     # set the wandb project where this run will be logged
#     project='ILB',
#     # name=args['run_name'],
#     # track hyperparameters and run metadata
#     # config=wandb_config
# )
# run_name = run.name
# config = run.config

# Load the tabular data
X_train = pd.read_csv("data/X_train_J01Z4CN.csv")
y_train = pd.read_csv("data/y_train_OXxrJt1.csv")
X_test = pd.read_csv("data/X_test_BEhvxAN.csv")
y_random = pd.read_csv("data/y_random_MhJDhKK.csv")

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

datasets = [X_train, X_test]
for dataset in datasets:
    dataset['department'] = X_train.postal_code.apply(lambda x: str(x).zfill(5)[:2])
    dataset[CONT_COLS] = dataset[CONT_COLS].fillna(0.0).astype(float)
    dataset[CAT_COLS] = dataset[CAT_COLS].fillna('-1').astype(str)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# for col in ['city', 'postal_code']:
#     unknowns = set(X_valid[col].unique().tolist() + X_test[col].unique().tolist()) - set(X_train[col].unique().tolist())
#     X_valid[col] = X_valid[col].replace({value: '-1' for value in unknowns})
#     X_test[col] = X_test[col].replace({value: '-1' for value in unknowns})

categorical_dims =  {}
for cat_col in CAT_COLS:

    unknown = X_train[cat_col].nunique()

    oe = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=unknown,
        encoded_missing_value=unknown,
        dtype=int
    )
    # oe = LabelEncoder()
    X_train[cat_col] = oe.fit_transform(X_train[cat_col].values.reshape(-1, 1))
    X_valid[cat_col] = oe.transform(X_valid[cat_col].values.reshape(-1, 1))
    X_test[cat_col] = oe.transform(X_test[cat_col].values.reshape(-1, 1))
    categorical_dims[cat_col] = len(oe.categories_[0]) + 1

for cont_col in CONT_COLS:
    std = StandardScaler()
    X_train[cont_col] = std.fit_transform(X_train[cont_col].values.reshape(-1, 1))
    X_valid[cont_col] = std.transform(X_valid[cont_col].values.reshape(-1, 1))
    X_test[cont_col] = std.transform(X_test[cont_col].values.reshape(-1, 1))

cat_idxs = list(range(len(CAT_COLS)))
cat_dims = [categorical_dims[f] for f in CAT_COLS]
cat_emb_dim = [int(np.sqrt(categorical_dims[f])) for f in CAT_COLS]
cols = CAT_COLS + CONT_COLS
X_valid[CAT_COLS].min()
X_valid[CAT_COLS].max()


max_epochs = 100
aug = RegressionSMOTE(p=0.2)

clf = TabNetRegressor(
    cat_dims=cat_dims,
    cat_emb_dim=cat_emb_dim,
    cat_idxs=cat_idxs,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2)
    )
X_train[CAT_COLS].max().values

target_train = np.log(y_train.price.values.reshape(-1, 1))
target_valid = np.log(y_valid.price.values.reshape(-1, 1))

clf.fit(
    X_train=X_train[cols].values, y_train=target_train,
    eval_set=[(X_train[cols].values, target_train), (X_valid[cols].values, target_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['mae'],
    max_epochs=max_epochs,
    patience=50,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    # augmentations=aug, #aug
)



# regressor = CatBoostRegressor(
#     od_type=None,
#     eval_metric='MAPE',
#     verbose=1000
# )

# y_pred_train = estimator.fit(
#     X_train,
#     y_train.price.values,
#     X_valid=X_valid,
#     y_valid=y_valid.price.values,
#     callbacks=[WandbCallback()]
# )

# y_pred_train = estimator.predict(X_train)
# y_pred_valid = estimator.predict(X_valid)

# y_pred_test = estimator.predict(X_test)

# y_random['price'] = y_pred_test
# y_random.to_csv('./data/submission.csv', index=False)
# artifact = wandb.Artifact(name="submission", type="test predictions")
# artifact.add_file(local_path='./data/submission.csv')
# run.log_artifact(artifact)


# run.log(
#     {
#         'MAPE_train': MAPE(y_train.price, y_pred_train),
#         'MAPE_valid': MAPE(y_valid.price, y_pred_valid)
#     }
# )

# run.finish()



