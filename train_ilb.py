import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from catboost import CatBoostRegressor
# import yaml
# import argparse
# import wandb
# from icecream import ic
# from wandb.catboost import WandbCallback
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn_pandas import gen_features, DataFrameMapper
from crossval_ensemble.custom_pipeline import CustomRegressionPipeline
from crossval_ensemble.custom_transformed_target_regressor import CustomTransformedTargetRegressor

sys.path.append('/train')
sys.path.append('../pricing/train')

from utils import create_preprocessor  # noqa


MAKE_SUB = False
def create_preprocessor_2(cont_cols, cat_cols):
    cat_cols_list = [[cat_col] for cat_col in cat_cols]
    cont_cols_list = [[cont_col] for cont_col in cont_cols]

    gen_numeric = gen_features(
        columns=cont_cols_list,
        classes=[
            {
                "class": SimpleImputer,
                "strategy": "constant",
                "fill_value": 0.0
            },
            {
                "class": StandardScaler
            }
        ]
    )

    gen_categories = gen_features(
        columns=cat_cols_list,
        classes=[
            {
                "class": SimpleImputer,
                "strategy": "constant",
                "fill_value": -1
            },
            {
                "class": OrdinalEncoder,
                "handle_unknown": 'use_encoded_value',
                "unknown_value": -1,
                "encoded_missing_value": -1,
                "dtype": int
            }
        ]
    )

    # DataFrameMapper construction
    preprocess_mapper = DataFrameMapper(
        [*gen_numeric, *gen_categories],
        input_df=True,
        df_out=True
    )

    return preprocess_mapper


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
    dataset[CONT_COLS] = dataset[CONT_COLS].astype(float)
    dataset[CAT_COLS] = dataset[CAT_COLS].astype(str)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

preprocess_mapper = create_preprocessor(CONT_COLS, CAT_COLS)
regressor = CatBoostRegressor(
    od_type=None,
    verbose=1000
)
tt = CustomTransformedTargetRegressor(
    regressor=regressor,
    transformer=FunctionTransformer(func=np.log, inverse_func=np.exp)
    )

estimator = CustomRegressionPipeline(Pipeline(steps=[
            ('prepro', preprocess_mapper),
            ('estimator', tt)
            ]))

y_pred_train = estimator.fit(
    X_train,
    y_train.price.values,
    X_valid=X_valid,
    y_valid=y_valid.price.values
)

y_pred_train = estimator.predict(X_train)
y_pred_valid = estimator.predict(X_valid)
print(MAPE(y_train.price, y_pred_train))
print(MAPE(y_valid.price, y_pred_valid))
y_pred_test = estimator.predict(X_test)

if MAKE_SUB:
    y_random['price'] = y_pred_test
    y_random.to_csv('./data/submission.csv', index=False)
