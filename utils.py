from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn_pandas import gen_features, DataFrameMapper
from sklearn.preprocessing import QuantileTransformer


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


def create_preprocessor(cont_cols, cat_cols):
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


def prepare_datasets(X_train, X_test, quantile_transform=None, n_quantiles=None):


    if quantile_transform is not None:
        quantile = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=quantile_transform)
        cols_to_transform = ['size', 'land_size', 'energy_performance_value', 'ghg_value']

        X_train[cols_to_transform] = quantile.fit_transform(X_train[cols_to_transform])
        X_train[cols_to_transform] = X_train[cols_to_transform].fillna(
            X_train[cols_to_transform].min() - X_train[cols_to_transform].std()
            )
        X_test[cols_to_transform] = quantile.transform(X_test[cols_to_transform])
        X_test[cols_to_transform] = X_test[cols_to_transform].fillna(
            X_train[cols_to_transform].min() - X_train[cols_to_transform].std()
            )


    datasets = [X_train, X_test]
    for dataset in datasets:
        dataset['department'] = dataset['postal_code'].apply(lambda x: str(x).zfill(5)[:2])

        dataset.loc[
            dataset.nb_rooms.isnull() & dataset.nb_bedrooms.notnull(),
            'nb_rooms'
            ] = dataset.loc[
                dataset.nb_rooms.isnull() & dataset.nb_bedrooms.notnull(),
                'nb_bedrooms'
                ] + 1

        dataset[['energy_performance_value', 'ghg_value']] = dataset[
                ['energy_performance_value', 'ghg_value']
                ].fillna(-1.0).astype(float)

        dataset[CONT_COLS] = dataset[CONT_COLS].fillna(0.0).astype(float)

        dataset[CAT_COLS] = dataset[CAT_COLS].fillna('-1').astype(str)

    return X_train, X_test


def preprocess(X_train, y_train, X_test, valid_size=0.2, random_state=0, quantile_transform=None, n_quantiles=None):

    X_train, X_test = prepare_datasets(X_train, X_test, quantile_transform=quantile_transform, n_quantiles=n_quantiles)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=valid_size,
        random_state=random_state
        )

    categorical_dims =  {}
    for cat_col in CAT_COLS:

        unknown = X_train[cat_col].nunique()

        oe = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=unknown,
            encoded_missing_value=unknown,
            dtype=int
        )

        X_train[cat_col] = oe.fit_transform(X_train[cat_col].values.reshape(-1, 1))
        X_valid[cat_col] = oe.transform(X_valid[cat_col].values.reshape(-1, 1))
        X_test[cat_col] = oe.transform(X_test[cat_col].values.reshape(-1, 1))
        categorical_dims[cat_col] = len(oe.categories_[0]) + 1

    for cont_col in CONT_COLS:
        std = StandardScaler()
        X_train[cont_col] = std.fit_transform(X_train[cont_col].values.reshape(-1, 1))
        X_valid[cont_col] = std.transform(X_valid[cont_col].values.reshape(-1, 1))
        X_test[cont_col] = std.transform(X_test[cont_col].values.reshape(-1, 1))

    return X_train, y_train, X_valid, y_valid, X_test, categorical_dims
