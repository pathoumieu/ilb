import pandas as pd
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

    # Preprocessing pipeline for continuous columns
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

    # Preprocessing pipeline for categorical columns
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


def prepare_datasets(X_train, X_test, quantile_transform=None, n_quantiles=None, clip_rooms=None):
    """
    Prepares and preprocesses training and testing datasets by applying transformations to the continuous
    and categorical columns, handling missing values, and clipping specific columns if needed.

    Parameters:
    X_train (pd.DataFrame): The training data.
    X_test (pd.DataFrame): The test data.
    quantile_transform (str or None): If specified, applies quantile transformation ('uniform' or 'normal').
    n_quantiles (int or None): The number of quantiles to use for transformation.
    clip_rooms (tuple or None): If specified, clips values of 'nb_rooms' and 'nb_bedrooms' columns between
                                the given lower and upper bounds.

    Returns:
    pd.DataFrame, pd.DataFrame: The preprocessed X_train and X_test datasets.
    """

    # Apply Quantile Transformation to specified columns if requested
    if quantile_transform is not None:
        qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=quantile_transform)

        # Columns to apply the transformation
        cols_to_transform = ['size', 'land_size', 'energy_performance_value', 'ghg_value']

        # Fit and transform training data
        X_train[cols_to_transform] = qt.fit_transform(X_train[cols_to_transform])

        # Fill missing values in transformed columns
        X_train[cols_to_transform] = X_train[cols_to_transform].fillna(
            X_train[cols_to_transform].min() - X_train[cols_to_transform].std()
        )

        # Transform the test data using the same quantile transformer
        X_test[cols_to_transform] = qt.transform(X_test[cols_to_transform])

        # Fill missing values in test data (same method as for train)
        X_test[cols_to_transform] = X_test[cols_to_transform].fillna(
            X_train[cols_to_transform].min() - X_train[cols_to_transform].std()
        )

    # Apply clipping to 'nb_rooms' and 'nb_bedrooms' columns if specified
    if clip_rooms is not None:
        # Clip the 'nb_rooms' and 'nb_bedrooms' columns between the provided range
        X_train[['nb_rooms', 'nb_bedrooms']] = X_train[['nb_rooms', 'nb_bedrooms']].clip(clip_rooms)
        X_test[['nb_rooms', 'nb_bedrooms']] = X_test[['nb_rooms', 'nb_bedrooms']].clip(clip_rooms)

    # List of datasets to preprocess (train and test)
    datasets = [X_train, X_test]

    for dataset in datasets:
        # Extract department from postal_code (first two digits)
        dataset['department'] = dataset['postal_code'].apply(lambda x: str(x).zfill(5)[:2])

        # Handle missing 'nb_rooms' by filling it based on 'nb_bedrooms' value
        dataset.loc[
            dataset.nb_rooms.isnull() & dataset.nb_bedrooms.notnull(),
            'nb_rooms'
        ] = dataset.loc[
            dataset.nb_rooms.isnull() & dataset.nb_bedrooms.notnull(),
            'nb_bedrooms'
        ] + 1

        # Fill missing values for continuous columns with a default value of -1.0
        dataset[['energy_performance_value', 'ghg_value']] = dataset[
            ['energy_performance_value', 'ghg_value']
        ].fillna(-1.0).astype(float)

        # Fill missing values in continuous columns with 0.0
        dataset[CONT_COLS] = dataset[CONT_COLS].fillna(0.0).astype(float)

        # Fill missing values in categorical columns with '-1' and convert them to string
        dataset[CAT_COLS] = dataset[CAT_COLS].fillna('-1').astype(str)

    return X_train, X_test


def preprocess(
        X_train,
        y_train,
        X_test,
        valid_size=0.2,
        random_state=0,
        quantile_transform=None,
        n_quantiles=None,
        clip_rooms=None,
        ):
    """
    Preprocesses training, validation, and test datasets by applying various transformations,
    including quantile transformations, missing value handling, ordinal encoding, and scaling.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The target variable for training.
    X_test (pd.DataFrame): The test data features.
    valid_size (float): Proportion of training data to reserve for validation.
    random_state (int): Seed for random number generator to ensure reproducibility.
    quantile_transform (str or None): If specified, applies quantile transformation ('uniform' or 'normal').
    n_quantiles (int or None): The number of quantiles to use for transformation.
    clip_rooms (tuple or None): If specified, clips values of 'nb_rooms' and 'nb_bedrooms'
    between the given bounds.

    Returns:
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, dict:
        Preprocessed X_train, y_train, X_valid, y_valid, X_test, and a dictionary containing
        the dimensions of categorical columns.
    """

    # Prepare datasets (handle missing values, transformations, and feature engineering)
    X_train, X_test = prepare_datasets(
        X_train,
        X_test,
        quantile_transform=quantile_transform,
        n_quantiles=n_quantiles,
        clip_rooms=clip_rooms,
    )

    # Split the training data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=valid_size,
        random_state=random_state
    )

    # Dictionary to store the number of categories for each categorical column
    categorical_dims = {}

    # Apply ordinal encoding to categorical columns
    for cat_col in CAT_COLS:
        # Determine the number of unique values in the column
        unknown = X_train[cat_col].nunique()

        # Initialize the OrdinalEncoder
        oe = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=unknown,
            encoded_missing_value=unknown,
            dtype=int
        )

        # Apply ordinal encoding to training, validation, and test datasets
        X_train[cat_col] = oe.fit_transform(X_train[cat_col].values.reshape(-1, 1))
        X_valid[cat_col] = oe.transform(X_valid[cat_col].values.reshape(-1, 1))
        X_test[cat_col] = oe.transform(X_test[cat_col].values.reshape(-1, 1))

        # Store the number of categories for the current categorical column
        categorical_dims[cat_col] = len(oe.categories_[0]) + 1

    # Apply standard scaling to continuous columns
    for cont_col in CONT_COLS:
        # Initialize the StandardScaler
        std = StandardScaler()

        # Fit the scaler on the training data and transform the training, validation, and test sets
        X_train[cont_col] = std.fit_transform(X_train[cont_col].values.reshape(-1, 1))
        X_valid[cont_col] = std.transform(X_valid[cont_col].values.reshape(-1, 1))
        X_test[cont_col] = std.transform(X_test[cont_col].values.reshape(-1, 1))

    return X_train, y_train, X_valid, y_valid, X_test, categorical_dims


def process_and_enrich_features(X_train, X_test, y_train, size_cutoff=1000, valid_size=0.2, random_state=0):
    """
    Cleans and enriches the training and test datasets by adding new features, handling missing values,
    and performing various transformations. This includes adding density-related features,
    calculating price per m2, and handling missing size data.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    X_test (pd.DataFrame): The test data features.
    y_train (pd.Series): The target variable for training.
    size_cutoff (int): The threshold size value above which certain properties are flagged for size cleaning.
    valid_size (float): Proportion of training data to reserve for validation.
    random_state (int): Seed for random number generator to ensure reproducibility.

    Returns:
    pd.DataFrame, pd.DataFrame:
        Processed X_train and X_test datasets with enriched features.
    """

    # Combine training and test datasets for consistent processing
    X = pd.concat([X_train, X_test], axis=0)

    # Calculate density by postal code and department
    X = X.merge(
        X.groupby('postal_code').id_annonce.count().reset_index().rename(columns={'id_annonce': 'density'}),
        on='postal_code',
        how='left'
    )
    X['department'] = X['postal_code'].apply(lambda x: str(x).zfill(5)[:2])

    X = X.merge(
        X.groupby('department').id_annonce.count().reset_index().rename(
            columns={'id_annonce': 'density_department'}
            ),
        on='department',
        how='left'
    )

    # Compute density ratio
    X['density_ratio'] = X.density / X.density_department

    # Merge target variable 'y_train' into X
    X = X.merge(y_train, how='left', on='id_annonce')

    # Clean 'size' feature for larger properties above the cutoff
    X['size_clean'] = X['size']
    X.loc[
        X.property_type.isin(['appartement', 'maison']) & (X['size'] > size_cutoff),
        'size_clean'
    ] = None

    # Create new feature 'size_ratio'
    X['size_ratio'] = X['size'] / X['land_size']

    # Calculate price per square meter
    X['price_m2'] = X['price'] / X['size_clean']

    # Create validation set
    X_train_, _ = train_test_split(
        X_train,
        test_size=valid_size,
        random_state=random_state
    )
    train_index = X_train_.index

    # Enrich dataset with price per m2 by postal_code, department, and city
    X = X.merge(
        X.iloc[train_index].groupby(['postal_code', 'property_type']).price_m2.mean().reset_index().rename(
            columns={'price_m2': 'price_m2_type_zipcode'}
        ),
        on=['postal_code', 'property_type'],
        how='left'
    )

    X = X.merge(
        X.iloc[train_index].groupby(['department', 'property_type']).price_m2.mean().reset_index().rename(
            columns={'price_m2': 'price_m2_type_deptcode'}
        ),
        on=['department', 'property_type'],
        how='left'
    )

    X = X.merge(
        X.iloc[train_index].groupby(['city', 'property_type']).price_m2.mean().reset_index().rename(
            columns={'price_m2': 'price_m2_type_city'}
        ),
        on=['city', 'property_type'],
        how='left'
    )

    # Add quantiles of price per m2 for each group (postal_code, department, city)
    for quantile, quantile_name in zip([0.25, 0.5, 0.75], ['25', '50', '75']):
        X = X.merge(
            X.iloc[train_index].groupby(
                ['postal_code', 'property_type']
                ).price_m2.quantile(quantile).reset_index().rename(
                columns={'price_m2': f'price_m2_type_zipcode{quantile_name}'}
            ),
            on=['postal_code', 'property_type'],
            how='left'
        )
        X = X.merge(
            X.iloc[train_index].groupby(
                ['department', 'property_type']
                ).price_m2.quantile(quantile).reset_index().rename(
                columns={'price_m2': f'price_m2_type_deptcode{quantile_name}'}
            ),
            on=['department', 'property_type'],
            how='left'
        )
        X = X.merge(
            X.iloc[train_index].groupby(
                ['city', 'property_type']
                ).price_m2.quantile(quantile).reset_index().rename(
                columns={'price_m2': f'price_m2_type_city{quantile_name}'}
            ),
            on=['city', 'property_type'],
            how='left'
        )

    # Add standard deviation of price per m2 for each group
    X = X.merge(
        X.iloc[train_index].groupby(['postal_code', 'property_type']).price_m2.std().reset_index().rename(
            columns={'price_m2': 'price_m2_type_zipcode_std'}
        ),
        on=['postal_code', 'property_type'],
        how='left'
    )
    X = X.merge(
        X.iloc[train_index].groupby(['department', 'property_type']).price_m2.std().reset_index().rename(
            columns={'price_m2': 'price_m2_type_deptcode_std'}
        ),
        on=['department', 'property_type'],
        how='left'
    )
    X = X.merge(
        X.iloc[train_index].groupby(['city', 'property_type']).price_m2.std().reset_index().rename(
            columns={'price_m2': 'price_m2_type_city_std'}
        ),
        on=['city', 'property_type'],
        how='left'
    )

    # Fill missing size values for properties in the training set based on price per m2
    fillna_size = X.price / X.price_m2_type_zipcode
    mask_fillna = X['size_clean'].isnull() & X.index.isin(train_index)
    X.loc[mask_fillna, 'size_clean'] = fillna_size[mask_fillna]

    # Clean size ratio
    X['size_ratio_clean'] = X['size_clean'] / X['land_size']

    # Clean number of photos (limit to 6)
    X['nb_photos_clean'] = X['nb_photos'].apply(lambda x: min(x, 6))

    # Return the cleaned and enriched datasets
    return X.loc[:len(X_train) - 1], X.loc[len(X_train):]
