import os
import sys
import yaml
import argparse
import wandb
import umap
import numpy as np
import pandas as pd
from icecream import ic
from sklearn.preprocessing import FunctionTransformer
from catboost import CatBoostRegressor
from wandb.catboost import WandbCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from crossval_ensemble.custom_transformed_target_regressor import CustomTransformedTargetRegressor
from crossval_ensemble.crossval_pipeline import CrossvalRegressionPipeline
from sklearn.decomposition import PCA

sys.path.append(os.getcwd())
from preprocess.pseudo_labels import add_pseudo_labels  # noqa
from preprocess.preprocess import create_preprocessor, prepare_datasets, process_and_enrich_features, CAT_COLS, CONT_COLS  # noqa


if __name__ == "__main__":

    config_file_dir = os.environ.get("CONFIG_FILE_DIR", f"{os.getcwd()}/pipe_ilb")
    data_file_dir = os.environ.get("DATA_FILE_DIR", f"{os.getcwd()}/data")

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config-path', type=str, help='', default=f'{config_file_dir}/config.yml')
    parser.add_argument('--run-name', type=str, help='', default=None)

    args = vars(parser.parse_args())

    # Get config and params
    with open(args['config_path']) as file_:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        default_config = yaml.load(file_, Loader=yaml.FullLoader)
    wandb_config = {key: value for key, value in default_config.items() if key != 'model_params'}
    wandb_config.update(default_config['model_params'])

    wandb.login()
    tags = default_config['tags']
    run = wandb.init(
        # set the wandb project where this run will be logged
        project='ILB',
        tags=tags,
        name=args['run_name'],
        # track hyperparameters and run metadata
        config=wandb_config
    )
    run_name = run.name
    config = run.config

    # Load the tabular data
    X_train = pd.read_csv(f"{data_file_dir}/X_train_J01Z4CN.csv")
    y_train = pd.read_csv(f"{data_file_dir}/y_train_OXxrJt1.csv")
    X_test = pd.read_csv(f"{data_file_dir}/X_test_BEhvxAN.csv")
    y_random = pd.read_csv(f"{data_file_dir}/y_random_MhJDhKK.csv")

    X_train, X_test = process_and_enrich_features(
        X_train,
        X_test,
        y_train,
        config.get('size_cutoff'),
        valid_size=config.get('valid_size'),
        random_state=0
        )

    X_train, X_test = prepare_datasets(
        X_train,
        X_test,
        quantile_transform=config.get('quantile_tranform'),
        n_quantiles=config.get('n_quantiles'),
        clip_rooms=config.get('clip_rooms')
        )

    artifact_version = config.get('image_features_version')
    if config.get('use_image_features'):
        artifact = run.use_artifact(f"train_image_features:{artifact_version}")
        datadir = artifact.download()
        train_features_df = pd.read_csv(datadir + '/X_train_image_features.csv')
        artifact = run.use_artifact(f"test_image_features:{artifact_version}")
        datadir = artifact.download()
        test_features_df = pd.read_csv(datadir + '/X_test_image_features.csv')

        X_train = X_train.merge(train_features_df, on='id_annonce', how='left')
        X_test = X_test.merge(test_features_df, on='id_annonce', how='left')

        image_feature_cols = [f'image_feature_{i}' for i in range(len(train_features_df.columns) - 1)]
        cont_cols = CONT_COLS + image_feature_cols

        n_pca_features = config.get('n_pca_features')
        if n_pca_features is not None:
            pca = PCA(n_components=n_pca_features)
            pca_feature_cols = [f'pca_feature_{i}' for i in range(n_pca_features)]
            pca.fit(pd.concat([X_train[image_feature_cols], X_test[image_feature_cols]], axis=0))
            X_train[pca_feature_cols] = pca.transform(X_train[image_feature_cols])
            X_test[pca_feature_cols] = pca.transform(X_test[image_feature_cols])
            cont_cols = CONT_COLS + pca_feature_cols

        n_umap_features = config.get('n_umap_features')
        if n_umap_features is not None:
            umap_model = umap.UMAP(n_components=n_umap_features, random_state=42)
            umap_feature_cols = [f'umap_feature_{i}' for i in range(n_umap_features)]
            umap_model.fit(pd.concat([X_train[image_feature_cols], X_test[image_feature_cols]], axis=0))
            X_train[umap_feature_cols] = umap_model.transform(X_train[image_feature_cols])
            X_test[umap_feature_cols] = umap_model.transform(X_test[image_feature_cols])
            cont_cols = CONT_COLS + umap_feature_cols

    else:
        cont_cols = CONT_COLS

    if config.get('columns_to_add') is not None:
        cont_cols += config.get('columns_to_add')
    if config.get('dont_use_size'):
        cont_cols.remove('size')

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=config.get('valid_size'), random_state=0
        )

    # Add pseudo labels to enrich data
    X_train, y_train = add_pseudo_labels(X_train, X_test, y_train, run, config)

    preprocess_mapper = create_preprocessor(cont_cols, CAT_COLS)

    run_name = run.name
    config = run.config
    # Fit model
    model_params = {
        'early_stopping_rounds': config.get('early_stopping_rounds'),
        'eval_metric': config.get('eval_metric'),
        'iterations': config.get('iterations'),
        'learning_rate': config.get('learning_rate'),
        'loss_function': config.get('loss_function'),
        'max_depth': config.get('max_depth'),
        'one_hot_max_size': config.get('one_hot_max_size'),
        'random_seed': config.get('random_seed'),
        'use_best_model': True,
        'od_type': None,
        'verbose': config.get('verbose'),
        'thread_count': 8
        }

    regressor = CatBoostRegressor(**model_params)

    tt = CustomTransformedTargetRegressor(
        regressor=regressor,
        transformer=FunctionTransformer(func=np.log, inverse_func=np.exp)
        )

    estimator = CrossvalRegressionPipeline(
        steps=[
            ('prepro', preprocess_mapper),
            ('estimator', tt)
        ],
        n_folds=config.get('n_folds')
    )

    y_pred_train = estimator.fit(
        X_train,
        y_train.price.values,
        # X_valid=X_valid,
        # y_valid=y_valid.price.values,
        callbacks=[WandbCallback()],
        cat_features=CAT_COLS
    )

    ic(estimator.get_feature_importance(prettified=True))

    y_pred_train = estimator.predict(X_train)
    y_pred_valid = estimator.predict(X_valid)

    y_pred_test = estimator.predict(X_test)

    y_random['price'] = y_pred_test
    y_random.to_csv(f'{data_file_dir}/submission.csv', index=False)
    artifact = wandb.Artifact(name="submission", type="test predictions")
    artifact.add_file(local_path=f'{data_file_dir}/submission.csv')
    run.log_artifact(artifact)

    run.log(
        {
            'MAPE_train': MAPE(y_train.price, y_pred_train),
            'MAPE_valid': MAPE(y_valid.price, y_pred_valid)
        }
    )

    run.finish()
