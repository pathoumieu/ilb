import os, yaml, argparse, sys
import wandb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from catboost import CatBoostRegressor
from wandb.catboost import WandbCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from crossval_ensemble.custom_pipeline import CustomRegressionPipeline
from crossval_ensemble.custom_transformed_target_regressor import CustomTransformedTargetRegressor

sys.path.append(os.getcwd())
from utils import create_preprocessor, prepare_datasets, CAT_COLS, CONT_COLS


if __name__ == "__main__":

    cfd = os.environ.get("CONFIG_FILE_DIR", f"{os.getcwd()}/pipe_ilb")
    dfd = os.environ.get("DATA_FILE_DIR", f"{os.getcwd()}/data")

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config-path', type=str, help='', default=f'{cfd}/config.yml')
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
    X_train = pd.read_csv(f"{dfd}/X_train_J01Z4CN.csv")
    y_train = pd.read_csv(f"{dfd}/y_train_OXxrJt1.csv")
    X_test = pd.read_csv(f"{dfd}/X_test_BEhvxAN.csv")
    y_random = pd.read_csv(f"{dfd}/y_random_MhJDhKK.csv")

    if config.get('use_image_features'):
        artifact = run.use_artifact("train_image_features:v0")
        datadir = artifact.download()
        train_features_df = pd.read_csv(datadir + '/X_train_image_features.csv')
        X_train = X_train.merge(train_features_df, on='id_annonce', how='left')

        artifact = run.use_artifact("test_image_features:v0")
        datadir = artifact.download()
        test_features_df = pd.read_csv(datadir + '/X_test_image_features.csv')
        X_test = X_test.merge(test_features_df, on='id_annonce', how='left')
        cont_cols = CONT_COLS + [f'image_feature_{i}' for i in range(len(train_features_df.columns) - 1)]
    else:
        cont_cols = CONT_COLS

    X_train, X_test = prepare_datasets(X_train, X_test)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

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

    estimator = CustomRegressionPipeline(Pipeline(steps=[
                ('prepro', preprocess_mapper),
                ('estimator', tt)
                ]))

    y_pred_train = estimator.fit(
        X_train,
        y_train.price.values,
        X_valid=X_valid,
        y_valid=y_valid.price.values,
        callbacks=[WandbCallback()],
        cat_features=CAT_COLS
    )

    y_pred_train = estimator.predict(X_train)
    y_pred_valid = estimator.predict(X_valid)

    y_pred_test = estimator.predict(X_test)

    y_random['price'] = y_pred_test
    y_random.to_csv(f'{dfd}/submission.csv', index=False)
    artifact = wandb.Artifact(name="submission", type="test predictions")
    artifact.add_file(local_path=f'{dfd}/submission.csv')
    run.log_artifact(artifact)


    run.log(
        {
            'MAPE_train': MAPE(y_train.price, y_pred_train),
            'MAPE_valid': MAPE(y_valid.price, y_pred_valid)
        }
    )

    run.finish()
