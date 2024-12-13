import pandas as pd


def add_pseudo_labels(X_train, X_test, y_train, run, config):
    """
    Adds pseudo-labels to the training dataset based on uncertainty filtering.

    Pseudo-labeling involves using predictions from a trained model on test data
    as additional training data. This is particularly useful when the test data
    includes properties similar to the training data but lacks true labels.
    The `uncertainty level` is defined as the average prediction error on the
    training data grouped by `zip_code` for each property in the test dataset.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training feature dataset.
    X_test : pd.DataFrame
        The test feature dataset.
    y_train : pd.DataFrame
        The training target dataset containing at least the column 'price'.
    run : wandb.Run
        The wandb run object to retrieve artifacts.
    config : dict
        Configuration dictionary containing relevant parameters for pseudo-labeling.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Updated training feature and target datasets with pseudo-labels added.

    Notes
    -----
    - The pseudo-labeling process filters predictions from the test set based on
      their uncertainty, defined as a threshold (`uncertainty_level` in config).
    - Pseudo-labels with lower uncertainty are appended to the training set.
    """
    if config.get('pseudo_labels'):
        # Determine the name of the pseudo-label artifact
        if isinstance(config.get('pseudo_labels'), bool):
            pseudo_label_names = "submission_peach_salad_mild_feather_stack7030_uncertainty"
        else:
            pseudo_label_names = config.get('pseudo_labels')

        # Load the pseudo-labels artifact from wandb
        artifact = run.use_artifact(f'{pseudo_label_names}:v0')
        datadir = artifact.download()
        pseudo_labels = pd.read_csv(f"{datadir}/{pseudo_label_names}.csv")

        # Retrieve the uncertainty level threshold from config
        ul = config.get('uncertainty_level', 0.5)  # Default threshold if not provided

        # Filter pseudo-labels based on uncertainty
        # Uncertainty here represents the average prediction error per zip_code in the train set
        valid_pseudo_labels = pseudo_labels[pseudo_labels.uncertainty < ul]

        # Merge the selected pseudo-labels into the training set
        X_train = pd.concat([X_train, X_test.loc[valid_pseudo_labels.index]], axis=0)
        y_train = pd.concat([
            y_train,
            valid_pseudo_labels[['id_annonce', 'price']]
        ], axis=0)

    return X_train, y_train
