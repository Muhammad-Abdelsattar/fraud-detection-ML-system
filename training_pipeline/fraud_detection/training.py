import numpy as np
import xgboost as xgb
import optuna
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from config import STUDY_NAME, N_TRIALS, N_SPLITS_CV


def _build_pipeline(hyperparams, numerical_features, categorical_features):
    """Internal helper to build the scikit-learn pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                xgb.XGBClassifier(
                    **hyperparams, missing=np.nan, use_label_encoder=False, random_state=42
                ),
            ),
        ]
    )


def _objective(trial, X, y, numerical_features, categorical_features):
    """The objective function for Optuna HPO."""
    scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "scale_pos_weight": scale_pos_weight,
    }


    pipeline = _build_pipeline(params, numerical_features, categorical_features)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    scores = []

    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        pipeline.fit(X_train, y_train)
        preds_proba = pipeline.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds_proba)
        scores.append(score)

    return np.mean(scores)


def run_hpo_and_train(X_train, y_train, numerical_features, categorical_features):
    """
    Runs Optuna HPO to find the best hyperparameters and trains the final model.
    Returns the trained pipeline.
    """
    study = optuna.create_study(study_name=STUDY_NAME, direction="maximize")
    study.optimize(
        lambda trial: _objective(
            trial, X_train, y_train, numerical_features, categorical_features
        ),
        n_trials=N_TRIALS,
    )

    print(f"\nBest HPO trial completed with ROC-AUC: {study.best_value:.4f}")
    print("Best hyperparameters:", study.best_params)

    print("\nTraining final model with best hyperparameters...")
    final_pipeline = _build_pipeline(
        study.best_params, numerical_features, categorical_features
    )
    final_pipeline.fit(X_train, y_train)

    return final_pipeline
