# ml_pipeline_updated.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import optuna
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import inspect
import json # For saving feature list

# Import configurations
import config

# MLflow (optional)
if config.ENABLE_MLFLOW:
    try:
        import mlflow
        import mlflow.sklearn # or relevant mlflow.<flavor>
        logger_mlflow = logging.getLogger("mlflow") # To potentially silence verbose mlflow logs if needed
        # logger_mlflow.setLevel(logging.WARNING)
    except ImportError:
        config.ENABLE_MLFLOW = False
        logging.warning("MLflow not installed or import failed. Disabling MLflow logging.")


# Ensure output directory exists
os.makedirs(config.MODEL_OUTPUT_PATH, exist_ok=True)

# Logging setup
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
                    format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

logger.info(f"Model output directory: {config.MODEL_OUTPUT_PATH}")
logger.info(f"MLflow enabled: {config.ENABLE_MLFLOW}")
if config.ENABLE_MLFLOW:
    logger.info(f"MLflow tracking URI: {config.MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment name: {config.MLFLOW_EXPERIMENT_NAME}")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


# --- DATA LOADING ---
PROCESSED_DATA_FULL_PATH = os.path.join(config.PROCESSED_OUTPUT_FOLDER, config.FINAL_FEATURE_FILENAME)
logger.info(f"Loading preprocessed data from: {PROCESSED_DATA_FULL_PATH}")
if not os.path.exists(PROCESSED_DATA_FULL_PATH):
    logger.error(f"FATAL: Processed data file not found: {PROCESSED_DATA_FULL_PATH}. Run app_updated.py first.")
    raise FileNotFoundError(f"Processed data file not found: {PROCESSED_DATA_FULL_PATH}")
try:
    data = pd.read_csv(PROCESSED_DATA_FULL_PATH)
    logger.info(f"Data loaded successfully. Shape: {data.shape}")
except Exception as e:
    logger.error(f"Error loading data: {e}", exc_info=True); raise
if data.empty:
    logger.error("Loaded dataframe is empty."); raise ValueError("Loaded dataframe is empty.")

# --- FEATURE AND TARGET PREPARATION ---
if config.CONFIG_TARGET_COL_NAME not in data.columns:
    logger.error(f"FATAL: Target column '{config.CONFIG_TARGET_COL_NAME}' not found in data."); raise ValueError(f"Target column '{config.CONFIG_TARGET_COL_NAME}' not found.")
y = data[config.CONFIG_TARGET_COL_NAME].copy()

exclude_cols_for_features = [config.CONFIG_TARGET_COL_NAME]
for col_to_check in ['Datetime', 'Date', 'Time', 'Datetime_str']: # Add any other specific ID or helper columns
    if col_to_check in data.columns:
        exclude_cols_for_features.append(col_to_check)
exclude_cols_for_features = list(set(exclude_cols_for_features))

feature_names = [col for col in data.columns if col not in exclude_cols_for_features]
if not feature_names:
    logger.error("FATAL: No feature columns identified after exclusions."); raise ValueError("No feature columns.")
X = data[feature_names].copy()

logger.info(f"Target column: '{config.CONFIG_TARGET_COL_NAME}' (derived using FUTURE_PERIOD={config.CONFIG_FUTURE_PERIOD} in app_updated.py)")
logger.info(f"Number of features: {len(feature_names)}. Features (first 5): {feature_names[:5]}...")
logger.info(f"Shape of X: {X.shape}, Shape of y: {y.shape}. Target distribution:\n{y.value_counts(normalize=True)}")

if X.isnull().values.any():
    logger.warning(f"Features X contain NaN values! Filling with 0. Review app_updated.py."); X = X.fillna(0)
if y.isnull().values.any():
    logger.error(f"FATAL: Target y contains NaN values!"); raise ValueError("Target y contains NaN values.")

# --- OPTUNA HYPERPARAMETER OPTIMIZATION ---
def objective(trial):
    # Optional: Log Optuna trial with MLflow if enabled
    # with mlflow.start_run(nested=True, run_name=f"OptunaTrial_{trial.number}"):
    #     mlflow.log_param("trial_number", trial.number)

    model_name = trial.suggest_categorical('model', ['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression'])
    # if config.ENABLE_MLFLOW: mlflow.log_param("model_type_trial", model_name)
    model = None
    try:
        if model_name == "RandomForest":
            params = {'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300, step=25), # Adjusted range
                      'max_depth': trial.suggest_int('rf_max_depth', 3, 20), # Adjusted range
                      'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 30), # Adjusted
                      'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 20), # Adjusted
                      'class_weight': trial.suggest_categorical('rf_class_weight', ['balanced', 'balanced_subsample', None]),
                      'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', 0.5, 0.7, None]),
                      'random_state': 42, 'n_jobs': -1}
            model = RandomForestClassifier(**params)
        elif model_name == "XGBoost":
            params = {'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300, step=25),
                      'max_depth': trial.suggest_int('xgb_max_depth', 2, 10), # Adjusted
                      'learning_rate': trial.suggest_float('xgb_learning_rate', 0.005, 0.2, log=True), # Adjusted
                      'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0), # Adjusted
                      'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0), # Adjusted
                      'gamma': trial.suggest_float('xgb_gamma', 0, 1.0), # Adjusted
                      'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-7, 5.0, log=True), # Adjusted
                      'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-7, 5.0, log=True), # Adjusted
                      'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'} # 'use_label_encoder': False removed for XGB >1.3
            if hasattr(xgb.XGBClassifier(), 'use_label_encoder'): # Check for older XGBoost
                 params['use_label_encoder'] = False
            model = xgb.XGBClassifier(**params)
        elif model_name == "LightGBM":
            params = {'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 300, step=25),
                      'max_depth': trial.suggest_int('lgb_max_depth', 3, 15), # Adjusted
                      'learning_rate': trial.suggest_float('lgb_learning_rate', 0.005, 0.2, log=True),
                      'num_leaves': trial.suggest_int('lgb_num_leaves', 10, 150), # Adjusted
                      'subsample': trial.suggest_float('lgb_subsample', 0.5, 1.0),
                      'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.5, 1.0),
                      'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-7, 5.0, log=True),
                      'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-7, 5.0, log=True),
                      'min_child_samples': trial.suggest_int('lgb_min_child_samples', 10, 100), # Adjusted
                      'class_weight': trial.suggest_categorical('lgb_class_weight', ['balanced', None]),
                      'random_state': 42, 'n_jobs': -1, 'verbose': -1}
            model = lgb.LGBMClassifier(**params)
        else: # LogisticRegression
            params = {'C': trial.suggest_float('lr_C', 1e-4, 100.0, log=True), # Adjusted
                      'penalty': trial.suggest_categorical('lr_penalty', ['l1', 'l2']),
                      'solver': 'saga',
                      'max_iter': trial.suggest_int('lr_max_iter', 300, 3000), # Adjusted
                      'class_weight': trial.suggest_categorical('lr_class_weight', ['balanced', None]),
                      'random_state': 42, 'n_jobs': -1}
            model = LogisticRegression(**params)

        # if config.ENABLE_MLFLOW: mlflow.log_params(params)

        f1_scores_cv = []
        ts_split = TimeSeriesSplit(n_splits=config.N_SPLITS_CV_OPTUNA)
        for fold_idx, (train_idx, val_idx) in enumerate(ts_split.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if len(X_train) < config.MIN_SAMPLES_CV_TRAIN_FOLD_OPTUNA or X_val.empty: continue
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = f1_score(y_val, preds, average='weighted', zero_division=0)
            f1_scores_cv.append(score)

        if not f1_scores_cv: return 0.0
        mean_f1 = np.mean(f1_scores_cv)
        logger.debug(f"Trial {trial.number} ({model_name}): Mean F1 (weighted) = {mean_f1:.4f}")
        # if config.ENABLE_MLFLOW: mlflow.log_metric("mean_cv_f1_weighted", mean_f1)
        return mean_f1
    except Exception as e:
        logger.error(f"Optuna Trial {trial.number} ({model_name}) failed: {e}", exc_info=False) # Set exc_info=False for less verbose logs during Optuna
        # if config.ENABLE_MLFLOW: mlflow.log_metric("mean_cv_f1_weighted", -1) # Log failure
        return float('-inf')

# Start main MLflow run if enabled
main_mlflow_run_id = None
if config.ENABLE_MLFLOW:
    try:
        active_run = mlflow.start_run(run_name="Forex_Model_Training_Pipeline")
        main_mlflow_run_id = active_run.info.run_id
        mlflow.log_param("N_TRIALS_OPTUNA", config.N_TRIALS_OPTUNA)
        mlflow.log_param("N_SPLITS_CV_OPTUNA", config.N_SPLITS_CV_OPTUNA)
        mlflow.log_param("CONFIG_FUTURE_PERIOD", config.CONFIG_FUTURE_PERIOD)
        mlflow.log_param("PROCESSED_DATA_PATH", PROCESSED_DATA_FULL_PATH)
        mlflow.log_param("data_shape", str(data.shape))
    except Exception as e:
        logger.error(f"Failed to start main MLflow run: {e}")
        config.ENABLE_MLFLOW = False # Disable if start fails

logger.info(f"🔍 Starting hyperparameter optimization with Optuna ({config.N_TRIALS_OPTUNA} trials)...")
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=10))
try:
    study.optimize(objective, n_trials=config.N_TRIALS_OPTUNA, timeout=7200) # 2-hour timeout
except Exception as e:
    logger.error(f"Optuna study failed: {e}", exc_info=True);
    if config.ENABLE_MLFLOW and main_mlflow_run_id: mlflow.end_run(status="FAILED")
    raise
if not study.trials or not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
     logger.error("Optuna study failed to complete any trials successfully.")
     if config.ENABLE_MLFLOW and main_mlflow_run_id: mlflow.end_run(status="FAILED")
     raise RuntimeError("Optuna study failed to complete any trials successfully.")


best_trial = study.best_trial
best_params_optuna = best_trial.params
best_model_name = best_params_optuna['model']
best_f1_score_optuna = best_trial.value

logger.info(f"\n--- Optuna Optimization Results ---")
logger.info(f"Best model: {best_model_name}, Best F1 (weighted CV): {best_f1_score_optuna:.4f}")
logger.info(f"Best parameters: {best_params_optuna}")

if config.ENABLE_MLFLOW and main_mlflow_run_id:
    mlflow.log_param("best_model_type", best_model_name)
    mlflow.log_metric("best_optuna_cv_f1_weighted", best_f1_score_optuna)
    mlflow.log_params({f"best_param_{k}": v for k,v in best_params_optuna.items()})


# --- FINAL MODEL PREPARATION AND TRAINING ---
logger.info("\n🏗️ Building final model with best parameters...")
final_model_hyperparams = best_params_optuna.copy()
final_model_hyperparams.pop('model')
common_fixed = {'random_state': 42, 'n_jobs': -1}
if best_model_name == 'XGBoost':
    common_fixed.update({'eval_metric': 'logloss'})
    if hasattr(xgb.XGBClassifier(), 'use_label_encoder'):
         common_fixed['use_label_encoder'] = False
elif best_model_name == 'LightGBM': common_fixed.update({'verbose': -1})
elif best_model_name == 'LogisticRegression':
    if 'solver' not in final_model_hyperparams: final_model_hyperparams['solver'] = 'saga'
    if 'max_iter' not in final_model_hyperparams : final_model_hyperparams['max_iter'] = 3000

final_constructor_params = {**final_model_hyperparams, **common_fixed}
model_map = {'RandomForest': RandomForestClassifier, 'XGBoost': xgb.XGBClassifier,
             'LightGBM': lgb.LGBMClassifier, 'LogisticRegression': LogisticRegression}
ModelClass = model_map[best_model_name]
sig = inspect.signature(ModelClass.__init__)
valid_params = {k: v for k, v in final_constructor_params.items() if k in sig.parameters}
logger.info(f"Final constructor parameters for {best_model_name}: {valid_params}")
final_model = ModelClass(**valid_params)

logger.info("\n🔄 Training final model on ALL available data (X, y)...")
try:
    final_model.fit(X, y) # X here uses the 'feature_names' list
    logger.info("✅ Final model trained successfully.")
except Exception as e:
    logger.error(f"Error training final model: {e}", exc_info=True);
    if config.ENABLE_MLFLOW and main_mlflow_run_id: mlflow.end_run(status="FAILED")
    raise

# --- SAVE FEATURE LIST ---
feature_list_path = os.path.join(config.MODEL_OUTPUT_PATH, config.FEATURE_LIST_FILENAME)
try:
    with open(feature_list_path, 'w') as f:
        json.dump(feature_names, f) # feature_names is the list of columns used for X
    logger.info(f"✅ Feature list saved to: {feature_list_path}")
    if config.ENABLE_MLFLOW and main_mlflow_run_id:
        mlflow.log_artifact(feature_list_path, "feature_info")
except Exception as e:
    logger.error(f"Error saving feature list: {e}")


# --- FEATURE IMPORTANCE (if applicable) ---
importances = None
if hasattr(final_model, 'feature_importances_'): importances = final_model.feature_importances_
elif hasattr(final_model, 'coef_'): importances = np.abs(final_model.coef_[0]) if final_model.coef_.ndim > 1 else np.abs(final_model.coef_)

if importances is not None and len(feature_names) == len(importances):
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    logger.info("\nTop 15 Features (from final model):"); print(fi_df.head(15))
    fi_path_csv = os.path.join(config.MODEL_OUTPUT_PATH, 'feature_importance.csv')
    fi_df.to_csv(fi_path_csv, index=False)

    plt.figure(figsize=(10, max(6, len(fi_df.head(20)) * 0.35)))
    sns.barplot(x='Importance', y='Feature', data=fi_df.head(20), palette="viridis_r")
    plt.title('Top 20 Feature Importances'); plt.tight_layout()
    fi_plot_path = os.path.join(config.MODEL_OUTPUT_PATH, 'feature_importance_plot.png');
    plt.savefig(fi_plot_path); plt.close()
    logger.info(f"Feature importance plot saved to {fi_plot_path}")
    if config.ENABLE_MLFLOW and main_mlflow_run_id:
        mlflow.log_artifact(fi_path_csv, "feature_info")
        mlflow.log_artifact(fi_plot_path, "plots")
else: logger.warning("Could not extract or align feature importances.")

# --- SAVE MODEL AND PARAMETERS ---
model_save_path = os.path.join(config.MODEL_OUTPUT_PATH, config.SAVED_MODEL_FILENAME)
joblib.dump(final_model, model_save_path)
logger.info(f"✅ Final model saved to: '{model_save_path}'")

if config.ENABLE_MLFLOW and main_mlflow_run_id:
    # Log the model using MLflow's model persistence
    # This depends on the model type, e.g., mlflow.sklearn.log_model for scikit-learn models
    # Example for sklearn compatible models:
    try:
        # Infer signature for MLflow model logging
        # This can be tricky for some complex models or pipelines
        # from mlflow.models.signature import infer_signature
        # signature = infer_signature(X_train_sample, model.predict(X_train_sample)) # Using a sample
        # mlflow.sklearn.log_model(final_model, "forex_model", signature=signature)
        mlflow.sklearn.log_model(final_model, "forex_model_files", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)

        logger.info(f"✅ Model also logged to MLflow artifact store.")
    except Exception as e:
        logger.error(f"Failed to log model to MLflow: {e}")


params_save_path = os.path.join(config.MODEL_OUTPUT_PATH, config.SAVED_PARAMS_FILENAME)
try:
    with open(params_save_path, 'w') as f:
        f.write(f"Best Model Type (Optuna): {best_model_name}\n")
        f.write(f"Best Optuna CV F1 Score (weighted): {best_f1_score_optuna:.4f}\n")
        f.write(f"Target Definition FUTURE_PERIOD: {config.CONFIG_FUTURE_PERIOD}\n")
        f.write("\n--- Optuna Best Hyperparameters (includes 'model' key) ---\n")
        for k, v in best_params_optuna.items(): f.write(f"{k}: {v}\n")
        f.write("\n--- Final Model Constructor Parameters (used for saving) ---\n")
        for k, v in valid_params.items(): f.write(f"{k}: {v}\n")
        # Feature names are now saved in a separate JSON file, but can list them here too
        f.write(f"\n--- Number of Features Used: {len(feature_names)} ---\n")
        f.write(f"Feature list saved to: {config.FEATURE_LIST_FILENAME}\n")
    logger.info(f"✅ Model parameters and info saved to: {params_save_path}")
    if config.ENABLE_MLFLOW and main_mlflow_run_id:
        mlflow.log_artifact(params_save_path, "model_info")
except Exception as e: logger.error(f"Error saving parameters: {e}")


if config.ENABLE_MLFLOW and main_mlflow_run_id:
    mlflow.end_run()

logger.info("\n🏁 ML pipeline completed!")
logger.info(f"Next step: Use '{model_save_path}' in 'backtest_updated.py'.")