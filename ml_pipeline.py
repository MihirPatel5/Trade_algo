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

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = os.path.join('forex_preprocessed', 'finalfeature.csv') # From app_py_v4.py
MODEL_OUTPUT_PATH = 'forex_models'
SAVED_MODEL_FILENAME = 'best_trading_model.pkl' 
SAVED_PARAMS_FILENAME = 'model_parameters.txt'

TARGET_COL_NAME = 'Target' # Must match the target column name from app_py_v4.py
# CRITICAL: This MUST be consistent with CONFIG_FUTURE_PERIOD in app_py_v4.py
# Used here for logging, validation, and context.
CONFIG_FUTURE_PERIOD = 2 

# Optuna Settings
N_SPLITS_CV_OPTUNA = 5             
N_TRIALS_OPTUNA = 50 # Can increase for more thorough search, e.g., 50-100
MIN_SAMPLES_CV_TRAIN_FOLD_OPTUNA = 100 # Min samples for more stable CV folds in Optuna

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
logger.info(f"Model output directory: {MODEL_OUTPUT_PATH}")

# --- DATA LOADING ---
logger.info(f"Loading preprocessed data from: {PROCESSED_DATA_PATH}")
if not os.path.exists(PROCESSED_DATA_PATH):
    logger.error(f"FATAL: Processed data file not found: {PROCESSED_DATA_PATH}. Run app_py_v4.py first.")
    raise FileNotFoundError(f"Processed data file not found: {PROCESSED_DATA_PATH}")
try:
    data = pd.read_csv(PROCESSED_DATA_PATH)
    logger.info(f"Data loaded successfully. Shape: {data.shape}")
except Exception as e:
    logger.error(f"Error loading data: {e}", exc_info=True); raise
if data.empty:
    logger.error("Loaded dataframe is empty."); raise ValueError("Loaded dataframe is empty.")

# --- FEATURE AND TARGET PREPARATION ---
if TARGET_COL_NAME not in data.columns:
    logger.error(f"FATAL: Target column '{TARGET_COL_NAME}' not found in data."); raise ValueError(f"Target column '{TARGET_COL_NAME}' not found.")
y = data[TARGET_COL_NAME].copy()

# Define feature columns: Exclude target and any explicit non-feature columns
# 'Datetime' should have been reset to a regular column by app_py_v4.py
exclude_cols_for_features = [TARGET_COL_NAME]
# Add other known non-feature columns if they exist (e.g., 'Date', 'Time' if not used as Datetime index)
for col_to_check in ['Datetime', 'Date', 'Time', 'Datetime_str']: # Add any other specific ID or helper columns
    if col_to_check in data.columns:
        exclude_cols_for_features.append(col_to_check)
exclude_cols_for_features = list(set(exclude_cols_for_features)) # Unique

feature_names = [col for col in data.columns if col not in exclude_cols_for_features]
if not feature_names:
    logger.error("FATAL: No feature columns identified after exclusions."); raise ValueError("No feature columns.")
X = data[feature_names].copy()

logger.info(f"Target column: '{TARGET_COL_NAME}' (derived using FUTURE_PERIOD={CONFIG_FUTURE_PERIOD} in app_py_v4.py)")
logger.info(f"Number of features: {len(feature_names)}. Features (first 5): {feature_names[:5]}...")
logger.info(f"Shape of X: {X.shape}, Shape of y: {y.shape}. Target distribution:\n{y.value_counts(normalize=True)}")

# Sanity checks for NaNs (should be handled by app_py_v4.py)
if X.isnull().values.any():
    logger.warning(f"Features X contain NaN values! Filling with 0. Review app_py_v4.py."); X = X.fillna(0)
if y.isnull().values.any():
    logger.error(f"FATAL: Target y contains NaN values!"); raise ValueError("Target y contains NaN values.")

# --- OPTUNA HYPERPARAMETER OPTIMIZATION ---
def objective(trial):
    model_name = trial.suggest_categorical('model', ['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression'])
    model = None
    try: # Define hyperparameter search spaces
        if model_name == "RandomForest":
            params = {'n_estimators': trial.suggest_int('rf_n_estimators', 50, 500, step=25),
                      'max_depth': trial.suggest_int('rf_max_depth', 3, 30),
                      'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 50),
                      'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 30),
                      'class_weight': trial.suggest_categorical('rf_class_weight', ['balanced', 'balanced_subsample', None]),
                      'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', 0.5, 0.7, None]), # Added fractional options
                      'random_state': 42, 'n_jobs': -1}
            model = RandomForestClassifier(**params)
        elif model_name == "XGBoost":
            params = {'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 500, step=25),
                      'max_depth': trial.suggest_int('xgb_max_depth', 2, 15),
                      'learning_rate': trial.suggest_float('xgb_learning_rate', 0.001, 0.3, log=True),
                      'subsample': trial.suggest_float('xgb_subsample', 0.4, 1.0),
                      'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.4, 1.0),
                      'gamma': trial.suggest_float('xgb_gamma', 0, 2.0), 
                      'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True), 
                      'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
                      'random_state': 42, 'n_jobs': -1, 'use_label_encoder': False, 'eval_metric': 'logloss'}
            model = xgb.XGBClassifier(**params)
        elif model_name == "LightGBM":
            params = {'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 500, step=25),
                      'max_depth': trial.suggest_int('lgb_max_depth', 3, 20), 
                      'learning_rate': trial.suggest_float('lgb_learning_rate', 0.001, 0.3, log=True),
                      'num_leaves': trial.suggest_int('lgb_num_leaves', 10, 200), # Wider range
                      'subsample': trial.suggest_float('lgb_subsample', 0.4, 1.0), 
                      'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.4, 1.0),
                      'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 10.0, log=True),
                      'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 10.0, log=True),
                      'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 150),
                      'class_weight': trial.suggest_categorical('lgb_class_weight', ['balanced', None]),
                      'random_state': 42, 'n_jobs': -1, 'verbose': -1}
            model = lgb.LGBMClassifier(**params)
        else: # LogisticRegression
            params = {'C': trial.suggest_float('lr_C', 1e-5, 1000.0, log=True), 
                      'penalty': trial.suggest_categorical('lr_penalty', ['l1', 'l2']), 
                      'solver': 'saga', 
                      'max_iter': trial.suggest_int('lr_max_iter', 500, 5000), 
                      'class_weight': trial.suggest_categorical('lr_class_weight', ['balanced', None]),
                      'random_state': 42, 'n_jobs': -1}
            model = LogisticRegression(**params)

        f1_scores_cv = []
        ts_split = TimeSeriesSplit(n_splits=N_SPLITS_CV_OPTUNA)
        for fold_idx, (train_idx, val_idx) in enumerate(ts_split.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if len(X_train) < MIN_SAMPLES_CV_TRAIN_FOLD_OPTUNA or X_val.empty: continue
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = f1_score(y_val, preds, average='weighted', zero_division=0)
            f1_scores_cv.append(score)
        
        if not f1_scores_cv: return 0.0 # Optuna maximizes, so 0 is poor
        mean_f1 = np.mean(f1_scores_cv)
        logger.debug(f"Trial {trial.number} ({model_name}): Mean F1 (weighted) = {mean_f1:.4f}")
        return mean_f1
    except Exception as e:
        logger.error(f"Optuna Trial {trial.number} ({model_name}) failed: {e}", exc_info=False)
        return float('-inf') # Return very low for failed trials

logger.info(f"🔍 Starting hyperparameter optimization with Optuna ({N_TRIALS_OPTUNA} trials)...")
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=10))
try:
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA, timeout=7200) # e.g., 2-hour timeout
except Exception as e: logger.error(f"Optuna study failed: {e}", exc_info=True); raise
if not study.trials or not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
     raise RuntimeError("Optuna study failed to complete any trials successfully.")

best_trial = study.best_trial
best_params_optuna = best_trial.params 
best_model_name = best_params_optuna['model']
best_f1_score_optuna = best_trial.value

logger.info(f"\n--- Optuna Optimization Results ---")
logger.info(f"Best model: {best_model_name}, Best F1 (weighted CV): {best_f1_score_optuna:.4f}")
logger.info(f"Best parameters: {best_params_optuna}")

# --- FINAL MODEL PREPARATION AND TRAINING ---
logger.info("\n🏗️ Building final model with best parameters...")
final_model_hyperparams = best_params_optuna.copy()
final_model_hyperparams.pop('model') 
common_fixed = {'random_state': 42, 'n_jobs': -1} # Common params for all models
if best_model_name == 'XGBoost': common_fixed.update({'use_label_encoder': False, 'eval_metric': 'logloss'})
elif best_model_name == 'LightGBM': common_fixed.update({'verbose': -1})
elif best_model_name == 'LogisticRegression':
    if 'solver' not in final_model_hyperparams: final_model_hyperparams['solver'] = 'saga'
    if 'max_iter' not in final_model_hyperparams : final_model_hyperparams['max_iter'] = 3000 # Default

final_constructor_params = {**final_model_hyperparams, **common_fixed}
model_map = {'RandomForest': RandomForestClassifier, 'XGBoost': xgb.XGBClassifier,
             'LightGBM': lgb.LGBMClassifier, 'LogisticRegression': LogisticRegression}
ModelClass = model_map[best_model_name]
sig = inspect.signature(ModelClass.__init__) # Get valid constructor args
valid_params = {k: v for k, v in final_constructor_params.items() if k in sig.parameters}
logger.info(f"Final constructor parameters for {best_model_name}: {valid_params}")
final_model = ModelClass(**valid_params)

logger.info("\n🔄 Training final model on ALL available data (X, y)...")
try:
    final_model.fit(X, y)
    logger.info("✅ Final model trained successfully.")
except Exception as e: logger.error(f"Error training final model: {e}", exc_info=True); raise

# --- FEATURE IMPORTANCE (if applicable) ---
importances = None
if hasattr(final_model, 'feature_importances_'): importances = final_model.feature_importances_
elif hasattr(final_model, 'coef_'): importances = np.abs(final_model.coef_[0]) if final_model.coef_.ndim > 1 else np.abs(final_model.coef_)

if importances is not None and len(feature_names) == len(importances):
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    logger.info("\nTop 15 Features (from final model):"); print(fi_df.head(15))
    fi_df.to_csv(os.path.join(MODEL_OUTPUT_PATH, 'feature_importance.csv'), index=False)
    plt.figure(figsize=(10, max(6, len(fi_df.head(20)) * 0.35))) # Dynamic height
    sns.barplot(x='Importance', y='Feature', data=fi_df.head(20), palette="viridis_r")
    plt.title('Top 20 Feature Importances'); plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'feature_importance_plot.png')); plt.close()
else: logger.warning("Could not extract or align feature importances.")

# --- SAVE MODEL AND PARAMETERS ---
model_save_path = os.path.join(MODEL_OUTPUT_PATH, SAVED_MODEL_FILENAME)
joblib.dump(final_model, model_save_path)
logger.info(f"✅ Final model saved to: '{model_save_path}'")

params_save_path = os.path.join(MODEL_OUTPUT_PATH, SAVED_PARAMS_FILENAME)
try:
    with open(params_save_path, 'w') as f:
        f.write(f"Best Model Type (Optuna): {best_model_name}\n")
        f.write(f"Best Optuna CV F1 Score (weighted): {best_f1_score_optuna:.4f}\n")
        f.write(f"Target Definition FUTURE_PERIOD: {CONFIG_FUTURE_PERIOD}\n")
        f.write("\n--- Optuna Best Hyperparameters (includes 'model' key) ---\n")
        for k, v in best_params_optuna.items(): f.write(f"{k}: {v}\n")
        f.write("\n--- Final Model Constructor Parameters (used for saving) ---\n")
        for k, v in valid_params.items(): f.write(f"{k}: {v}\n")
        f.write("\n--- Features Used in Training ---\n" + "\n".join(feature_names))
    logger.info(f"✅ Model parameters and info saved to: {params_save_path}")
except Exception as e: logger.error(f"Error saving parameters: {e}")

logger.info("\n🏁 ML pipeline completed!")
logger.info(f"Next step: Use '{model_save_path}' in 'backtesting_script_v4.py'.")
