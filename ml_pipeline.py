import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

PROCESSED_DATA_PATH = 'forex_preprocessed/finalfeature.csv'
MODEL_OUTPUT_PATH = 'forex_models'
FUTURE_PERIOD = 1 
N_SPLITS_CV = 5
N_TRIALS_OPTUNA = 30

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
logger.info(f"Model output directory: {MODEL_OUTPUT_PATH}")

logger.info(f"Loading data from: {PROCESSED_DATA_PATH}")
if not os.path.exists(PROCESSED_DATA_PATH):
    logger.error(f"Data file not found: {PROCESSED_DATA_PATH}")
    raise FileNotFoundError(f"Data file not found: {PROCESSED_DATA_PATH}")

try:
    data_preview = pd.read_csv(PROCESSED_DATA_PATH, nrows=5)
    logger.info(f"Data columns: {data_preview.columns.tolist()}")

    date_col = None
    if 'Datetime' in data_preview.columns:
        date_col = 'Datetime'
    elif 'Date' in data_preview.columns:
        date_col = 'Date'
    elif any('date' in col.lower() for col in data_preview.columns):
        date_col = next((col for col in data_preview.columns if 'date' in col.lower()), None)

    if date_col:
        logger.info(f"Using date column: {date_col}")
        data = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=[date_col])
    else:
        logger.warning("No date column identified. Loading without parsing dates.")
        data = pd.read_csv(PROCESSED_DATA_PATH)

    logger.info(f"Data loaded successfully with shape: {data.shape}")

except Exception as e:
    logger.error(f"Error loading or processing data: {e}")
    raise

if data.empty:
    logger.error("The loaded dataframe is empty.")
    raise ValueError("The loaded dataframe is empty. Please check your data file.")

price_cols = [col for col in ['Close', 'close', 'Price', 'price'] if col in data.columns]
if not price_cols:
    logger.error("No price column found (looking for 'Close', 'close', 'Price', or 'price')")
    raise ValueError("No price column found in the data.")
price_col = price_cols[0]
logger.info(f"Using price column: {price_col}")

data['target'] = (data[price_col].shift(-FUTURE_PERIOD) > data[price_col]).astype(int)
logger.info(f"Target distribution:\n{data['target'].value_counts(normalize=True)}")

initial_rows = len(data)
data.dropna(subset=['target'], inplace=True)
rows_after_target_dropna = len(data)
logger.info(f"Dropped {initial_rows - rows_after_target_dropna} rows with NaN target.")

exclude_columns = [date_col] if date_col else []
exclude_columns.extend(['target', price_col])
exclude_columns.extend([col for col in data.columns if col.lower() in ['datetime', 'date', 'time']])
exclude_columns = list(set(col for col in exclude_columns if col in data.columns))

features = [col for col in data.columns if col not in exclude_columns]
target = 'target'

logger.info(f"Selected {len(features)} features: {features}")

X = data[features].copy()
y = data[target].copy()

if X.isnull().values.any():
    nan_counts = X.isnull().sum()
    logger.warning(f"Features contain NaN values:\n{nan_counts[nan_counts > 0]}")
    X = X.fillna(0)
    logger.info("Filled NaN values in features with 0.")
else:
     logger.info("No NaN values found in selected features.")


# Optuna objective function
def objective(trial):
    model_name = trial.suggest_categorical('model', ['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression'])

    model = None

    try:
        if model_name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=10),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        elif model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=10),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            }
            model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')

        elif model_name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=10),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 150),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
            }
            model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1)

        else:
            params = {}
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
            params['penalty'] = penalty
            params['C'] = trial.suggest_float('C', 0.001, 100.0, log=True)
            params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', None])

            if penalty == 'elasticnet':
                params['solver'] = 'saga'
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            elif penalty == 'l1':
                params['solver'] = 'saga'
            elif penalty == 'l2':
                 params['solver'] = 'saga'
            else:
                params['solver'] = 'saga'
            model = LogisticRegression(**params, random_state=42, max_iter=1500, n_jobs=-1)
        accuracies = []
        ts_split = TimeSeriesSplit(n_splits=N_SPLITS_CV)

        for train_idx, val_idx in ts_split.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if X_train.empty or y_train.empty:
                 logger.warning(f"Skipping fold due to empty train data in trial {trial.number}")
                 continue

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            accuracies.append(acc)
        if not accuracies:
            logger.warning(f"Trial {trial.number} resulted in no valid accuracy scores.")
            return 0.0

        mean_accuracy = np.mean(accuracies)
        return mean_accuracy

    except Exception as e:
        logger.error(f"Error in Trial {trial.number} ({model_name}) with params {trial.params}: {e}", exc_info=False) # Log error without full traceback noise
        return 0.0


# --- Run Optuna Study ---
logger.info(f"🔍 Starting hyperparameter optimization with Optuna ({N_TRIALS_OPTUNA} trials)...")
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

try:
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA, timeout=600)
except Exception as e:
    logger.error(f"Optimization study failed: {e}", exc_info=True)
    raise

if not study.trials:
     logger.error("Optuna study finished without any successful trials.")
     raise RuntimeError("Optuna study failed to complete any trials.")


best_trial = study.best_trial
best_params_with_model = best_trial.params
best_model_name = best_params_with_model['model']
best_value = best_trial.value

logger.info(f"\n--- Optimization Results ---")
logger.info(f"Best model: {best_model_name}")
logger.info(f"Best parameters found: {best_params_with_model}")
logger.info(f"Best average CV accuracy: {best_value:.4f}")

logger.info("\n🏗️ Building final model with best parameters...")
final_params = best_params_with_model.copy()
final_params.pop('model')
    
if best_model_name == 'RandomForest':
    final_model = RandomForestClassifier(**final_params, random_state=42, n_jobs=-1)
elif best_model_name == 'XGBoost':
    final_model = xgb.XGBClassifier(**final_params, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
elif best_model_name == 'LightGBM':
    final_model = lgb.LGBMClassifier(**final_params, random_state=42, n_jobs=-1)
else:
    if 'solver' not in final_params:
         penalty = final_params.get('penalty')
         if penalty == 'elasticnet' or penalty == 'l1':
             final_params['solver'] = 'saga'
         elif penalty is None:
              final_params['solver'] = 'saga'
         else:
              final_params['solver'] = 'saga'
         logger.info(f"Explicitly setting solver to '{final_params['solver']}' for final LogisticRegression model.")

    if final_params.get('penalty') == 'elasticnet' and 'l1_ratio' not in final_params:
         logger.error("l1_ratio missing from best params for elasticnet Logistic Regression!")
         l1_ratio_val = best_trial.params.get('l1_ratio')
         if l1_ratio_val is not None:
              final_params['l1_ratio'] = l1_ratio_val
         else:
              raise ValueError("Cannot build final ElasticNet model without l1_ratio in best params.")


    final_model = LogisticRegression(**final_params, random_state=42, max_iter=1500, n_jobs=-1)


# --- Final Evaluation ---
logger.info("\n📊 Evaluating final model with time series validation...")
final_accuracies = []
y_true_all = []
y_pred_all = []

ts_split_final = TimeSeriesSplit(n_splits=N_SPLITS_CV)
fold = 1
for train_idx, test_idx in ts_split_final.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if X_train.empty or y_train.empty or X_test.empty:
        logger.warning(f"Skipping final evaluation fold {fold} due to empty data.")
        fold += 1
        continue

    logger.info(f"Evaluating fold {fold}/{N_SPLITS_CV}...")
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    final_accuracies.append(acc)
    logger.info(f"Fold {fold} Accuracy: {acc:.4f}")

    y_true_all.extend(y_test.tolist())
    y_pred_all.extend(y_pred.tolist())
    fold += 1

if not final_accuracies:
     logger.error("Final evaluation could not be performed (no valid folds).")
else:
    logger.info(f"\nAverage accuracy across final time series folds: {np.mean(final_accuracies):.4f}")
    logger.info("Classification report (overall):")
    print(classification_report(y_true_all, y_pred_all))

    cm = confusion_matrix(y_true_all, y_pred_all)
    logger.info("Confusion matrix (overall):")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Sell (0)', 'Pred Buy (1)'], yticklabels=['True Sell (0)', 'True Buy (1)'])
    plt.title('Overall Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(MODEL_OUTPUT_PATH, 'confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix plot saved to {cm_path}")
    plt.close()


# --- Feature Importance ---
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    feature_names = X.columns
elif hasattr(final_model, 'coef_'):
    importances = np.abs(final_model.coef_[0])
    feature_names = X.columns
else:
    importances = None

if importances is not None:
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    logger.info("\nTop 15 features by importance:")
    print(feature_importance.head(15))

    # Save feature importance
    fi_path = os.path.join(MODEL_OUTPUT_PATH, 'feature_importance.csv')
    feature_importance.to_csv(fi_path, index=False)
    logger.info(f"✅ Feature importance saved to {fi_path}")

    # Plot Feature Importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    fi_plot_path = os.path.join(MODEL_OUTPUT_PATH, 'feature_importance.png')
    plt.savefig(fi_plot_path)
    logger.info(f"Feature importance plot saved to {fi_plot_path}")
    plt.close()


# --- Train Final Model on All Data ---
logger.info("\n🔄 Training final model on all available data...")
try:
    final_model.fit(X, y)
    logger.info("✅ Final model trained successfully on all data.")
except Exception as e:
    logger.error(f"Error training final model on all data: {e}", exc_info=True)
    raise

# --- Save Final Model and Parameters ---
model_path = os.path.join(MODEL_OUTPUT_PATH, 'best_trading_model.pkl')
joblib.dump(final_model, model_path)
logger.info(f"✅ Final model saved as '{model_path}'")

params_path = os.path.join(MODEL_OUTPUT_PATH, 'model_parameters.txt')
try:
    with open(params_path, 'w') as f:
        f.write(f"Best Model Found: {best_model_name}\n")
        f.write(f"Best CV Accuracy: {best_value:.4f}\n")
        f.write("\n--- Best Parameters ---\n")
        for key, val in best_params_with_model.items():
            if isinstance(val, float):
                 f.write(f"{key}: {val:.6f}\n")
            else:
                 f.write(f"{key}: {val}\n")
        f.write("\n--- Features Used ---\n")
        f.write("\n".join(features))
    logger.info(f"✅ Model parameters saved to {params_path}")
except Exception as e:
     logger.error(f"Error saving model parameters: {e}")


logger.info("\n🏁 ML pipeline completed successfully!")