import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import lightgbm as lgb
import xgboost as xgb
import optuna
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
import warnings

# --- Configuration ---
PROCESSED_DATA_PATH = 'forex_preprocessed/finalfeature.csv'
MODEL_OUTPUT_PATH = 'forex_models'
TARGET_PRICE_COL = 'Close'  # The column name for the price to predict
DATE_COL_CANDIDATES = ['datetime', 'date', 'time'] # Potential names for the date column

FUTURE_PERIOD = 2           # How many periods ahead to predict
N_SPLITS_CV = 5             # Number of splits for TimeSeriesSplit in Optuna
N_TRIALS_OPTUNA = 30        # Number of Optuna trials
WALK_FORWARD_STEP = 10      # Step size for walk-forward validation
ROLLING_WINDOW_SIZE = 500   # Window size for rolling backtest training
MIN_SAMPLES_ROLLING_BACKTEST_TRAIN = 100 # Min samples for each training window in rolling backtest

# Function to determine initial training window size for walk-forward
def get_min_initial_train_window_size(data_len):
    return max(100, int(data_len * 0.7))

MIN_SAMPLES_CV_TRAIN_FOLD = 50 # Minimum samples for a training fold in Optuna's CV
MIN_DATA_LEN_FOR_PROCESSING = 100 # Minimum data length after initial processing

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning) # XGBoost UserWarning, etc.
warnings.filterwarnings('ignore', category=FutureWarning) 

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

def load_and_prepare_data(file_path, date_col_candidates):
    """Load and prepare data with robust error handling and date parsing."""
    logger.info(f"Loading data from: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        # Auto-detect date column
        date_col = None
        temp_df_for_header = pd.read_csv(file_path, nrows=0) # Read only header
        header = temp_df_for_header.columns.tolist()
        date_col = next((col for col in header if col.lower() in date_col_candidates), None)

        if date_col:
            data = pd.read_csv(file_path, parse_dates=[date_col])
            data = data.sort_values(date_col)
            data.set_index(date_col, inplace=True)
            logger.info(f"Data loaded with '{date_col}' as index. Shape: {data.shape}")
        else:
            data = pd.read_csv(file_path)
            logger.warning("No date column found or specified. Assuming data is ordered chronologically. Index will be range-based.")
            # Ensure data is sorted if no date column (though less reliable)
            # For example, if there's an 'Unnamed: 0' or similar that implies order
            if 'Unnamed: 0' in data.columns and pd.api.types.is_numeric_dtype(data['Unnamed: 0']):
                 data = data.sort_values('Unnamed: 0').reset_index(drop=True)


        if data.empty:
            logger.error("Loaded data is empty.")
            raise ValueError("Loaded data is empty")
            
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data

    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}", exc_info=True)
        raise

def prepare_features_targets(data, price_col, future_period, min_data_len):
    """
    Prepare features (X) and target (y) for classification.
    The target is binary: 1 if future price > current price, 0 otherwise.
    """
    logger.info(f"Preparing features and targets using price column: '{price_col}' and future period: {future_period}")

    if price_col not in data.columns:
        logger.error(f"Price column '{price_col}' not found in data columns: {data.columns.tolist()}")
        raise ValueError(f"Price column '{price_col}' not found in data")
        
    processed_data = data.copy()

    # Create future price column
    processed_data['future_price'] = processed_data[price_col].shift(-future_period)
    
    # Define target: 1 if future price is higher, 0 otherwise
    # Ensure we only create target where future_price is not NaN
    processed_data['target'] = np.nan
    valid_target_idx = processed_data['future_price'].notna() & processed_data[price_col].notna()
    processed_data.loc[valid_target_idx, 'target'] = (processed_data.loc[valid_target_idx, 'future_price'] > processed_data.loc[valid_target_idx, price_col]).astype(int)

    # Drop rows with NaN in target or future_price (typically the last 'future_period' rows)
    processed_data.dropna(subset=['future_price', 'target'], inplace=True)
    
    if len(processed_data) < min_data_len:
        logger.error(f"After processing, only {len(processed_data)} samples remain. Minimum {min_data_len} required.")
        raise ValueError(f"Not enough data after processing. Need at least {min_data_len} samples, got {len(processed_data)}")

    # Define features: all columns except price_col, future_price, and target.
    # Also exclude any column that was explicitly the date column if it wasn't set as index (though it should be)
    # Or any other non-feature columns like IDs if they exist.
    # For simplicity, we assume other columns are features.
    feature_cols = [col for col in processed_data.columns if col not in [price_col, 'future_price', 'target']]
    
    # If the original price_col is not meant to be a feature, remove it.
    # Often, indicators are derived from price, and price itself might not be used directly.
    # For this example, we'll assume all other columns are features.
    # If you have specific feature engineering, those columns would be selected.
    # Example: if 'Open', 'High', 'Low' exist and 'Close' is TARGET_PRICE_COL,
    # 'Close' might be excluded from features if it's too correlated or causes leakage.
    # For now, we keep all non-target/future/price_col columns.
    # If TARGET_PRICE_COL is part of feature_cols (e.g. it wasn't dropped above), it will be included.
    # Let's explicitly remove TARGET_PRICE_COL from features to avoid direct leakage of current price if not desired.
    # However, lagged prices or returns are common features.
    # We will assume for now that all columns other than 'future_price' and 'target' are potential features.
    # The user's original script implied 'features' were all other columns.

    X = processed_data[feature_cols].copy()
    y = processed_data['target'].copy()

    # Ensure no NaN values in features (can happen with TA indicators with their own lookback)
    X.fillna(method='ffill', inplace=True) # Forward fill
    X.fillna(method='bfill', inplace=True) # Backward fill for any remaining NaNs at the beginning
    if X.isnull().any().any():
        logger.warning(f"NaN values still present in features X after ffill/bfill. Dropping rows with NaNs in X.")
        nan_in_X_rows = X.isnull().any(axis=1)
        X = X[~nan_in_X_rows]
        y = y[~nan_in_X_rows]
        processed_data = processed_data[~nan_in_X_rows]
        if len(X) < min_data_len:
            logger.error(f"After removing NaNs from X, only {len(X)} samples remain. Minimum {min_data_len} required.")
            raise ValueError(f"Not enough data after X NaN removal. Need {min_data_len}, got {len(X)}")


    logger.info(f"Features and targets prepared. X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Feature columns: {X.columns.tolist()}")
    
    return processed_data, X, y

# Global X, y for Optuna objective function (common practice with Optuna)
X_global_optuna = None
y_global_optuna = None

def objective(trial):
    """Optimization objective for Optuna with enhanced logging and validation."""
    global X_global_optuna, y_global_optuna # Use the global X and y

    if X_global_optuna is None or y_global_optuna is None:
        logger.error("Global X or y for Optuna not set.")
        return float('-inf') # Should not happen if main script calls this correctly

    model_name = trial.suggest_categorical('model', ['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression'])
    
    try:
        if model_name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300), # Reduced range for speed
                'max_depth': trial.suggest_int('rf_max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                'class_weight': trial.suggest_categorical('rf_class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestClassifier(**params)
        elif model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('xgb_max_depth', 2, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('xgb_gamma', 0, 0.5),
                'random_state': 42,
                'n_jobs': -1,
                'use_label_encoder': False, # Deprecated, set to False
                'eval_metric': 'logloss' # Or 'auc'
            }
            model = xgb.XGBClassifier(**params)
        elif model_name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 10, 80),
                'max_depth': trial.suggest_int('lgb_max_depth', 3, 15),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.2, log=True),
                'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 50),
                'class_weight': trial.suggest_categorical('lgb_class_weight', ['balanced', None]),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1 # Suppress LightGBM verbosity during tuning
            }
            model = lgb.LGBMClassifier(**params)
        else:  # LogisticRegression
            params = {
                'C': trial.suggest_float('lr_C', 0.001, 10.0, log=True), # Reduced upper C
                'penalty': trial.suggest_categorical('lr_penalty', ['l1', 'l2']),
                'solver': 'saga', # Saga supports l1 and l2
                'max_iter': trial.suggest_int('lr_max_iter', 500, 1500), # Increased max_iter
                'class_weight': trial.suggest_categorical('lr_class_weight', ['balanced', None]),
                'random_state': 42,
                'n_jobs': -1
            }
            model = LogisticRegression(**params)

        tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
        scores = []
        
        for train_idx, test_idx in tscv.split(X_global_optuna):
            X_train, X_test = X_global_optuna.iloc[train_idx], X_global_optuna.iloc[test_idx]
            y_train, y_test = y_global_optuna.iloc[train_idx], y_global_optuna.iloc[test_idx]
            
            if len(X_train) < MIN_SAMPLES_CV_TRAIN_FOLD or len(X_test) == 0:
                logger.debug(f"Skipping CV fold: train size {len(X_train)}, test size {len(X_test)}")
                continue
                
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            # Using F1-score as it's good for potentially imbalanced classes
            scores.append(f1_score(y_test, preds, average='weighted', zero_division=0)) 

        if not scores:
            logger.warning(f"Trial {trial.number} ({model_name}): No valid CV scores obtained. Returning 0.")
            return 0.0 # Return 0 if no scores, Optuna maximizes.
            
        mean_score = np.mean(scores)
        logger.debug(f"Trial {trial.number} ({model_name}): Mean F1 = {mean_score:.4f} with params {params}")
        return mean_score

    except Exception as e:
        logger.error(f"Trial {trial.number} ({model_name}) failed with {str(e)}", exc_info=False) # Set exc_info to False for less verbose logs during Optuna
        return float('-inf') # Optuna will handle this as a failed trial

def get_feature_importances(model, features_list):
    """Extract feature importances in a model-agnostic way."""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(features_list, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        # For Logistic Regression, coef_ can be (1, n_features) for binary or (n_classes, n_features)
        if model.coef_.ndim > 1 and model.coef_.shape[0] == 1: # Binary classification
             return dict(zip(features_list, np.abs(model.coef_[0])))
        elif model.coef_.ndim == 1: # Should not happen with sklearn's LogisticRegression
             return dict(zip(features_list, np.abs(model.coef_)))
        else: # Multi-class (not expected here but good to handle)
            # Averaging importance across classes or taking max, here just sum of abs for simplicity
            return dict(zip(features_list, np.sum(np.abs(model.coef_), axis=0)))
    logger.warning(f"Could not extract feature importances for model type: {type(model)}")
    return {}

def walk_forward_validation(model_template, X, y, initial_train_window_size, step_size):
    """
    Enhanced walk-forward validation with progress tracking and model versioning.
    `model_template` is an unfitted model instance.
    """
    logger.info(f"Starting Walk-Forward Validation: initial_window={initial_train_window_size}, step={step_size}")
    predictions_all = []
    actuals_all = []
    model_versions = [] # To store trained models at each step

    if len(X) < initial_train_window_size + step_size:
        logger.error(f"Not enough data for walk-forward. Need at least {initial_train_window_size + step_size} samples. Got {len(X)}")
        raise ValueError(f"Data too short for walk-forward. Need {initial_train_window_size + step_size}, have {len(X)}.")
        
    num_iterations = (len(X) - initial_train_window_size) // step_size + 1
    
    with tqdm(total=num_iterations, desc="Walk-Forward Validation") as pbar:
        for i in range(initial_train_window_size, len(X), step_size):
            train_end_idx = i
            test_start_idx = i
            test_end_idx = min(i + step_size, len(X))

            if test_start_idx >= test_end_idx: # No more data to test
                break

            X_train, y_train = X.iloc[:train_end_idx], y.iloc[:train_end_idx]
            X_test, y_test = X.iloc[test_start_idx:test_end_idx], y.iloc[test_start_idx:test_end_idx]

            if X_test.empty or y_test.empty:
                logger.warning(f"Skipping WF step at index {i}: Test set is empty.")
                continue

            current_model = clone(model_template) # Clone the base model for fresh training
            try:
                current_model.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Training failed during WF at train_end_idx {train_end_idx}: {str(e)}")
                # Optionally, append NaNs or skip predictions for this step
                predictions_all.extend([np.nan] * len(y_test)) # Or some other placeholder
                actuals_all.extend(y_test.values)
                pbar.update(1)
                continue


            model_versions.append({
                'train_end_index_val': X_train.index[-1] if isinstance(X_train.index, pd.DatetimeIndex) else train_end_idx-1,
                'model_object': current_model, # Storing the actual model object
                'features_used': X.columns.tolist(),
                'feature_importances': get_feature_importances(current_model, X.columns.tolist())
            })

            preds = current_model.predict(X_test)
            predictions_all.extend(preds)
            actuals_all.extend(y_test.values)
            pbar.update(1)

    if not predictions_all or not actuals_all:
        logger.error("No predictions collected in walk-forward validation. Check data and window sizes.")
        raise ValueError("No predictions collected in walk-forward validation")

    # Filter out NaNs if any were added due to training failures before calculating metrics
    valid_indices = [j for j, p in enumerate(predictions_all) if not pd.isna(p)]
    if not valid_indices:
        logger.error("All predictions are NaN in walk-forward validation.")
        return {}, [] # Return empty if all failed

    actuals_filtered = np.array(actuals_all)[valid_indices]
    predictions_filtered = np.array(predictions_all)[valid_indices]


    metrics = {
        'accuracy': accuracy_score(actuals_filtered, predictions_filtered),
        'precision': precision_score(actuals_filtered, predictions_filtered, zero_division=0, average='weighted'),
        'recall': recall_score(actuals_filtered, predictions_filtered, zero_division=0, average='weighted'),
        'f1': f1_score(actuals_filtered, predictions_filtered, zero_division=0, average='weighted')
    }
    logger.info(f"Walk-forward validation completed. Metrics: {metrics}")
    return metrics, model_versions


def rolling_backtest(model_template, data_full, X_features, y_target, 
                     window_size, step_size, min_train_samples):
    """
    Robust rolling window backtest implementation.
    `data_full` is the DataFrame from prepare_features_targets, containing original price, future_price, and target.
    `X_features` and `y_target` are the feature matrix and target series.
    """
    logger.info(f"Starting Rolling Backtest: window_size={window_size}, step={step_size}, min_train_samples={min_train_samples}")
    results_list = []
    
    num_iterations = (len(X_features) - window_size) // step_size + 1
    if num_iterations <= 0 :
        logger.error(f"Not enough data for rolling backtest. Data length {len(X_features)}, window_size {window_size}")
        raise ValueError(f"Data too short for rolling backtest. Needs more than {window_size} samples, got {len(X_features)}")

    with tqdm(total=num_iterations, desc="Rolling Backtest") as pbar:
        for i in range(0, len(X_features) - window_size +1, step_size):
            train_start_idx = i
            train_end_idx = i + window_size
            
            test_start_idx = train_end_idx
            test_end_idx = min(train_end_idx + step_size, len(X_features))

            if test_start_idx >= test_end_idx: # No more data to test
                break

            X_train, y_train = X_features.iloc[train_start_idx:train_end_idx], y_target.iloc[train_start_idx:train_end_idx]
            X_test, y_test = X_features.iloc[test_start_idx:test_end_idx], y_target.iloc[test_start_idx:test_end_idx]
            
            # Corresponding original data for price lookups
            data_test_period = data_full.iloc[test_start_idx:test_end_idx]

            if len(X_train) < min_train_samples or X_test.empty:
                logger.warning(f"Skipping rolling backtest window {train_start_idx}-{train_end_idx}: train size {len(X_train)} < {min_train_samples} or test empty.")
                pbar.update(1) # Ensure progress bar updates even if skipping
                continue
            
            current_model = clone(model_template)
            try:
                current_model.fit(X_train, y_train)
                preds = current_model.predict(X_test)
                probas = current_model.predict_proba(X_test)[:, 1] # Probability of class 1
                
                for j, (pred, actual_target, prob) in enumerate(zip(preds, y_test.values, probas)):
                    test_point_original_index = X_test.index[j]
                    
                    results_list.append({
                        'timestamp': test_point_original_index, # Original index (e.g., datetime)
                        'prediction': pred,                     # Model's prediction (0 or 1)
                        'actual_target': actual_target,         # Actual target (0 or 1)
                        'probability_class1': prob,             # Probability of predicting class 1
                        'current_price_at_prediction': data_test_period.iloc[j][TARGET_PRICE_COL],
                        'future_price_actual': data_test_period.iloc[j]['future_price'], # Actual price 'FUTURE_PERIOD' ahead
                        'train_window_start_idx': X_train.index[0] if isinstance(X_train.index, pd.DatetimeIndex) else train_start_idx,
                        'train_window_end_idx': X_train.index[-1] if isinstance(X_train.index, pd.DatetimeIndex) else train_end_idx-1
                    })
            except Exception as e:
                logger.warning(f"Rolling backtest training/prediction failed at window {train_start_idx}-{train_end_idx}: {str(e)}")
            pbar.update(1)
    
    if not results_list:
        logger.warning("No results collected from rolling backtest.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)
    logger.info(f"Rolling backtest completed. {len(results_df)} predictions generated.")
    return results_df

def analyze_backtest_results(results_df, model_output_path):
    """Comprehensive backtest analysis, including strategy returns and plots."""
    if results_df.empty:
        logger.warning("Backtest results DataFrame is empty. Skipping analysis.")
        return

    logger.info("Analyzing backtest results...")

    # Calculate actual percentage return for the period the prediction was made for
    # This is (future_price / current_price_at_prediction) - 1
    results_df['actual_percentage_return'] = (results_df['future_price_actual'] / results_df['current_price_at_prediction']) - 1

    # Calculate strategy returns based on a long/short approach
    # If predict UP (1): strategy_return = actual_percentage_return
    # If predict DOWN (0): strategy_return = -actual_percentage_return (profit if actual_percentage_return is negative)
    results_df['strategy_return'] = np.where(
        results_df['prediction'] == 1,
        results_df['actual_percentage_return'],
        -results_df['actual_percentage_return'] 
    )
    
    # --- Performance Metrics ---
    total_trades = len(results_df)
    winning_trades = results_df[results_df['strategy_return'] > 0].shape[0]
    losing_trades = results_df[results_df['strategy_return'] < 0].shape[0]
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    sum_profit = results_df[results_df['strategy_return'] > 0]['strategy_return'].sum()
    sum_loss = abs(results_df[results_df['strategy_return'] < 0]['strategy_return'].sum())
    profit_factor = sum_profit / sum_loss if sum_loss > 0 else np.inf # Handle division by zero

    total_return_percentage = results_df['strategy_return'].sum() * 100 # Assuming returns are fractional

    logger.info("\n--- Strategy Performance Summary ---")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Winning Trades: {winning_trades}")
    logger.info(f"Losing Trades: {losing_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Total Strategy Return: {total_return_percentage:.2f}%")
    logger.info(f"Profit Factor: {profit_factor:.2f}")

    # --- Plot Equity Curve ---
    if 'timestamp' in results_df.columns and isinstance(results_df['timestamp'].iloc[0], pd.Timestamp):
        plot_index = pd.to_datetime(results_df['timestamp'])
    else:
        plot_index = results_df.index

    cumulative_returns = (1 + results_df['strategy_return']).cumprod() -1 # Compounded returns
    
    plt.figure(figsize=(14, 7))
    # cumulative_returns.index = plot_index # Ensure index is datetime if available
    plt.plot(plot_index, cumulative_returns * 100) # Plot as percentage
    plt.title('Strategy Equity Curve (Cumulative Returns %)')
    plt.xlabel('Time' if isinstance(plot_index, pd.DatetimeIndex) else 'Trade Number')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)
    equity_curve_path = os.path.join(model_output_path, 'equity_curve.png')
    plt.savefig(equity_curve_path)
    plt.close()
    logger.info(f"Equity curve saved to {equity_curve_path}")

    # --- Plot Rolling Accuracy ---
    # This is accuracy of predicting the binary target, not strategy success directly
    results_df['correct_target_prediction'] = results_df['prediction'] == results_df['actual_target']
    rolling_accuracy_window = min(100, len(results_df) // 5 if len(results_df) > 20 else len(results_df)) # Dynamic window
    if rolling_accuracy_window > 0 :
        rolling_accuracy = results_df['correct_target_prediction'].rolling(window=rolling_accuracy_window).mean()
        plt.figure(figsize=(14, 7))
        # rolling_accuracy.index = plot_index
        plt.plot(plot_index, rolling_accuracy * 100)
        plt.title(f'Rolling Target Prediction Accuracy ({rolling_accuracy_window}-period window %)')
        plt.xlabel('Time' if isinstance(plot_index, pd.DatetimeIndex) else 'Trade Number')
        plt.ylabel('Accuracy (%)')
        plt.axhline(50, color='r', linestyle='--', label='50% Accuracy (Baseline)')
        plt.legend()
        plt.grid(True)
        rolling_accuracy_path = os.path.join(model_output_path, 'rolling_accuracy.png')
        plt.savefig(rolling_accuracy_path)
        plt.close()
        logger.info(f"Rolling accuracy plot saved to {rolling_accuracy_path}")

    # --- Classification Report for Target Prediction ---
    logger.info("\n--- Classification Report (Predicting Target Up/Down) ---")
    # Ensure actual_target and prediction are of the same type, e.g. int
    report = classification_report(results_df['actual_target'].astype(int), results_df['prediction'].astype(int), zero_division=0)
    logger.info(f"\n{report}")
    
    # Confusion Matrix
    cm = confusion_matrix(results_df['actual_target'].astype(int), results_df['prediction'].astype(int))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Down', 'Predicted Up'], yticklabels=['Actual Down', 'Actual Up'])
    plt.title('Confusion Matrix for Target Prediction')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(model_output_path, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")


def save_backtest_results(results_df, model_output_path):
    """Save backtest results DataFrame to CSV with a timestamp."""
    if results_df.empty:
        logger.info("No backtest results to save.")
        return
    
    version = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"backtest_results_{version}.csv"
    results_path = os.path.join(model_output_path, results_filename)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved backtest results to {results_path}")


if __name__ == "__main__":
    try:
        # --- 1. Data Loading and Preparation ---
        raw_data = load_and_prepare_data(PROCESSED_DATA_PATH, DATE_COL_CANDIDATES)
        
        # `processed_data` contains 'future_price' and 'target' columns, and has DatetimeIndex if date_col was found
        # `X_full` contains only feature columns, `y_full` contains only the target column
        processed_data, X_full, y_full = prepare_features_targets(
            raw_data, 
            price_col=TARGET_PRICE_COL, 
            future_period=FUTURE_PERIOD,
            min_data_len=MIN_DATA_LEN_FOR_PROCESSING
        )
        
        # Set global X, y for Optuna's objective function
        # Optuna works on a snapshot of the data, typically the full dataset available for training/validation
        X_global_optuna = X_full
        y_global_optuna = y_full

        # --- 2. Hyperparameter Optimization with Optuna ---
        logger.info("Starting Hyperparameter Optimization with Optuna...")
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=N_TRIALS_OPTUNA, n_jobs=1) # n_jobs=1 for sequential, -1 for parallel if safe

        best_trial = study.best_trial
        logger.info(f"Optuna Best Trial Value (F1-score): {best_trial.value:.4f}")
        logger.info(f"Optuna Best Parameters: {best_trial.params}")
        
        # --- 3. Prepare Best Model ---
        best_params_from_optuna = best_trial.params
        model_type = best_params_from_optuna.pop('model') # Remove 'model' key
        
        # Clean up parameter names (remove prefixes like 'rf_', 'xgb_')
        prefix_map = {'RandomForest': 'rf_', 'XGBoost': 'xgb_', 'LightGBM': 'lgb_', 'LogisticRegression': 'lr_'}
        param_prefix_to_remove = prefix_map.get(model_type, "")
        
        cleaned_best_params = {}
        for k, v in best_params_from_optuna.items():
            if k.startswith(param_prefix_to_remove):
                cleaned_best_params[k[len(param_prefix_to_remove):]] = v
            else:
                cleaned_best_params[k] = v # Keep if no prefix matched (e.g. common params)
        
        # Add fixed params like random_state, n_jobs
        common_fixed_params = {'random_state': 42, 'n_jobs': -1}
        if model_type == 'XGBoost':
            common_fixed_params['use_label_encoder'] = False
            common_fixed_params['eval_metric'] = 'logloss'
        if model_type == 'LightGBM':
             common_fixed_params['verbose'] = -1


        final_model_params = {**cleaned_best_params, **common_fixed_params}
        # Filter out any params not accepted by the model constructor
        
        model_constructor_map = {
            'RandomForest': RandomForestClassifier,
            'XGBoost': xgb.XGBClassifier,
            'LightGBM': lgb.LGBMClassifier,
            'LogisticRegression': LogisticRegression
        }
        
        SelectedModelClass = model_constructor_map[model_type]
        
        # Inspect constructor to only pass valid parameters
        import inspect
        sig = inspect.signature(SelectedModelClass.__init__)
        valid_params_for_constructor = {k: v for k, v in final_model_params.items() if k in sig.parameters}

        logger.info(f"Instantiating best model: {model_type} with params: {valid_params_for_constructor}")
        best_model_template = SelectedModelClass(**valid_params_for_constructor)
        
        # --- 4. Walk-Forward Validation (Optional but Recommended) ---
        # Uses the best model *template* (unfitted) found by Optuna
        # The initial training window size for WF
        initial_wf_train_window = get_min_initial_train_window_size(len(X_full))
        
        if len(X_full) > initial_wf_train_window + WALK_FORWARD_STEP : # Ensure enough data for at least one WF step
            logger.info("Performing Walk-Forward Validation with the best model configuration...")
            wf_metrics, wf_model_versions = walk_forward_validation(
                best_model_template, X_full, y_full, 
                initial_train_window_size=initial_wf_train_window, 
                step_size=WALK_FORWARD_STEP
            )
            logger.info(f"Walk-Forward Validation Metrics:\n{pd.Series(wf_metrics)}")
            
            # Save the last trained model from walk-forward validation as the "production" model
            if wf_model_versions:
                production_model = wf_model_versions[-1]['model_object']
                prod_model_path = os.path.join(MODEL_OUTPUT_PATH, 'production_model_from_wf.pkl')
                joblib.dump(production_model, prod_model_path)
                logger.info(f"Production model (last from WF) saved to {prod_model_path}")
            else:
                logger.warning("No model versions from walk-forward validation. Training a final model on full data.")
                production_model = clone(best_model_template).fit(X_full, y_full) # Fallback
                prod_model_path = os.path.join(MODEL_OUTPUT_PATH, 'production_model_full_data.pkl')
                joblib.dump(production_model, prod_model_path)
                logger.info(f"Production model (trained on full data) saved to {prod_model_path}")

        else:
            logger.warning("Skipping Walk-Forward Validation due to insufficient data length relative to initial window and step size.")
            logger.info("Training final model on the entire dataset instead.")
            production_model = clone(best_model_template).fit(X_full, y_full)
            prod_model_path = os.path.join(MODEL_OUTPUT_PATH, 'production_model_full_data.pkl')
            joblib.dump(production_model, prod_model_path)
            logger.info(f"Production model (trained on full data) saved to {prod_model_path}")

        # --- 5. Rolling Backtest with the "Production" Model Configuration ---
        # Use the 'production_model' (which is an instance already trained on the latest WF data or full data)
        # Or, for a true rolling backtest, you use the 'best_model_template' and retrain at each step.
        # The current rolling_backtest function retrains, so we pass the template.
        logger.info(f"Performing Rolling Backtest with model type: {model_type} and best params...")
        
        if len(X_full) > ROLLING_WINDOW_SIZE :
            backtest_results_df = rolling_backtest(
                best_model_template, # Pass the unfitted template with best params
                processed_data,      # Full data with prices for return calculation
                X_full, y_full,
                window_size=ROLLING_WINDOW_SIZE,
                step_size=WALK_FORWARD_STEP, # Can be same as WF step or different
                min_train_samples=MIN_SAMPLES_ROLLING_BACKTEST_TRAIN
            )
            
            save_backtest_results(backtest_results_df, MODEL_OUTPUT_PATH)
            
            # --- 6. Analyze and Report Backtest Results ---
            if not backtest_results_df.empty:
                analyze_backtest_results(backtest_results_df, MODEL_OUTPUT_PATH)
            else:
                logger.info("Rolling backtest did not produce results. Skipping analysis.")
        else:
            logger.warning("Skipping Rolling Backtest due to insufficient data length relative to rolling window size.")

        logger.info("Backtesting process completed successfully.")
        
    except FileNotFoundError as fnf_err:
        logger.error(f"Critical Error: Data file not found. {str(fnf_err)}")
    except ValueError as val_err:
        logger.error(f"Critical Error: Data validation or processing issue. {str(val_err)}")
    except Exception as e:
        logger.error(f"Main process failed with an unexpected error: {str(e)}", exc_info=True)
        # raise # Optionally re-raise the exception if you want the script to exit with an error code

