import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit # Keep for potential re-evaluation, not Optuna
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.base import clone # To clone the loaded model for retraining in WF/Rolling
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
import warnings

# --- Configuration ---
PROCESSED_DATA_PATH = os.path.join('forex_preprocessed', 'finalfeature.csv')
MODEL_OUTPUT_PATH = 'forex_models' # Where models are saved by ml_pipeline.py
LOAD_MODEL_NAME = 'best_trading_model.pkl' # Model to load for backtesting
TARGET_COL_NAME_IN_DATA = 'Target' # Must match the target column name in finalfeature.csv

TARGET_PRICE_COL = 'Close'  # The column name in finalfeature.csv for the original price
# This is used to calculate actual returns for the strategy.

# CRITICAL: This MUST be consistent with CONFIG_FUTURE_PERIOD in app.py and ml_pipeline.py
# It defines the horizon the loaded model was trained to predict.
# The 'future_price' column in the backtest results will be based on this.
CONFIG_FUTURE_PERIOD = 2 

# Backtesting specific parameters
WALK_FORWARD_STEP = 10      # Step size for walk-forward validation re-training
ROLLING_WINDOW_SIZE = 500   # Window size for rolling backtest training data
MIN_SAMPLES_ROLLING_BACKTEST_TRAIN = 100 # Min samples for each training window in rolling backtest
MIN_SAMPLES_WALK_FORWARD_TRAIN = 200 # Min samples for initial training in walk-forward

# Function to determine initial training window size for walk-forward
def get_min_initial_train_window_size(data_len, min_abs_size):
    # Use a significant portion of data or a minimum absolute size
    return max(min_abs_size, int(data_len * 0.5)) # e.g., 50% of data or min_abs_size

MIN_DATA_LEN_FOR_PROCESSING = 100 # Minimum data length after initial processing in this script

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning) 
warnings.filterwarnings('ignore', category=FutureWarning) 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True) # Ensure output path for plots exists

def load_data_and_model(data_path, model_dir, model_name):
    """Loads preprocessed data and the trained model."""
    logger.info(f"Loading preprocessed data from: {data_path}")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    try:
        data = pd.read_csv(data_path)
        # Attempt to parse Datetime if it exists, for plotting primarily
        if 'Datetime' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
            # data.set_index('Datetime', inplace=True, drop=False) # Keep Datetime as col
        logger.info(f"Data loaded. Shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise
    
    model_path = os.path.join(model_dir, model_name)
    logger.info(f"Loading trained model from: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}. Run ml_pipeline.py first.")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        trained_model = joblib.load(model_path)
        logger.info(f"Model loaded: {type(trained_model)}")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise
        
    return data, trained_model

def prepare_backtest_features_targets(data, target_col_name_in_data, price_col, 
                                      future_period_for_returns, min_data_len):
    """
    Prepares X, y for backtesting from the loaded data.
    Crucially, it re-creates 'future_price' based on `future_period_for_returns`
    for calculating strategy returns. The model's own 'Target' is already in the data.
    """
    logger.info(f"Preparing features and targets for backtest...")
    logger.info(f"  Using target column from data: '{target_col_name_in_data}'")
    logger.info(f"  Using price column for returns: '{price_col}'")
    logger.info(f"  Defining future price for returns based on FUTURE_PERIOD = {future_period_for_returns}")

    if target_col_name_in_data not in data.columns:
        raise ValueError(f"Model's target column '{target_col_name_in_data}' not found in data.")
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' for returns calculation not found.")

    processed_data = data.copy()

    # Create 'future_price' column specifically for backtest return calculation
    # This ensures the returns align with the model's prediction horizon.
    processed_data['future_price_for_returns'] = processed_data[price_col].shift(-future_period_for_returns)
    
    # The model's target ('Target') is already in processed_data from finalfeature.csv
    y = processed_data[target_col_name_in_data]

    # Drop rows where 'future_price_for_returns' is NaN (affects last few rows)
    # Also drop if the original target is NaN (shouldn't happen if app.py is correct)
    processed_data.dropna(subset=['future_price_for_returns', target_col_name_in_data], inplace=True)
    y = y[processed_data.index] # Align y with dropped rows in processed_data
    
    if len(processed_data) < min_data_len:
        raise ValueError(f"Not enough data after processing for backtest. Need {min_data_len}, got {len(processed_data)}")

    # Define features: all columns except target and other specific exclusions
    exclude_cols = [target_col_name_in_data, 'future_price_for_returns', 'Datetime', 'Date', 'Time'] # Add common date/time cols
    feature_cols = [col for col in processed_data.columns if col not in exclude_cols and col != price_col] # Also exclude raw price_col if it's not a feature
    
    # A more robust way to get feature names would be to load them from model_parameters.txt
    # For now, we infer, but this could be risky if columns change.
    # Ideally, ml_pipeline.py saves the feature list, and backtest.py loads it.
    # For simplicity here, we assume all other numeric columns (excluding the ones above and original price) are features.
    
    # Let's try to get feature names from the model if possible
    model_feature_names = None
    if hasattr(trained_model, 'feature_names_in_'):
        model_feature_names = list(trained_model.feature_names_in_)
    elif hasattr(trained_model, 'feature_name_'): # LightGBM
        model_feature_names = list(trained_model.feature_name_)
    
    if model_feature_names:
        logger.info(f"Attempting to use feature names from loaded model: {model_feature_names[:5]}...")
        # Check if all model features are in the current data
        missing_features = [mf for mf in model_feature_names if mf not in processed_data.columns]
        if missing_features:
            logger.error(f"FATAL: Features expected by the model are missing from the data: {missing_features}")
            logger.error(f"Available data columns: {processed_data.columns.tolist()}")
            raise ValueError("Mismatch between model's expected features and available data features.")
        feature_cols = model_feature_names # Use features the model was trained on
    else:
        logger.warning("Could not get feature names from the loaded model. Inferring features. This might be risky.")
        # Fallback to inferring (as above), but filter to only numeric types
        feature_cols = [f_col for f_col in feature_cols if pd.api.types.is_numeric_dtype(processed_data[f_col])]


    X = processed_data[feature_cols].copy()

    # Ensure no NaN values in final X (should be handled by app.py, but double check)
    if X.isnull().any().any():
        logger.warning("NaN values found in X for backtesting. Filling with 0. Review app.py.")
        X.fillna(0, inplace=True)
    
    logger.info(f"Features and targets for backtest prepared. X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Using {len(feature_cols)} features for backtesting: {feature_cols[:5]}...")
    
    return processed_data, X, y


def get_model_feature_importances(model, features_list):
    """Extract feature importances if available."""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(features_list, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        return dict(zip(features_list, np.abs(coef)))
    return {}

def walk_forward_validation(model_template, X_full, y_full, 
                            initial_train_window_size, step_size, features_list_for_model):
    logger.info(f"Starting Walk-Forward Validation: initial_window={initial_train_window_size}, step={step_size}")
    predictions_all, actuals_all = [], []
    model_versions_info = []

    if len(X_full) < initial_train_window_size + step_size:
        raise ValueError(f"Data too short for walk-forward. Need {initial_train_window_size + step_size}, have {len(X_full)}.")
        
    num_iterations = (len(X_full) - initial_train_window_size) // step_size + 1
    
    with tqdm(total=num_iterations, desc="Walk-Forward Validation") as pbar:
        for i in range(initial_train_window_size, len(X_full), step_size):
            train_end_idx = i
            test_start_idx = i
            test_end_idx = min(i + step_size, len(X_full))

            if test_start_idx >= test_end_idx: break

            X_train, y_train = X_full.iloc[:train_end_idx], y_full.iloc[:train_end_idx]
            X_test, y_test = X_full.iloc[test_start_idx:test_end_idx], y_full.iloc[test_start_idx:test_end_idx]

            if X_test.empty: continue

            current_model_instance = clone(model_template)
            try:
                current_model_instance.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"WF Training failed at train_end_idx {train_end_idx}: {e}")
                predictions_all.extend([np.nan] * len(y_test))
                actuals_all.extend(y_test.values)
                pbar.update(1)
                continue
            
            # Store info about this model version (optional)
            model_versions_info.append({
                'train_end_datetime': X_train.index[-1] if isinstance(X_train.index, pd.DatetimeIndex) else train_end_idx-1,
                'feature_importances': get_model_feature_importances(current_model_instance, features_list_for_model)
            })

            preds = current_model_instance.predict(X_test)
            predictions_all.extend(preds)
            actuals_all.extend(y_test.values)
            pbar.update(1)

    if not predictions_all: raise ValueError("No predictions from walk-forward validation.")
    
    valid_indices = [j for j, p in enumerate(predictions_all) if not pd.isna(p)]
    if not valid_indices: return {}, []

    actuals_f = np.array(actuals_all)[valid_indices]
    predictions_f = np.array(predictions_all)[valid_indices]

    metrics = {
        'accuracy': accuracy_score(actuals_f, predictions_f),
        'precision': precision_score(actuals_f, predictions_f, zero_division=0, average='weighted'),
        'recall': recall_score(actuals_f, predictions_f, zero_division=0, average='weighted'),
        'f1': f1_score(actuals_f, predictions_f, zero_division=0, average='weighted')
    }
    logger.info(f"Walk-forward validation completed. Metrics: {metrics}")
    return metrics, model_versions_info


def rolling_backtest(model_template, data_with_prices, X_features, y_target, 
                     window_size, step_size, min_train_samples, 
                     price_col_for_returns, future_price_col_for_returns):
    logger.info(f"Starting Rolling Backtest: window_size={window_size}, step={step_size}")
    results_list = []
    
    num_iterations = (len(X_features) - window_size) // step_size + 1
    if num_iterations <= 0 :
        raise ValueError(f"Data too short for rolling backtest. Needs > {window_size} samples, got {len(X_features)}")

    with tqdm(total=num_iterations, desc="Rolling Backtest") as pbar:
        for i in range(0, len(X_features) - window_size + 1, step_size): # Ensure +1 for full range
            train_start_idx = i
            train_end_idx = i + window_size
            test_start_idx = train_end_idx
            test_end_idx = min(train_end_idx + step_size, len(X_features))

            if test_start_idx >= test_end_idx: break

            X_train, y_train = X_features.iloc[train_start_idx:train_end_idx], y_target.iloc[train_start_idx:train_end_idx]
            X_test, y_test = X_features.iloc[test_start_idx:test_end_idx], y_target.iloc[test_start_idx:test_end_idx]
            
            data_test_period = data_with_prices.iloc[test_start_idx:test_end_idx]

            if len(X_train) < min_train_samples or X_test.empty:
                pbar.update(1)
                continue
            
            current_model_instance = clone(model_template)
            try:
                current_model_instance.fit(X_train, y_train)
                preds = current_model_instance.predict(X_test)
                probas = current_model_instance.predict_proba(X_test)[:, 1] # Prob of class 1
                
                for j, (pred, actual_target_val, prob) in enumerate(zip(preds, y_test.values, probas)):
                    original_idx_timestamp = X_test.index[j] # This is RangeIndex if no datetime index from CSV
                    # Try to get Datetime from data_with_prices if available
                    timestamp_val = data_test_period['Datetime'].iloc[j] if 'Datetime' in data_test_period.columns else original_idx_timestamp
                    
                    results_list.append({
                        'timestamp': timestamp_val,
                        'prediction': pred, # Model's prediction (0 or 1)
                        'actual_target': actual_target_val, # Actual target (0 or 1)
                        'probability_class1': prob,
                        'current_price_at_prediction': data_test_period.iloc[j][price_col_for_returns],
                        'future_price_actual': data_test_period.iloc[j][future_price_col_for_returns],
                    })
            except Exception as e:
                logger.warning(f"Rolling backtest iter failed at window {train_start_idx}-{train_end_idx}: {e}")
            pbar.update(1)
    
    if not results_list: return pd.DataFrame()
    results_df = pd.DataFrame(results_list)
    logger.info(f"Rolling backtest completed. {len(results_df)} predictions generated.")
    return results_df

def analyze_backtest_results(results_df, output_dir):
    if results_df.empty:
        logger.warning("Backtest results DataFrame is empty. Skipping analysis.")
        return

    logger.info("Analyzing backtest results...")
    results_df['actual_percentage_return'] = (results_df['future_price_actual'] / results_df['current_price_at_prediction']) - 1
    results_df['strategy_return'] = np.where(
        results_df['prediction'] == 1,
        results_df['actual_percentage_return'],
        -results_df['actual_percentage_return'] 
    )
    
    total_trades = len(results_df)
    winning_trades = results_df[results_df['strategy_return'] > 0].shape[0]
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    sum_profit = results_df[results_df['strategy_return'] > 0]['strategy_return'].sum()
    sum_loss = abs(results_df[results_df['strategy_return'] < 0]['strategy_return'].sum())
    profit_factor = sum_profit / sum_loss if sum_loss > 0 else np.inf
    total_return_percentage = results_df['strategy_return'].sum() * 100

    logger.info("\n--- Strategy Performance Summary ---")
    logger.info(f"Total Trades: {total_trades}, Win Rate: {win_rate:.2f}%")
    logger.info(f"Total Strategy Return: {total_return_percentage:.2f}%, Profit Factor: {profit_factor:.2f}")

    # Plot Equity Curve
    plot_index = pd.to_datetime(results_df['timestamp'], errors='coerce') if 'timestamp' in results_df.columns else results_df.index
    if pd.api.types.is_datetime64_any_dtype(plot_index):
         plot_index_label = 'Time'
    else: # If timestamp is just trade number or could not be parsed
        plot_index = results_df.index 
        plot_index_label = 'Trade Number'

    cumulative_returns = (1 + results_df['strategy_return']).cumprod() - 1
    
    plt.figure(figsize=(14, 7))
    plt.plot(plot_index, cumulative_returns * 100)
    plt.title('Strategy Equity Curve (Cumulative Returns %)')
    plt.xlabel(plot_index_label); plt.ylabel('Cumulative Return (%)'); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'equity_curve.png')); plt.close()
    logger.info(f"Equity curve saved to {os.path.join(output_dir, 'equity_curve.png')}")

    # Classification Report
    logger.info("\n--- Classification Report (Predicting Target Up/Down) ---")
    report = classification_report(results_df['actual_target'].astype(int), results_df['prediction'].astype(int), zero_division=0)
    logger.info(f"\n{report}")
    cm = confusion_matrix(results_df['actual_target'].astype(int), results_df['prediction'].astype(int))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Down', 'Pred Up'], yticklabels=['Actual Down', 'Actual Up'])
    plt.title('Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png')); plt.close()
    logger.info(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

def save_backtest_results_csv(results_df, output_dir):
    if results_df.empty: return
    version = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"backtest_results_{version}.csv")
    results_df.to_csv(path, index=False)
    logger.info(f"Saved backtest results to {path}")

if __name__ == "__main__":
    try:
        # --- 1. Load Data and Trained Model ---
        # The model loaded here is the one trained and saved by ml_pipeline.py
        full_data, trained_model = load_data_and_model(
            PROCESSED_DATA_PATH, MODEL_OUTPUT_PATH, LOAD_MODEL_NAME
        )
        
        # --- 2. Prepare Data for Backtesting ---
        # This re-aligns features and target, and creates 'future_price_for_returns'
        # using the CONFIG_FUTURE_PERIOD.
        backtest_data_df, X_backtest, y_backtest = prepare_backtest_features_targets(
            full_data, 
            target_col_name_in_data=TARGET_COL_NAME_IN_DATA, # 'Target'
            price_col=TARGET_PRICE_COL, # 'Close'
            future_period_for_returns=CONFIG_FUTURE_PERIOD, # e.g., 2
            min_data_len=MIN_DATA_LEN_FOR_PROCESSING
        )
        
        # --- 3. Walk-Forward Validation (using the loaded model template) ---
        # This assesses model stability by retraining on expanding windows.
        initial_wf_train_window = get_min_initial_train_window_size(len(X_backtest), MIN_SAMPLES_WALK_FORWARD_TRAIN)
        
        if len(X_backtest) > initial_wf_train_window + WALK_FORWARD_STEP:
            logger.info("Performing Walk-Forward Validation with the loaded model configuration...")
            wf_metrics, _ = walk_forward_validation(
                trained_model, # Pass the loaded model as a template
                X_backtest, y_backtest, 
                initial_train_window_size=initial_wf_train_window, 
                step_size=WALK_FORWARD_STEP,
                features_list_for_model=X_backtest.columns.tolist() # Pass feature names
            )
            logger.info(f"Walk-Forward Validation Metrics (based on model's target prediction):\n{pd.Series(wf_metrics)}")
        else:
            logger.warning("Skipping Walk-Forward Validation due to insufficient data length.")

        # --- 4. Rolling Backtest (using the loaded model template) ---
        # This simulates how the strategy would have performed by retraining on a rolling window.
        logger.info(f"Performing Rolling Backtest with loaded model type: {type(trained_model)}...")
        
        if len(X_backtest) > ROLLING_WINDOW_SIZE + WALK_FORWARD_STEP : # Ensure enough data
            # 'backtest_data_df' contains 'future_price_for_returns' and 'TARGET_PRICE_COL'
            backtest_results_df = rolling_backtest(
                trained_model,      # Pass the loaded model as a template
                backtest_data_df,   # Full data with prices for return calculation
                X_backtest, y_backtest,
                window_size=ROLLING_WINDOW_SIZE,
                step_size=WALK_FORWARD_STEP, 
                min_train_samples=MIN_SAMPLES_ROLLING_BACKTEST_TRAIN,
                price_col_for_returns=TARGET_PRICE_COL, # e.g. 'Close'
                future_price_col_for_returns='future_price_for_returns' # Created in prepare_backtest_features_targets
            )
            
            save_backtest_results_csv(backtest_results_df, MODEL_OUTPUT_PATH)
            
            # --- 5. Analyze and Report Backtest Results ---
            if not backtest_results_df.empty:
                analyze_backtest_results(backtest_results_df, MODEL_OUTPUT_PATH)
            else:
                logger.info("Rolling backtest did not produce results. Skipping analysis.")
        else:
            logger.warning("Skipping Rolling Backtest due to insufficient data length.")

        logger.info("\n--- Key Configuration for this Backtest Run ---")
        logger.info(f"Data Source: {PROCESSED_DATA_PATH}")
        logger.info(f"Loaded Model: {LOAD_MODEL_NAME} (Type: {type(trained_model)})")
        logger.info(f"Model's Prediction Horizon (FUTURE_PERIOD): {CONFIG_FUTURE_PERIOD} periods")
        logger.info(f"Price column for returns: '{TARGET_PRICE_COL}'")
        logger.info(f"Walk-Forward Step: {WALK_FORWARD_STEP}, Rolling Window Size: {ROLLING_WINDOW_SIZE}")
        logger.info("-------------------------------------------------")
        logger.info("Backtesting process completed successfully.")
        
    except FileNotFoundError as fnf_err:
        logger.error(f"Critical Error: A required file was not found. {str(fnf_err)}")
    except ValueError as val_err:
        logger.error(f"Critical Error: Data validation or processing issue. {str(val_err)}")
    except Exception as e:
        logger.error(f"Main backtesting process failed: {str(e)}", exc_info=True)

