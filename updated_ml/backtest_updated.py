# backtest_updated.py

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.base import clone
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
import warnings
import json # For loading feature list

# Import configurations
import config

# MLflow (optional, primarily for loading if models are tracked)
# if config.ENABLE_MLFLOW:
#     try:
#         import mlflow
#     except ImportError:
#         config.ENABLE_MLFLOW = False
#         logging.warning("MLflow not installed or import failed. Disabling MLflow usage for loading.")

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Ensure output path for plots exists (though MODEL_OUTPUT_PATH is for models, plots can go here too)
os.makedirs(config.MODEL_OUTPUT_PATH, exist_ok=True)

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
                    format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_data_model_and_features(data_path, model_dir, model_name, feature_list_filename):
    logger.info(f"Loading preprocessed data from: {data_path}")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}"); raise FileNotFoundError(f"Data file not found: {data_path}")
    try:
        data = pd.read_csv(data_path)
        if 'Datetime' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
        logger.info(f"Data loaded. Shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True); raise

    model_path = os.path.join(model_dir, model_name)
    logger.info(f"Loading trained model from: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}. Run ml_pipeline_updated.py first.")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        trained_model = joblib.load(model_path)
        logger.info(f"Model loaded: {type(trained_model)}")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True); raise

    feature_list_path = os.path.join(model_dir, feature_list_filename)
    logger.info(f"Loading feature list from: {feature_list_path}")
    if not os.path.exists(feature_list_path):
        logger.error(f"Feature list file not found: {feature_list_path}. Ensure ml_pipeline_updated.py was run.")
        raise FileNotFoundError(f"Feature list file not found: {feature_list_path}")
    try:
        with open(feature_list_path, 'r') as f:
            model_feature_names = json.load(f)
        logger.info(f"Feature list loaded. {len(model_feature_names)} features.")
    except Exception as e:
        logger.error(f"Error loading feature list: {e}", exc_info=True); raise

    return data, trained_model, model_feature_names


def prepare_backtest_features_targets(data, target_col_name_in_data, model_features,
                                      price_col_for_returns, future_period_for_returns, min_data_len):
    logger.info(f"Preparing features and targets for backtest...")
    logger.info(f"  Using target column from data: '{target_col_name_in_data}'")
    logger.info(f"  Using price column for returns: '{price_col_for_returns}'")
    logger.info(f"  Defining future price for returns based on FUTURE_PERIOD = {future_period_for_returns}")

    if target_col_name_in_data not in data.columns:
        raise ValueError(f"Model's target column '{target_col_name_in_data}' not found in data.")
    if price_col_for_returns not in data.columns:
        raise ValueError(f"Price column '{price_col_for_returns}' for returns calculation not found.")

    processed_data = data.copy()
    processed_data['future_price_for_returns'] = processed_data[price_col_for_returns].shift(-future_period_for_returns)
    y_actual_target = processed_data[target_col_name_in_data] # This is the model's target (0 or 1)

    # Drop rows where 'future_price_for_returns' is NaN or original target is NaN
    processed_data.dropna(subset=['future_price_for_returns', target_col_name_in_data], inplace=True)
    y_actual_target = y_actual_target[processed_data.index]

    if len(processed_data) < min_data_len:
        raise ValueError(f"Not enough data after processing for backtest. Need {min_data_len}, got {len(processed_data)}")

    # Verify all model_features are in processed_data.columns
    missing_features = [mf for mf in model_features if mf not in processed_data.columns]
    if missing_features:
        logger.error(f"FATAL: Features expected by the model are missing from the data: {missing_features}")
        logger.error(f"Available data columns: {processed_data.columns.tolist()}")
        raise ValueError("Mismatch between model's expected features and available data features.")

    X = processed_data[model_features].copy()

    if X.isnull().any().any():
        logger.warning("NaN values found in X for backtesting. Filling with 0. Review app_updated.py.")
        X.fillna(0, inplace=True)

    logger.info(f"Features and targets for backtest prepared. X shape: {X.shape}, y_actual_target shape: {y_actual_target.shape}")
    logger.info(f"Using {len(model_features)} features for backtesting (loaded from list): {model_features[:5]}...")

    return processed_data, X, y_actual_target


def get_model_feature_importances(model, features_list):
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

    if len(X_full) < initial_train_window_size + step_size: # Make sure there's at least one test step
        logger.warning(f"Data too short for meaningful walk-forward. Need {initial_train_window_size + step_size}, have {len(X_full)}. Skipping.")
        return {}, []

    num_iterations = (len(X_full) - initial_train_window_size) // step_size
    if (len(X_full) - initial_train_window_size) % step_size > 0 : num_iterations +=1 # for the last partial step

    with tqdm(total=num_iterations, desc="Walk-Forward Validation") as pbar:
        for i in range(initial_train_window_size, len(X_full), step_size):
            train_end_idx = i
            test_start_idx = i
            test_end_idx = min(i + step_size, len(X_full))

            if test_start_idx >= test_end_idx: break

            X_train, y_train = X_full.iloc[:train_end_idx], y_full.iloc[:train_end_idx]
            X_test, y_test = X_full.iloc[test_start_idx:test_end_idx], y_full.iloc[test_start_idx:test_end_idx]

            if X_test.empty: continue
            # Ensure y_test aligns with X_test after iloc, especially if indices are not continuous
            y_test_aligned = y_full.loc[X_test.index]


            current_model_instance = clone(model_template)
            try:
                current_model_instance.fit(X_train[features_list_for_model], y_train) # Ensure correct features are used
            except Exception as e:
                logger.warning(f"WF Training failed at train_end_idx {train_end_idx}: {e}")
                predictions_all.extend([np.nan] * len(y_test_aligned))
                actuals_all.extend(y_test_aligned.values)
                pbar.update(1)
                continue

            preds = current_model_instance.predict(X_test[features_list_for_model])
            predictions_all.extend(preds)
            actuals_all.extend(y_test_aligned.values)
            pbar.update(1)

    if not predictions_all:
        logger.warning("No predictions from walk-forward validation.")
        return {}, []

    valid_indices = [j for j, p in enumerate(predictions_all) if not pd.isna(p)]
    if not valid_indices:
        logger.warning("No valid predictions from walk-forward validation after NaN check.")
        return {}, []


    actuals_f = np.array(actuals_all)[valid_indices]
    predictions_f = np.array(predictions_all)[valid_indices]

    metrics = {
        'accuracy': accuracy_score(actuals_f, predictions_f),
        'precision': precision_score(actuals_f, predictions_f, zero_division=0, average='weighted'),
        'recall': recall_score(actuals_f, predictions_f, zero_division=0, average='weighted'),
        'f1': f1_score(actuals_f, predictions_f, zero_division=0, average='weighted')
    }
    logger.info(f"Walk-forward validation completed. Metrics (model target prediction): {metrics}")
    return metrics, [] # model_versions_info removed for simplicity here


def rolling_backtest(model_template, data_with_prices, X_features_full, y_target_full, model_feature_list,
                     window_size, step_size, min_train_samples,
                     price_col_for_returns, future_price_col_for_returns, transaction_cost_percent):
    logger.info(f"Starting Rolling Backtest: window_size={window_size}, step={step_size}, transaction_cost={transaction_cost_percent*100:.4f}%")
    results_list = []

    num_iterations = (len(X_features_full) - window_size) // step_size
    if (len(X_features_full) - window_size) % step_size > 0 : num_iterations +=1

    if num_iterations <= 0 :
        logger.warning(f"Data too short for rolling backtest. Needs > {window_size} samples, got {len(X_features_full)}. Skipping.")
        return pd.DataFrame()

    with tqdm(total=num_iterations, desc="Rolling Backtest") as pbar:
        for i in range(0, len(X_features_full) - window_size + 1, step_size):
            train_start_idx = i
            train_end_idx = i + window_size
            test_start_idx = train_end_idx
            test_end_idx = min(train_end_idx + step_size, len(X_features_full))

            if test_start_idx >= test_end_idx: break

            X_train = X_features_full.iloc[train_start_idx:train_end_idx][model_feature_list]
            y_train = y_target_full.iloc[train_start_idx:train_end_idx]
            X_test = X_features_full.iloc[test_start_idx:test_end_idx][model_feature_list]
            y_test_actual_model_target = y_target_full.iloc[test_start_idx:test_end_idx] # Actual 0/1 model target for this period

            data_test_period = data_with_prices.iloc[test_start_idx:test_end_idx] # Original data for price info

            if len(X_train) < min_train_samples or X_test.empty:
                pbar.update(1)
                continue

            current_model_instance = clone(model_template)
            try:
                current_model_instance.fit(X_train, y_train)
                preds_model_target = current_model_instance.predict(X_test) # Predicts 0 or 1
                probas_class1 = current_model_instance.predict_proba(X_test)[:, 1]

                for j, (model_pred, actual_model_target_val, prob) in enumerate(zip(preds_model_target, y_test_actual_model_target.values, probas_class1)):
                    original_data_row = data_test_period.iloc[j]
                    timestamp_val = original_data_row['Datetime'] if 'Datetime' in original_data_row else X_test.index[j]

                    current_price = original_data_row[price_col_for_returns]
                    future_price  = original_data_row[future_price_col_for_returns]

                    # Calculate pnl before transaction costs
                    # Assuming a simple strategy: if model_pred is 1 (Up), we go long. If 0 (Down), we could go short or do nothing.
                    # For simplicity, let's assume: pred=1 -> Long, pred=0 -> Short (symmetric for now)
                    # This needs to be more nuanced based on your actual trading logic.
                    if pd.isna(current_price) or pd.isna(future_price):
                        raw_percentage_return = 0
                    elif current_price == 0: # Avoid division by zero
                        raw_percentage_return = 0
                    else:
                        raw_percentage_return = (future_price / current_price) - 1

                    if model_pred == 1: # Predicted Up (Long position)
                        strategy_raw_return = raw_percentage_return
                    else: # Predicted Down (Short position)
                        strategy_raw_return = -raw_percentage_return

                    # Apply transaction costs (applied on both open and close, so effectively *2 for a round trip if modeled simply this way)
                    # Or, more simply, just reduce the return of each leg.
                    # Let's assume cost is applied to the gross return of the position.
                    final_strategy_return = strategy_raw_return - transaction_cost_percent # For one way, or *2 for round trip

                    results_list.append({
                        'timestamp': timestamp_val,
                        'model_prediction': model_pred, # Model's direct output (0 or 1)
                        'actual_model_target': actual_model_target_val, # The target the model was trained on
                        'probability_class1': prob,
                        'current_price_at_trade': current_price,
                        'future_price_actual': future_price,
                        'strategy_return_with_costs': final_strategy_return
                    })
            except Exception as e:
                logger.warning(f"Rolling backtest iter failed at window {train_start_idx}-{train_end_idx}: {e}", exc_info=False)
            pbar.update(1)

    if not results_list:
        logger.warning("Rolling backtest did not produce any results.")
        return pd.DataFrame()
    results_df = pd.DataFrame(results_list)
    logger.info(f"Rolling backtest completed. {len(results_df)} prediction points generated.")
    return results_df


def analyze_backtest_results(results_df, output_dir):
    if results_df.empty:
        logger.warning("Backtest results DataFrame is empty. Skipping analysis.")
        return

    logger.info("Analyzing backtest results...")
    # 'strategy_return_with_costs' is already calculated

    total_trades = len(results_df)
    # Note: Defining "winning trades" depends on whether strategy_return > 0 (after costs)
    winning_trades = results_df[results_df['strategy_return_with_costs'] > 0].shape[0]
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    sum_profit = results_df[results_df['strategy_return_with_costs'] > 0]['strategy_return_with_costs'].sum()
    sum_loss = abs(results_df[results_df['strategy_return_with_costs'] < 0]['strategy_return_with_costs'].sum())
    profit_factor = sum_profit / sum_loss if sum_loss > 0 else np.inf
    total_return_percentage = results_df['strategy_return_with_costs'].sum() * 100 # Sum of percentage returns

    logger.info("\n--- Strategy Performance Summary (After Costs) ---")
    logger.info(f"Total Trade Points: {total_trades}, Win Rate: {win_rate:.2f}%")
    logger.info(f"Total Strategy Return (sum of %): {total_return_percentage:.2f}%, Profit Factor: {profit_factor:.2f}")
    logger.info(f"Average Return per Trade Point: {results_df['strategy_return_with_costs'].mean()*100:.4f}%")


    plot_index = pd.to_datetime(results_df['timestamp'], errors='coerce')
    if plot_index.isna().all(): # If all timestamps failed to parse
        plot_index = results_df.index
        plot_index_label = 'Trade Number'
    else:
        plot_index_label = 'Time'


    cumulative_returns = (1 + results_df['strategy_return_with_costs']).cumprod() -1 # Starts from 0

    plt.figure(figsize=(14, 7))
    plt.plot(plot_index, cumulative_returns * 100) # Plot as percentage
    plt.title(f'Strategy Equity Curve (Cumulative Returns %, Costs: {config.TRANSACTION_COST_PERCENT*100:.3f}%)')
    plt.xlabel(plot_index_label); plt.ylabel('Cumulative Return (%)'); plt.grid(True)
    eq_curve_path = os.path.join(output_dir, 'equity_curve_with_costs.png')
    plt.savefig(eq_curve_path); plt.close()
    logger.info(f"Equity curve saved to {eq_curve_path}")

    logger.info("\n--- Classification Report (Model's Prediction vs Model's Actual Target) ---")
    # Ensure 'actual_model_target' and 'model_prediction' are integers and handle NaNs if any slipped through
    report_df = results_df[['actual_model_target', 'model_prediction']].dropna().astype(int)
    if not report_df.empty:
        report = classification_report(report_df['actual_model_target'], report_df['model_prediction'], zero_division=0)
        logger.info(f"\n{report}")
        cm = confusion_matrix(report_df['actual_model_target'], report_df['model_prediction'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Down/Side', 'Pred Up'], yticklabels=['Actual Down/Side', 'Actual Up'])
        plt.title('Confusion Matrix (Model Target Prediction)'); plt.ylabel('Actual Model Target'); plt.xlabel('Predicted Model Target')
        cm_path = os.path.join(output_dir, 'confusion_matrix_model_target.png')
        plt.savefig(cm_path); plt.close()
        logger.info(f"Confusion matrix saved to {cm_path}")
    else:
        logger.warning("Not enough data for classification report after NaN drop.")


def save_backtest_results_csv(results_df, output_dir):
    if results_df.empty: return
    version = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"backtest_results_detailed_{version}.csv")
    results_df.to_csv(path, index=False)
    logger.info(f"Saved detailed backtest results to {path}")

if __name__ == "__main__":
    try:
        PROCESSED_DATA_FULL_PATH = os.path.join(config.PROCESSED_OUTPUT_FOLDER, config.FINAL_FEATURE_FILENAME)
        full_data, trained_model, model_feature_names = load_data_model_and_features(
            PROCESSED_DATA_FULL_PATH,
            config.MODEL_OUTPUT_PATH,
            config.LOAD_MODEL_NAME,
            config.FEATURE_LIST_FILENAME
        )

        backtest_data_df, X_backtest, y_backtest_model_target = prepare_backtest_features_targets(
            full_data,
            target_col_name_in_data=config.CONFIG_TARGET_COL_NAME,
            model_features=model_feature_names,
            price_col_for_returns=config.TARGET_PRICE_COL,
            future_period_for_returns=config.CONFIG_FUTURE_PERIOD,
            min_data_len=config.MIN_DATA_LEN_FOR_PROCESSING
        )

        initial_wf_train_window = max(config.MIN_SAMPLES_WALK_FORWARD_TRAIN, int(len(X_backtest) * 0.5))
        if len(X_backtest) > initial_wf_train_window + config.WALK_FORWARD_STEP:
            logger.info("Performing Walk-Forward Validation with the loaded model configuration...")
            wf_metrics, _ = walk_forward_validation(
                trained_model,
                X_backtest, y_backtest_model_target,
                initial_train_window_size=initial_wf_train_window,
                step_size=config.WALK_FORWARD_STEP,
                features_list_for_model=model_feature_names
            )
            if wf_metrics: logger.info(f"Walk-Forward Validation Metrics (model target prediction):\n{pd.Series(wf_metrics)}")
        else:
            logger.warning("Skipping Walk-Forward Validation due to insufficient data length for the configured parameters.")


        logger.info(f"Performing Rolling Backtest with loaded model type: {type(trained_model)}...")
        if len(X_backtest) > config.ROLLING_WINDOW_SIZE + config.WALK_FORWARD_STEP :
            backtest_results_df = rolling_backtest(
                trained_model,
                backtest_data_df, # This df contains the original prices needed for returns
                X_backtest, y_backtest_model_target, # Features and the model's target
                model_feature_list=model_feature_names,
                window_size=config.ROLLING_WINDOW_SIZE,
                step_size=config.WALK_FORWARD_STEP,
                min_train_samples=config.MIN_SAMPLES_ROLLING_BACKTEST_TRAIN,
                price_col_for_returns=config.TARGET_PRICE_COL,
                future_price_col_for_returns='future_price_for_returns',
                transaction_cost_percent=config.TRANSACTION_COST_PERCENT
            )

            save_backtest_results_csv(backtest_results_df, config.MODEL_OUTPUT_PATH)

            if not backtest_results_df.empty:
                analyze_backtest_results(backtest_results_df, config.MODEL_OUTPUT_PATH)
            else:
                logger.info("Rolling backtest did not produce results. Skipping analysis.")
        else:
            logger.warning("Skipping Rolling Backtest due to insufficient data length for the configured parameters.")

        logger.info("\n--- Key Configuration for this Backtest Run ---")
        logger.info(f"Data Source: {PROCESSED_DATA_FULL_PATH}")
        logger.info(f"Loaded Model: {config.LOAD_MODEL_NAME} (Type: {type(trained_model)})")
        logger.info(f"Model's Prediction Horizon (CONFIG_FUTURE_PERIOD): {config.CONFIG_FUTURE_PERIOD} periods")
        logger.info(f"Price column for returns: '{config.TARGET_PRICE_COL}'")
        logger.info(f"Transaction Cost Per Trade Leg: {config.TRANSACTION_COST_PERCENT*100:.4f}%")
        logger.info(f"Walk-Forward Step: {config.WALK_FORWARD_STEP}, Rolling Window Size: {config.ROLLING_WINDOW_SIZE}")
        logger.info("-------------------------------------------------")
        logger.info("Backtesting process completed successfully.")

    except FileNotFoundError as fnf_err:
        logger.error(f"Critical Error: A required file was not found. {str(fnf_err)}", exc_info=True)
    except ValueError as val_err:
        logger.error(f"Critical Error: Data validation or processing issue. {str(val_err)}", exc_info=True)
    except Exception as e:
        logger.error(f"Main backtesting process failed: {str(e)}", exc_info=True)