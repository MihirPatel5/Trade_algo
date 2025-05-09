    # config.py

import os

# ======= DIRECTORY CONFIGURATION =======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_OUTPUT_FOLDER_NAME = "forex_data"
PROCESSED_OUTPUT_FOLDER_NAME = "forex_preprocessed"
MODEL_OUTPUT_FOLDER_NAME = "forex_models"

RAW_OUTPUT_FOLDER = os.path.join(BASE_DIR, RAW_OUTPUT_FOLDER_NAME)
PROCESSED_OUTPUT_FOLDER = os.path.join(BASE_DIR, PROCESSED_OUTPUT_FOLDER_NAME)
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, MODEL_OUTPUT_FOLDER_NAME)

# ======= DATA SOURCE CONFIGURATION =======
# Default Yahoo Finance settings
DEFAULT_YAHOO_PAIR = 'EURUSD=X'
DEFAULT_YAHOO_INTERVAL = '1h'
DEFAULT_YAHOO_PERIOD = '3y'

# Default Binance settings (example)
DEFAULT_BINANCE_PAIR = 'EURUSDT'
DEFAULT_BINANCE_INTERVAL = '1h'
DEFAULT_BINANCE_LIMIT = 1000

# ======= FEATURE ENGINEERING & TARGET CONFIGURATION =======
# CRITICAL: Must be consistent across scripts
CONFIG_FUTURE_PERIOD = 2  # How many periods into the future to predict
CONFIG_TARGET_COL_NAME = 'Target'  # Standardized target column name
FINAL_FEATURE_FILENAME = 'finalfeature.csv' # Standardized name for the output data file
SCALER_FILENAME_PREFIX = 'scaler' # Prefix for scaler file, will be <prefix>_<source>.joblib
FEATURE_LIST_FILENAME = 'feature_list.json' # Filename to save the list of features used for training

# ======= ML PIPELINE CONFIGURATION =======
SAVED_MODEL_FILENAME = 'best_trading_model.pkl'
SAVED_PARAMS_FILENAME = 'model_parameters.txt'

# Optuna Settings
N_SPLITS_CV_OPTUNA = 5
N_TRIALS_OPTUNA = 50  # Increase for more thorough search (e.g., 50-100)
MIN_SAMPLES_CV_TRAIN_FOLD_OPTUNA = 100

# ======= BACKTESTING CONFIGURATION =======
LOAD_MODEL_NAME = SAVED_MODEL_FILENAME # Model to load for backtesting
TARGET_PRICE_COL = 'Close'  # Column name for original price for returns calculation

WALK_FORWARD_STEP = 10  # Step size for walk-forward/rolling re-training
ROLLING_WINDOW_SIZE = 500  # Window size for rolling backtest training data
MIN_SAMPLES_ROLLING_BACKTEST_TRAIN = 100
MIN_SAMPLES_WALK_FORWARD_TRAIN = 200
MIN_DATA_LEN_FOR_PROCESSING = 100

# Basic Transaction Cost Simulation (as a percentage of trade value)
TRANSACTION_COST_PERCENT = 0.0005  # e.g., 0.05% for spread + commission

# ======= LOGGING CONFIGURATION =======
LOG_LEVEL = "INFO"  # e.g., DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

# ======= MLFLOW CONFIGURATION (Placeholder) =======
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db" # Example: local SQLite DB
MLFLOW_EXPERIMENT_NAME = "Forex_Trading_Strategy"
ENABLE_MLFLOW = True # Set to False to disable MLflow logging