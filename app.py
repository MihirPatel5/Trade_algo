import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import ta
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib # For saving the scaler

# ======= CONFIGURATION =======
RAW_OUTPUT_FOLDER = "forex_data"
PROCESSED_OUTPUT_FOLDER = "forex_preprocessed"

# CRITICAL: This MUST be consistent across app.py, ml_pipeline.py, and backtest.py
# Defines how many periods into the future we are trying to predict the price movement for.
CONFIG_FUTURE_PERIOD = 2 
CONFIG_TARGET_COL_NAME = 'Target' # Standardized target column name
FINAL_FEATURE_FILENAME = 'finalfeature.csv' # Standardized name for the output data file

# Ensure output directories exist
os.makedirs(RAW_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_OUTPUT_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======= DATA FETCHING FUNCTIONS =======
def fetch_yahoo_forex(pair_symbol='EURUSD=X', interval='1h', period='3y'): # Default to 1h, 3 years
    """Fetches historical forex data from Yahoo Finance."""
    logger.info(f"Fetching Yahoo Finance data for {pair_symbol} (Interval: {interval}, Period: {period})...")
    data = yf.download(tickers=pair_symbol, interval=interval, period=period, progress=False, threads=False)
    if data.empty:
        logger.warning(f"No data returned from Yahoo Finance for {pair_symbol}.")
        return pd.DataFrame()
    data.reset_index(inplace=True)
    
    # Standardize 'Datetime' column name
    if 'Datetime' not in data.columns:
        if 'Date' in data.columns: # Common yfinance name
            data.rename(columns={'Date': 'Datetime'}, inplace=True)
        elif pd.api.types.is_datetime64_any_dtype(data.columns[0]): # If first col is datetime like
            data.rename(columns={data.columns[0]: 'Datetime'}, inplace=True)
        else:
            logger.error("Could not find a suitable Datetime column in Yahoo Finance data.")
            return pd.DataFrame()

    filename = os.path.join(RAW_OUTPUT_FOLDER, f'yahoo_{pair_symbol.replace("=X","")}_{interval}.csv')
    data.to_csv(filename, index=False)
    logger.info(f"✅ Yahoo data saved: {filename} (Shape: {data.shape})\n")
    return data

def fetch_binance_forex(pair_symbol='EURUSDT', interval='1h', limit=1000):
    """Fetches historical forex data from Binance."""
    logger.info(f"Fetching Binance data for {pair_symbol} (Interval: {interval}, Limit: {limit})...")
    valid_binance_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    if interval not in valid_binance_intervals:
        logger.error(f"Invalid interval '{interval}' for Binance. Valid intervals: {valid_binance_intervals}")
        return pd.DataFrame()

    url = f"https://api.binance.com/api/v3/klines?symbol={pair_symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
        data_json = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Binance API request error: {e}")
        return pd.DataFrame()

    if isinstance(data_json, dict) and 'code' in data_json: # Binance error format
        logger.error(f"❌ Binance API error: {data_json.get('msg', 'Unknown error')}")
        return pd.DataFrame()
    if not isinstance(data_json, list) or not data_json:
        logger.warning(f"No kline data returned from Binance for {pair_symbol}.")
        return pd.DataFrame()

    # Process Binance data
    df = pd.DataFrame(data_json, columns=[
        'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades',
        'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
    ])
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']] # Select relevant columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    filename = os.path.join(RAW_OUTPUT_FOLDER, f'binance_{pair_symbol}_{interval}.csv')
    df.to_csv(filename, index=False)
    logger.info(f"✅ Binance data saved: {filename} (Shape: {df.shape})\n")
    return df

def load_metatrader_csv(mt_csv_file_path):
    """Loads and standardizes data from a MetaTrader CSV export."""
    logger.info(f"Attempting to load MetaTrader CSV: {mt_csv_file_path}")
    if not os.path.exists(mt_csv_file_path):
        logger.error(f"❌ MetaTrader CSV file not found: {mt_csv_file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(mt_csv_file_path) 
        
        # Datetime parsing (adapt to your MT4/MT5 export format)
        if 'Date' in df.columns and 'Time' in df.columns:
            df['Datetime_str'] = df['Date'] + ' ' + df['Time']
            df['Datetime'] = pd.to_datetime(df['Datetime_str'], errors='coerce')
            logger.info("Combined 'Date' and 'Time' columns into 'Datetime'.")
        elif 'Time' in df.columns and 'Open' in df.columns: # If 'Time' column is already a full datetime
             df['Datetime'] = pd.to_datetime(df['Time'], errors='coerce')
        elif 'datetime' in (col.lower() for col in df.columns): # if a column is literally 'datetime' (case-insensitive)
            dt_col_name = [col for col in df.columns if col.lower() == 'datetime'][0]
            df.rename(columns={dt_col_name: 'Datetime'}, inplace=True)
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        else:
            logger.error("❌ Could not find/create 'Datetime' from MetaTrader CSV. Check columns.")
            return pd.DataFrame()

        # Standardize OHLCV column names
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower and 'Open' not in rename_map.values(): rename_map[col] = 'Open'
            elif 'high' in col_lower and 'High' not in rename_map.values(): rename_map[col] = 'High'
            elif 'low' in col_lower and 'Low' not in rename_map.values(): rename_map[col] = 'Low'
            elif 'close' in col_lower and 'Close' not in rename_map.values(): rename_map[col] = 'Close'
            elif ('volume' in col_lower or 'vol.' in col_lower or 'tickvol' in col_lower) and 'Volume' not in rename_map.values(): 
                rename_map[col] = 'Volume'
        df.rename(columns=rename_map, inplace=True)
        
        required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"❌ MetaTrader CSV missing required OHLC columns after renaming. Need: {required_cols}")
            return pd.DataFrame()
            
        std_cols = ['Datetime', 'Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns: std_cols.append('Volume')
        else: df['Volume'] = 0 # Add dummy volume if not present, as some indicators might use it
        
        df = df[std_cols]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['Datetime', 'Close'], inplace=True)
        df = df.sort_values(by='Datetime').reset_index(drop=True)
        
        logger.info(f"✅ MetaTrader CSV loaded and standardized. Shape: {df.shape}")
        base_name = os.path.splitext(os.path.basename(mt_csv_file_path))[0]
        output_raw_path = os.path.join(RAW_OUTPUT_FOLDER, f"mt_standardized_{base_name}.csv")
        df.to_csv(output_raw_path, index=False)
        logger.info(f"✅ Standardized MetaTrader data saved to: {output_raw_path}")
        return df
    except Exception as e:
        logger.error(f"❌ Error processing MetaTrader CSV {mt_csv_file_path}: {e}", exc_info=True)
        return pd.DataFrame()

# ======= DATA PREPROCESSING & FEATURE ENGINEERING =======
def preprocess_forex_data(input_file_path, future_period_for_target, target_col_name):
    """Main function to preprocess data and engineer features."""
    logger.info(f"\n🔄 Preprocessing {input_file_path} with FUTURE_PERIOD={future_period_for_target}...")
    
    if not os.path.exists(input_file_path):
        logger.error(f"❌ Input file not found: {input_file_path}")
        return None

    df = pd.read_csv(input_file_path)
    if df.empty:
        logger.error(f"❌ Error: Input file {input_file_path} is empty.")
        return None
        
    # Datetime Indexing
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df.dropna(subset=['Datetime'], inplace=True) # Drop rows where datetime couldn't be parsed
        df.set_index('Datetime', inplace=True)
    else: # Fallback if 'Datetime' column is missing after loading
        logger.warning("⚠️ No 'Datetime' column for index. Ensure data is sorted chronologically.")

    logger.info(f"Data shape after loading: {df.shape}")

    # Basic Cleaning
    core_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in core_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else: # If a core OHLCV column is missing, create it with NaNs or handle as error
            logger.warning(f"⚠️ Core column '{col}' not found. Creating as NaN or 0.")
            df[col] = 0 if col == 'Volume' else np.nan 
    
    df.dropna(subset=['Close'], inplace=True) # Close is essential
    df.sort_index(inplace=True) 
    df = df[~df.index.duplicated(keep='first')] # Remove duplicate timestamps
    
    logger.info(f"Data shape after initial cleaning & sorting: {df.shape}")
    if df.empty:
        logger.error("❌ Error: No data remains after initial cleaning.")
        return None

    # --- Feature Engineering ---
    # 1. Technical Indicators (using 'ta' library)
    logger.info("Adding technical indicators...")
    try:
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            macd = ta.trend.MACD(df['Close']); df['macd'] = macd.macd(); df['macd_signal'] = macd.macd_signal(); df['macd_diff'] = macd.macd_diff()
            bb = ta.volatility.BollingerBands(df['Close']); df['bb_hband'] = bb.bollinger_hband(); df['bb_lband'] = bb.bollinger_lband(); df['bb_mavg'] = bb.bollinger_mavg(); df['bb_pband'] = bb.bollinger_pband(); df['bb_wband'] = bb.bollinger_wband()
            df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']); df['stoch_k'] = stoch.stoch(); df['stoch_d'] = stoch.stoch_signal()
            adx_i = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']); df['adx'] = adx_i.adx(); df['adx_pos'] = adx_i.adx_pos(); df['adx_neg'] = adx_i.adx_neg()
            df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
            for w in [5, 10, 20, 30, 50, 100, 200]:
                if len(df) >= w:
                    df[f'sma_{w}'] = ta.trend.SMAIndicator(df['Close'], window=w).sma_indicator()
                    df[f'ema_{w}'] = ta.trend.EMAIndicator(df['Close'], window=w).ema_indicator()
                else: df[f'sma_{w}'] = np.nan; df[f'ema_{w}'] = np.nan
            logger.info("✅ Standard Technical Indicators Added.")
        else: logger.warning("⚠️ Skipping some TA indicators due to missing OHLC columns.")
    except Exception as e: logger.error(f"❌ Error adding TA: {e}", exc_info=True)

    # 2. Price Transformation Features
    logger.info("Adding price transformation features...")
    df['pct_change_1'] = df['Close'].pct_change(periods=1) * 100 
    df['log_return_1'] = np.log(df['Close'] / df['Close'].shift(1))
    if 'sma_20' in df.columns: df['close_minus_sma20'] = df['Close'] - df['sma_20']
    if 'ema_50' in df.columns: df['close_div_ema50'] = df['Close'] / df['ema_50'] # Ratio
    logger.info("✅ Price Transformation Features Added.")

    # 3. Volatility Features
    logger.info("Adding volatility features...")
    if 'log_return_1' in df.columns:
        for w in [10, 20, 50]:
            if len(df) >= w: df[f'volatility_{w}'] = df['log_return_1'].rolling(window=w).std() * np.sqrt(w) 
            else: df[f'volatility_{w}'] = np.nan
    logger.info("✅ Volatility Features Added.")
    
    # 4. Time-based Features (if index is Datetime)
    if isinstance(df.index, pd.DatetimeIndex):
        logger.info("Adding time-based features...")
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek 
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week.astype(int)
        df['month'] = df.index.month
        # Example: Binary feature for London/NY overlap (approx 12:00-16:00 UTC, depends on DST)
        # df['is_london_ny_overlap'] = ((df['hour'] >= 12) & (df['hour'] <= 16)).astype(int)
        logger.info("✅ Time-based Features Added.")
    else: logger.warning("⚠️ Skipping time-based features: index is not DatetimeIndex.")

    # 5. Lagged Features
    logger.info("Adding lagged features...")
    for lag_val in range(1, 6): 
        df[f'close_lag_{lag_val}'] = df['Close'].shift(lag_val)
        if 'Volume' in df.columns: df[f'volume_lag_{lag_val}'] = df['Volume'].shift(lag_val)
        if 'log_return_1' in df.columns: df[f'log_return_lag_{lag_val}'] = df['log_return_1'].shift(lag_val)
    logger.info("✅ Lagged Features Added.")

    # 6. Example: Market Regime Feature (Simple Volatility-Based)
    # if 'volatility_20' in df.columns and 'sma_100' in df.columns and 'sma_20' in df.columns:
    #     df['high_vol_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(window=50).mean() * 1.2).astype(int)
    #     df['uptrend_regime'] = (df['sma_20'] > df['sma_100']).astype(int)
    #     logger.info("✅ Simple Regime Features Added (Example).")

    # --- Define Target Variable ---
    # Target: 1 if price 'future_period_for_target' candles ahead is higher than current, 0 otherwise.
    df[target_col_name] = np.where(df['Close'].shift(-future_period_for_target) > df['Close'], 1, 0)
    logger.info(f"✅ Target ('{target_col_name}') for {future_period_for_target}-period future added.")
    logger.info(f"Target distribution:\n{df[target_col_name].value_counts(normalize=True, dropna=False)}")

    # Handle NaNs from indicators and lags (critical step)
    df.ffill(inplace=True) # Forward fill first
    df.bfill(inplace=True) # Then backward fill for any remaining at the start
    
    df_before_final_dropna = df.copy() # For debugging if all rows get dropped
    df.dropna(inplace=True) # Drop any rows that still have NaNs (especially the target)
    
    logger.info(f"Data shape after adding all features and dropping NaNs: {df.shape}")
    if df.empty:
        logger.error("❌ Error: No complete rows remain after feature engineering and NaN handling.")
        # Save intermediate for inspection
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        df_before_final_dropna.reset_index().to_csv(os.path.join(PROCESSED_OUTPUT_FOLDER, f'{base_name}_debug_before_dropna.csv'), index=False)
        return None

    # --- Feature Scaling ---
    cols_to_scale = df.select_dtypes(include=np.number).columns.tolist()
    if target_col_name in cols_to_scale: cols_to_scale.remove(target_col_name)
    # Exclude other known binary/categorical features not meant for scaling
    # Example: if 'is_london_ny_overlap' or 'high_vol_regime' were added
    # for cat_col in ['is_london_ny_overlap', 'high_vol_regime', 'hour', 'dayofweek', 'month']:
    #    if cat_col in cols_to_scale: cols_to_scale.remove(cat_col)
    
    if not cols_to_scale:
        logger.warning("⚠️ No numeric columns found for scaling (excluding target).")
    else:
        logger.info(f"Scaling {len(cols_to_scale)} columns...")
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        scaler_filename = os.path.join(PROCESSED_OUTPUT_FOLDER, f'{os.path.splitext(os.path.basename(input_file_path))[0]}_scaler.joblib')
        joblib.dump(scaler, scaler_filename)
        logger.info(f"✅ Scaler saved to {scaler_filename}. Data Normalized.")

    # Save processed data
    final_out_file_path = os.path.join(PROCESSED_OUTPUT_FOLDER, FINAL_FEATURE_FILENAME)
    df.reset_index(inplace=True) # Save Datetime index as a column
    df.to_csv(final_out_file_path, index=False)
    logger.info(f"🎉 Final feature data saved: {final_out_file_path} (Shape: {df.shape})\n")
    
    return df

# ======= MAIN EXECUTION =======
if __name__ == "__main__":
    # --- Step 1: Choose and Load/Fetch Raw Data ---
    # Default to Yahoo Finance data
    raw_data_df = fetch_yahoo_forex(pair_symbol='EURUSD=X', interval='1h', period='3y')
    source_file_path_for_processing = os.path.join(RAW_OUTPUT_FOLDER, 'yahoo_EURUSD_1h.csv')

    # --- OR ---
    # Example: Use data from a MetaTrader CSV export
    # mt4_csv_path = "path_to_your_mt4_data/EURUSD_1H_2020_2023.csv" # Replace with your actual path
    # if os.path.exists(mt4_csv_path):
    #     logger.info(f"MetaTrader CSV found at {mt4_csv_path}, attempting to use it.")
    #     raw_data_df = load_metatrader_csv(mt4_csv_path)
    #     if raw_data_df is not None and not raw_data_df.empty:
    #         # The load_metatrader_csv function saves a standardized version in RAW_OUTPUT_FOLDER
    #         base_name = os.path.splitext(os.path.basename(mt4_csv_path))[0]
    #         source_file_path_for_processing = os.path.join(RAW_OUTPUT_FOLDER, f"mt_standardized_{base_name}.csv")
    #     else:
    #         logger.warning(f"Failed to load MetaTrader CSV. Falling back to Yahoo Finance if previously fetched.")
    # else:
    #     logger.info(f"MetaTrader CSV not found at {mt4_csv_path}. Using Yahoo Finance data.")


    # --- Step 2: Preprocess the Chosen Data ---
    if raw_data_df is not None and not raw_data_df.empty and os.path.exists(source_file_path_for_processing):
        processed_dataframe = preprocess_forex_data(
            source_file_path_for_processing, 
            future_period_for_target=CONFIG_FUTURE_PERIOD, 
            target_col_name=CONFIG_TARGET_COL_NAME
        )
        if processed_dataframe is None:
            logger.error(f"CRITICAL: Preprocessing failed for {source_file_path_for_processing}.")
        else:
            logger.info(f"Preprocessing successful for {source_file_path_for_processing}.")
    else:
        logger.warning(f"Source file {source_file_path_for_processing} not found or raw data is empty. Skipping preprocessing.")

    logger.info("🏁 Data collection and preprocessing script finished!")
