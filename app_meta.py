import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import ta
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib # For saving the scaler
import datetime

# Attempt to import MetaTrader5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    # print("MetaTrader5 library not found. MT5 fetching will not be available.")
    # No need to print here, logger will handle it if MT5 is selected.

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

# Map human-readable intervals to MT5 timeframes
MT5_TIMEFRAME_MAP = {
    '1m': mt5.TIMEFRAME_M1 if MT5_AVAILABLE else None,
    '5m': mt5.TIMEFRAME_M5 if MT5_AVAILABLE else None,
    '15m': mt5.TIMEFRAME_M15 if MT5_AVAILABLE else None,
    '30m': mt5.TIMEFRAME_M30 if MT5_AVAILABLE else None,
    '1h': mt5.TIMEFRAME_H1 if MT5_AVAILABLE else None,
    '4h': mt5.TIMEFRAME_H4 if MT5_AVAILABLE else None,
    '1d': mt5.TIMEFRAME_D1 if MT5_AVAILABLE else None,
    '1w': mt5.TIMEFRAME_W1 if MT5_AVAILABLE else None,
    '1M': mt5.TIMEFRAME_MN1 if MT5_AVAILABLE else None,
}

# ======= DATA FETCHING FUNCTIONS =======
# def fetch_yahoo_forex(pair_symbol='EURUSD=X', interval='1h', period='3y'): # Default to 1h, 3 years
#     """Fetches historical forex data from Yahoo Finance."""
#     logger.info(f"Fetching Yahoo Finance data for {pair_symbol} (Interval: {interval}, Period: {period})...")
#     data = yf.download(tickers=pair_symbol, interval=interval, period=period, progress=False)
#     if data.empty:
#         logger.warning(f"No data returned from Yahoo Finance for {pair_symbol}.")
#         return pd.DataFrame()
#     data.reset_index(inplace=True)

#     # Standardize 'Datetime' column name
#     if 'Datetime' not in data.columns:
#         if 'Date' in data.columns: # Common yfinance name
#             data.rename(columns={'Date': 'Datetime'}, inplace=True)
#         elif 'index' in data.columns and pd.api.types.is_datetime64_any_dtype(data['index']): # Sometimes it's 'index'
#              data.rename(columns={'index': 'Datetime'}, inplace=True)
#         elif pd.api.types.is_datetime64_any_dtype(data.columns[0]): # If first col is datetime like
#             data.rename(columns={data.columns[0]: 'Datetime'}, inplace=True)
#         else:
#             logger.error("Could not find a suitable Datetime column in Yahoo Finance data.")
#             return pd.DataFrame()

#     filename = os.path.join(RAW_OUTPUT_FOLDER, f'yahoo_{pair_symbol.replace("=X","")}_{interval}.csv')
#     data.to_csv(filename, index=False)
#     logger.info(f"✅ Yahoo data saved: {filename} (Shape: {data.shape})\n")
#     return data

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

def fetch_metatrader5_data(mt5_symbol='EURUSD', interval='1h', num_bars=20000): # Increased default bars
    """Fetches historical forex data from MetaTrader 5."""
    if not MT5_AVAILABLE:
        logger.error("❌ MetaTrader5 library is not installed or failed to import. Cannot fetch MT5 data.")
        return pd.DataFrame()

    logger.info(f"Attempting to fetch MetaTrader 5 data for {mt5_symbol} (Interval: {interval}, Bars: {num_bars})...")

    mt5_tf = MT5_TIMEFRAME_MAP.get(interval)
    if mt5_tf is None:
        logger.error(f"❌ Invalid interval '{interval}' for MetaTrader 5. Supported: {list(MT5_TIMEFRAME_MAP.keys())}")
        return pd.DataFrame()

    # Initialize connection to MetaTrader 5 terminal
    if not mt5.initialize():
        logger.error(f"❌ mt5.initialize() failed, error code = {mt5.last_error()}")
        return pd.DataFrame()
    logger.info(f"MT5 initialized: {mt5.terminal_info()}")

    # Check if symbol is available
    symbol_info = mt5.symbol_info(mt5_symbol)
    if symbol_info is None:
        logger.error(f"❌ Symbol {mt5_symbol} not found in MetaTrader 5.")
        mt5.shutdown()
        return pd.DataFrame()
    if not symbol_info.visible:
        logger.info(f"Symbol {mt5_symbol} is not visible, trying to select it...")
        if not mt5.symbol_select(mt5_symbol, True):
            logger.error(f"❌ mt5.symbol_select({mt5_symbol}, True) failed, {mt5_symbol} might not be available from broker.")
            mt5.shutdown()
            return pd.DataFrame()
        logger.info(f"Symbol {mt5_symbol} selected successfully.")


    # Request historical data
    try:
        rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_tf, 0, num_bars)
    except Exception as e:
        logger.error(f"❌ Error fetching rates from MT5: {e}")
        mt5.shutdown()
        return pd.DataFrame()

    mt5.shutdown() # Shutdown connection after fetching
    logger.info("MT5 connection shut down.")

    if rates is None or len(rates) == 0:
        logger.warning(f"No data returned from MetaTrader 5 for {mt5_symbol} with timeframe {interval}.")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['Datetime'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume', # Using tick_volume as standard 'Volume'
        # 'real_volume': 'RealVolume' # You could include this if desired
    }, inplace=True)

    # Select and order standard columns
    std_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[std_cols]

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    filename = os.path.join(RAW_OUTPUT_FOLDER, f'mt5_{mt5_symbol}_{interval}.csv')
    df.to_csv(filename, index=False)
    logger.info(f"✅ MetaTrader 5 data saved: {filename} (Shape: {df.shape})\n")
    return df


def load_metatrader_csv(mt_csv_file_path):
    """Loads and standardizes data from a MetaTrader CSV export."""
    logger.info(f"Attempting to load MetaTrader CSV: {mt_csv_file_path}")
    if not os.path.exists(mt_csv_file_path):
        logger.error(f"❌ MetaTrader CSV file not found: {mt_csv_file_path}")
        return pd.DataFrame()
    try:
        # Try common MT4/MT5 export encodings
        try:
            df = pd.read_csv(mt_csv_file_path)
        except UnicodeDecodeError:
            logger.info("UTF-8 decoding failed, trying 'utf-16-le' for MetaTrader CSV.")
            df = pd.read_csv(mt_csv_file_path, encoding='utf-16-le', sep='\t' if '\t' in open(mt_csv_file_path, encoding='utf-16-le').readline() else ',')


        # Datetime parsing (adapt to your MT4/MT5 export format)
        if 'Date' in df.columns and 'Time' in df.columns: # Standard MT4 style
            df['Datetime_str'] = df['Date'] + ' ' + df['Time']
            df['Datetime'] = pd.to_datetime(df['Datetime_str'], errors='coerce')
            logger.info("Combined 'Date' and 'Time' columns into 'Datetime'.")
        elif '<DATE>' in df.columns and '<TIME>' in df.columns: # Newer MT5 style with <>
            df['Datetime_str'] = df['<DATE>'] + ' ' + df['<TIME>']
            df['Datetime'] = pd.to_datetime(df['Datetime_str'], errors='coerce')
            logger.info("Combined '<DATE>' and '<TIME>' columns into 'Datetime'.")
        elif 'Time' in df.columns and 'Open' in df.columns: # If 'Time' column is already a full datetime (common for some exports)
             df['Datetime'] = pd.to_datetime(df['Time'], errors='coerce')
        elif 'datetime' in (col.lower() for col in df.columns): # if a column is literally 'datetime' (case-insensitive)
            dt_col_name = [col for col in df.columns if col.lower() == 'datetime'][0]
            df.rename(columns={dt_col_name: 'Datetime'}, inplace=True)
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        else: # Attempt to find a single column that can be parsed as datetime
            found_dt_col = False
            for col in df.columns:
                try:
                    # Test if parsing a sample works
                    pd.to_datetime(df[col].dropna().iloc[:5], errors='raise')
                    df['Datetime'] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"Interpreted column '{col}' as 'Datetime'.")
                    if col != 'Datetime': # Rename if it wasn't already 'Datetime'
                        df.drop(columns=[col], inplace=True) # Avoid duplicate if original was 'Datetime' but different case
                    found_dt_col = True
                    break
                except (ValueError, TypeError, AttributeError):
                    continue
            if not found_dt_col:
                logger.error("❌ Could not find/create 'Datetime' from MetaTrader CSV. Check columns.")
                return pd.DataFrame()

        # Standardize OHLCV column names
        rename_map = {}
        for col_original in df.columns:
            col_lower = col_original.lower().replace('<','').replace('>','') # Remove MT5 tags for matching
            if 'open' in col_lower and 'Open' not in rename_map.values(): rename_map[col_original] = 'Open'
            elif 'high' in col_lower and 'High' not in rename_map.values(): rename_map[col_original] = 'High'
            elif 'low' in col_lower and 'Low' not in rename_map.values(): rename_map[col_original] = 'Low'
            elif 'close' in col_lower and 'Close' not in rename_map.values(): rename_map[col_original] = 'Close'
            elif ('volume' in col_lower or 'vol.' in col_lower or 'tickvol' in col_lower or 'vol' == col_lower) and 'Volume' not in rename_map.values():
                rename_map[col_original] = 'Volume'
        df.rename(columns=rename_map, inplace=True)

        required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close']
        missing_req = [col for col in required_cols if col not in df.columns]
        if missing_req:
            logger.error(f"❌ MetaTrader CSV missing required OHLC columns after renaming. Need: {required_cols}. Missing: {missing_req}")
            return pd.DataFrame()

        std_cols = ['Datetime', 'Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns:
            std_cols.append('Volume')
        else:
            df['Volume'] = 0 # Add dummy volume if not present
            logger.info("Added 'Volume' column with zeros as it was missing.")
            std_cols.append('Volume')

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
        # Ensure OHLC are available and numeric before TA
        ohlc_cols_present = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close'])
        if ohlc_cols_present and all(pd.api.types.is_numeric_dtype(df[col]) for col in ['Open', 'High', 'Low', 'Close']):
            df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            macd = ta.trend.MACD(df['Close']); df['macd'] = macd.macd(); df['macd_signal'] = macd.macd_signal(); df['macd_diff'] = macd.macd_diff()
            bb = ta.volatility.BollingerBands(df['Close']); df['bb_hband'] = bb.bollinger_hband(); df['bb_lband'] = bb.bollinger_lband(); df['bb_mavg'] = bb.bollinger_mavg(); #df['bb_pband'] = bb.bollinger_pband(); df['bb_wband'] = bb.bollinger_wband() # pband/wband can produce NaNs if mavg is 0
            df['bb_pband'] = bb.bollinger_pband().fillna(0) # Handle cases where mavg might be zero
            df['bb_wband'] = bb.bollinger_wband().fillna(0)

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
        else: logger.warning("⚠️ Skipping some TA indicators due to missing or non-numeric OHLC columns.")
    except Exception as e: logger.error(f"❌ Error adding TA: {e}", exc_info=True)

    # 2. Price Transformation Features
    logger.info("Adding price transformation features...")
    df['pct_change_1'] = df['Close'].pct_change(periods=1).fillna(0) * 100
    df['log_return_1'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    if 'sma_20' in df.columns and df['sma_20'].isnull().sum() < len(df): df['close_minus_sma20'] = df['Close'] - df['sma_20']
    else: df['close_minus_sma20'] = 0 # Default if SMA20 not available
    if 'ema_50' in df.columns and df['ema_50'].isnull().sum() < len(df) and not (df['ema_50'] == 0).any():
        df['close_div_ema50'] = df['Close'] / df['ema_50'] # Ratio
    else: df['close_div_ema50'] = 1 # Default if EMA50 not available or is zero
    logger.info("✅ Price Transformation Features Added.")

    # 3. Volatility Features
    logger.info("Adding volatility features...")
    if 'log_return_1' in df.columns:
        for w in [10, 20, 50]:
            if len(df) >= w: df[f'volatility_{w}'] = df['log_return_1'].rolling(window=w).std(ddof=0).fillna(0) * np.sqrt(w)
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
        logger.info("✅ Time-based Features Added.")
    else: logger.warning("⚠️ Skipping time-based features: index is not DatetimeIndex.")

    # 5. Lagged Features
    logger.info("Adding lagged features...")
    for lag_val in range(1, 6):
        df[f'close_lag_{lag_val}'] = df['Close'].shift(lag_val)
        if 'Volume' in df.columns: df[f'volume_lag_{lag_val}'] = df['Volume'].shift(lag_val)
        if 'log_return_1' in df.columns: df[f'log_return_lag_{lag_val}'] = df['log_return_1'].shift(lag_val)
    logger.info("✅ Lagged Features Added.")

    # --- Define Target Variable ---
    # Target: 1 if price 'future_period_for_target' candles ahead is higher than current, 0 otherwise.
    # Shift 'Close' price into the past to compare with current 'Close'
    # A positive shift (-N) brings future data to the current row for comparison.
    df[target_col_name] = np.where(df['Close'].shift(-future_period_for_target) > df['Close'], 1, 0)
    logger.info(f"✅ Target ('{target_col_name}') for {future_period_for_target}-period future added.")
    logger.info(f"Target distribution before NaN drop:\n{df[target_col_name].value_counts(normalize=True, dropna=False)}")

    # Handle NaNs from indicators and lags (critical step)
    # Option 1: Forward fill then backward fill (can propagate stale data at the beginning)
    # df.ffill(inplace=True)
    # df.bfill(inplace=True)
    # Option 2: Fill with a specific value like 0 or mean (can distort data if many NaNs)
    # numeric_cols = df.select_dtypes(include=np.number).columns
    # for col in numeric_cols:
    #     if col != target_col_name: # Don't fill target NaNs yet
    # df[col].fillna(0, inplace=True) # Or df[col].fillna(df[col].mean(), inplace=True)
    # Option 3: Drop rows with ANY NaNs (simplest, but can lose data, ESP. AT THE END DUE TO TARGET SHIFT)
    # But first, let's identify columns with high NaN percentage
    nan_info = df.isnull().sum()
    logger.info(f"NaN counts before extensive filling/dropping:\n{nan_info[nan_info > 0]}")

    # Fill specific columns known to generate initial NaNs with 0 or a safe value
    # This applies to indicators that need a warmup period.
    # For instance, pct_change, log_return, and TA indicators.
    # We already handled pct_change and log_return with .fillna(0) at creation.
    # For TA indicators and lags, ffill then bfill is often reasonable.
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df_before_final_dropna = df.copy() # For debugging if all rows get dropped
    
    # Crucially, drop rows where the target is NaN (these are the last few rows due to shifting)
    # Also drop any other rows that might still have NaNs, though ffill/bfill should minimize this.
    df.dropna(inplace=True)

    logger.info(f"Data shape after adding all features and dropping NaNs: {df.shape}")
    if df.empty:
        logger.error("❌ Error: No complete rows remain after feature engineering and NaN handling.")
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        df_before_final_dropna.reset_index().to_csv(os.path.join(PROCESSED_OUTPUT_FOLDER, f'{base_name}_debug_before_dropna.csv'), index=False)
        logger.info(f"Debug file saved to: {os.path.join(PROCESSED_OUTPUT_FOLDER, f'{base_name}_debug_before_dropna.csv')}")
        return None
    
    logger.info(f"Target distribution after NaN drop:\n{df[target_col_name].value_counts(normalize=True, dropna=False)}")


    # --- Feature Scaling ---
    cols_to_scale = df.select_dtypes(include=np.number).columns.tolist()
    if target_col_name in cols_to_scale: cols_to_scale.remove(target_col_name)

    # Exclude other known binary/categorical features not meant for scaling
    # These are already 0-1 or represent discrete categories
    time_based_features = ['hour', 'dayofweek', 'dayofmonth', 'weekofyear', 'month']
    for cat_col in time_based_features: # Add any other custom categorical/binary cols here
       if cat_col in cols_to_scale: cols_to_scale.remove(cat_col)

    if not cols_to_scale:
        logger.warning("⚠️ No numeric columns identified for scaling (excluding target and known categoricals).")
    else:
        logger.info(f"Scaling {len(cols_to_scale)} columns: {cols_to_scale}")
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        # Construct scaler filename carefully
        input_base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        scaler_filename = os.path.join(PROCESSED_OUTPUT_FOLDER, f'{input_base_name}_scaler.joblib')
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
    # --- Step 1: Choose Data Source ---
    # Options: "YAHOO", "BINANCE", "MT_CSV", "MT5_LIVE"
    DATA_SOURCE = "MT_CSV" # <<<<<<< CHANGE THIS TO SELECT YOUR DATA SOURCE

    raw_data_df = None
    source_file_path_for_processing = None # This will be the path to the CSV in RAW_OUTPUT_FOLDER

    if DATA_SOURCE == "YAHOO":
        # pair_symbol usually ends with "=X" for Yahoo forex
        # raw_data_df = fetch_yahoo_forex(pair_symbol='EURUSD=X', interval='1h', period='3y')
        if raw_data_df is not None and not raw_data_df.empty:
            source_file_path_for_processing = os.path.join(RAW_OUTPUT_FOLDER, 'yahoo_EURUSD_1h.csv')

    elif DATA_SOURCE == "BINANCE":
        # pair_symbol usually ends with "USDT" or similar for Binance
        raw_data_df = fetch_binance_forex(pair_symbol='EURUSDT', interval='1h', limit=2000) # 2000 for ~3 months of hourly
        if raw_data_df is not None and not raw_data_df.empty:
            source_file_path_for_processing = os.path.join(RAW_OUTPUT_FOLDER, 'binance_EURUSDT_1h.csv')

    elif DATA_SOURCE == "MT_CSV":
        # Replace with the actual path to YOUR MetaTrader CSV export
        mt_csv_path = "path_to_your_mt_data/EURUSD_H1_2020_2023.csv" # <--- IMPORTANT: SET YOUR PATH
        # mt_csv_path = r"C:\Users\YourUser\AppData\Roaming\MetaQuotes\Terminal\YourBrokerID\MQL5\Files\EURUSD_H1.csv" # Example typical MT5 export path

        if os.path.exists(mt_csv_path):
            logger.info(f"MetaTrader CSV found at {mt_csv_path}, attempting to use it.")
            raw_data_df = load_metatrader_csv(mt_csv_path) # This function also saves a standardized CSV
            if raw_data_df is not None and not raw_data_df.empty:
                base_name = os.path.splitext(os.path.basename(mt_csv_path))[0]
                source_file_path_for_processing = os.path.join(RAW_OUTPUT_FOLDER, f"mt_standardized_{base_name}.csv")
        else:
            logger.error(f"MetaTrader CSV file not found at {mt_csv_path}. Please check the path.")
            raw_data_df = pd.DataFrame() # Ensure it's an empty df if file not found

    elif DATA_SOURCE == "MT5_LIVE":
        if not MT5_AVAILABLE:
            logger.error("MT5_LIVE selected, but MetaTrader5 library is not available. Please install it.")
        else:
            # Ensure your MT5 terminal is running and you're logged in!
            # Symbol name must match what your broker uses (e.g., "EURUSD", "EURUSD.m", etc.)
            mt5_pair_symbol = "EURUSD"
            mt5_interval = "1h"
            # num_bars calculation: 3 years * approx 260 trading days/year * 24 hours/day
            # This is a rough estimate, adjust as needed. MT5 might have limits from broker.
            num_bars_for_3_years_hourly = 3 * 260 * 24
            #num_bars_for_3_years_hourly = 50000 # Or a large fixed number, max 100k for some brokers.
            
            raw_data_df = fetch_metatrader5_data(mt5_symbol=mt5_pair_symbol, interval=mt5_interval, num_bars=num_bars_for_3_years_hourly)
            if raw_data_df is not None and not raw_data_df.empty:
                source_file_path_for_processing = os.path.join(RAW_OUTPUT_FOLDER, f'mt5_{mt5_pair_symbol}_{mt5_interval}.csv')
    else:
        logger.error(f"Invalid DATA_SOURCE: '{DATA_SOURCE}'. Please choose from YAHOO, BINANCE, MT_CSV, MT5_LIVE.")
        raw_data_df = pd.DataFrame() # Ensure it's an empty df for invalid source


    # --- Step 2: Preprocess the Chosen Data ---
    if raw_data_df is not None and not raw_data_df.empty:
        if source_file_path_for_processing and os.path.exists(source_file_path_for_processing):
            processed_dataframe = preprocess_forex_data(
                source_file_path_for_processing,
                future_period_for_target=CONFIG_FUTURE_PERIOD,
                target_col_name=CONFIG_TARGET_COL_NAME
            )
            if processed_dataframe is None or processed_dataframe.empty:
                logger.error(f"CRITICAL: Preprocessing failed or returned empty DataFrame for {source_file_path_for_processing}.")
            else:
                logger.info(f"Preprocessing successful for {source_file_path_for_processing}.")
        else:
            logger.error(f"Source file for processing '{source_file_path_for_processing}' not found or not set. Skipping preprocessing.")
    else:
        logger.warning(f"No raw data loaded (Data source: {DATA_SOURCE}). Skipping preprocessing.")

    logger.info("🏁 Data collection and preprocessing script finished!")