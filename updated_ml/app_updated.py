# app_updated.py

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import ta
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib

# Import configurations
import config

# Ensure output directories exist
os.makedirs(config.RAW_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(config.PROCESSED_OUTPUT_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
                    format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# ======= DATA FETCHING FUNCTIONS =======
def fetch_yahoo_forex(pair_symbol=config.DEFAULT_YAHOO_PAIR,
                      interval=config.DEFAULT_YAHOO_INTERVAL,
                      period=config.DEFAULT_YAHOO_PERIOD):
    logger.info(f"Fetching Yahoo Finance data for {pair_symbol} (Interval: {interval}, Period: {period})...")
    data = yf.download(tickers=pair_symbol, interval=interval, period=period, progress=False, threads=False)
    if data.empty:
        logger.warning(f"No data returned from Yahoo Finance for {pair_symbol}.")
        return pd.DataFrame(), f'yahoo_{pair_symbol.replace("=X","")}_{interval}'
    data.reset_index(inplace=True)

    if 'Datetime' not in data.columns:
        if 'Date' in data.columns:
            data.rename(columns={'Date': 'Datetime'}, inplace=True)
        elif 'Timestamp' in data.columns: # yfinance can sometimes use Timestamp
            data.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
        elif pd.api.types.is_datetime64_any_dtype(data.columns[0]):
            data.rename(columns={data.columns[0]: 'Datetime'}, inplace=True)
        else:
            logger.error("Could not find a suitable Datetime column in Yahoo Finance data.")
            return pd.DataFrame(), f'yahoo_{pair_symbol.replace("=X","")}_{interval}'

    file_basename = f'yahoo_{pair_symbol.replace("=X","")}_{interval}'
    filename = os.path.join(config.RAW_OUTPUT_FOLDER, f'{file_basename}.csv')
    data.to_csv(filename, index=False)
    logger.info(f"✅ Yahoo data saved: {filename} (Shape: {data.shape})\n")
    return data, file_basename

def fetch_binance_forex(pair_symbol=config.DEFAULT_BINANCE_PAIR,
                        interval=config.DEFAULT_BINANCE_INTERVAL,
                        limit=config.DEFAULT_BINANCE_LIMIT):
    logger.info(f"Fetching Binance data for {pair_symbol} (Interval: {interval}, Limit: {limit})...")
    valid_binance_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    if interval not in valid_binance_intervals:
        logger.error(f"Invalid interval '{interval}' for Binance. Valid intervals: {valid_binance_intervals}")
        return pd.DataFrame(), f'binance_{pair_symbol}_{interval}'

    url = f"https://api.binance.com/api/v3/klines?symbol={pair_symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data_json = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Binance API request error: {e}")
        return pd.DataFrame(), f'binance_{pair_symbol}_{interval}'

    if isinstance(data_json, dict) and 'code' in data_json:
        logger.error(f"❌ Binance API error: {data_json.get('msg', 'Unknown error')}")
        return pd.DataFrame(), f'binance_{pair_symbol}_{interval}'
    if not isinstance(data_json, list) or not data_json:
        logger.warning(f"No kline data returned from Binance for {pair_symbol}.")
        return pd.DataFrame(), f'binance_{pair_symbol}_{interval}'

    df = pd.DataFrame(data_json, columns=[
        'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades',
        'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
    ])
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    file_basename = f'binance_{pair_symbol}_{interval}'
    filename = os.path.join(config.RAW_OUTPUT_FOLDER, f'{file_basename}.csv')
    df.to_csv(filename, index=False)
    logger.info(f"✅ Binance data saved: {filename} (Shape: {df.shape})\n")
    return df, file_basename

def load_metatrader_csv(mt_csv_file_path):
    logger.info(f"Attempting to load MetaTrader CSV: {mt_csv_file_path}")
    if not os.path.exists(mt_csv_file_path):
        logger.error(f"❌ MetaTrader CSV file not found: {mt_csv_file_path}")
        return pd.DataFrame(), None
    try:
        df = pd.read_csv(mt_csv_file_path)
        base_name_orig = os.path.splitext(os.path.basename(mt_csv_file_path))[0]

        if 'Date' in df.columns and 'Time' in df.columns:
            df['Datetime_str'] = df['Date'] + ' ' + df['Time']
            df['Datetime'] = pd.to_datetime(df['Datetime_str'], errors='coerce')
        elif 'Time' in df.columns and 'Open' in df.columns:
             df['Datetime'] = pd.to_datetime(df['Time'], errors='coerce')
        elif 'datetime' in (col.lower() for col in df.columns):
            dt_col_name = [col for col in df.columns if col.lower() == 'datetime'][0]
            df.rename(columns={dt_col_name: 'Datetime'}, inplace=True)
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        else:
            logger.error("❌ Could not find/create 'Datetime' from MetaTrader CSV.")
            return pd.DataFrame(), base_name_orig

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
            logger.error(f"❌ MetaTrader CSV missing required OHLC columns. Need: {required_cols}")
            return pd.DataFrame(), base_name_orig

        std_cols = ['Datetime', 'Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns: std_cols.append('Volume')
        else: df['Volume'] = 0

        df = df[std_cols]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=['Datetime', 'Close'], inplace=True)
        df = df.sort_values(by='Datetime').reset_index(drop=True)

        logger.info(f"✅ MetaTrader CSV loaded and standardized. Shape: {df.shape}")
        file_basename = f"mt_standardized_{base_name_orig}"
        output_raw_path = os.path.join(config.RAW_OUTPUT_FOLDER, f"{file_basename}.csv")
        df.to_csv(output_raw_path, index=False)
        logger.info(f"✅ Standardized MetaTrader data saved to: {output_raw_path}")
        return df, file_basename
    except Exception as e:
        logger.error(f"❌ Error processing MetaTrader CSV {mt_csv_file_path}: {e}", exc_info=True)
        return pd.DataFrame(), None


# ======= DATA PREPROCESSING & FEATURE ENGINEERING =======
def preprocess_forex_data(input_file_path, file_basename_for_scaler, future_period_for_target, target_col_name):
    logger.info(f"\n🔄 Preprocessing {input_file_path} with FUTURE_PERIOD={future_period_for_target}...")

    if not os.path.exists(input_file_path):
        logger.error(f"❌ Input file not found: {input_file_path}")
        return None
    try:
        df = pd.read_csv(input_file_path)
    except Exception as e:
        logger.error(f"❌ Could not read CSV {input_file_path}: {e}")
        return None

    if df.empty:
        logger.error(f"❌ Error: Input file {input_file_path} is empty.")
        return None

    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df.dropna(subset=['Datetime'], inplace=True)
        df.set_index('Datetime', inplace=True)
    else:
        logger.warning("⚠️ No 'Datetime' column for index. Ensure data is sorted chronologically.")

    logger.info(f"Data shape after loading: {df.shape}")

    core_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in core_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.warning(f"⚠️ Core column '{col}' not found. Creating as NaN or 0.")
            df[col] = 0 if col == 'Volume' else np.nan

    df.dropna(subset=['Close'], inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    logger.info(f"Data shape after initial cleaning & sorting: {df.shape}")
    if df.empty:
        logger.error("❌ Error: No data remains after initial cleaning.")
        return None

    # --- Feature Engineering ---
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

    logger.info("Adding price transformation features...")
    df['pct_change_1'] = df['Close'].pct_change(periods=1) * 100
    df['log_return_1'] = np.log(df['Close'] / df['Close'].shift(1))
    if 'sma_20' in df.columns: df['close_minus_sma20'] = df['Close'] - df['sma_20']
    if 'ema_50' in df.columns: df['close_div_ema50'] = df['Close'] / df['ema_50']
    logger.info("✅ Price Transformation Features Added.")

    logger.info("Adding volatility features...")
    if 'log_return_1' in df.columns:
        for w in [10, 20, 50]:
            if len(df) >= w: df[f'volatility_{w}'] = df['log_return_1'].rolling(window=w).std() * np.sqrt(w)
            else: df[f'volatility_{w}'] = np.nan
    logger.info("✅ Volatility Features Added.")

    if isinstance(df.index, pd.DatetimeIndex):
        logger.info("Adding time-based features...")
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week.astype(int)
        df['month'] = df.index.month
        logger.info("✅ Time-based Features Added.")
    else: logger.warning("⚠️ Skipping time-based features: index is not DatetimeIndex.")

    logger.info("Adding lagged features...")
    for lag_val in range(1, 6):
        df[f'close_lag_{lag_val}'] = df['Close'].shift(lag_val)
        if 'Volume' in df.columns: df[f'volume_lag_{lag_val}'] = df['Volume'].shift(lag_val)
        if 'log_return_1' in df.columns: df[f'log_return_lag_{lag_val}'] = df['log_return_1'].shift(lag_val)
    logger.info("✅ Lagged Features Added.")

    df[target_col_name] = np.where(df['Close'].shift(-future_period_for_target) > df['Close'], 1, 0)
    logger.info(f"✅ Target ('{target_col_name}') for {future_period_for_target}-period future added.")
    logger.info(f"Target distribution:\n{df[target_col_name].value_counts(normalize=True, dropna=False)}")

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df_before_final_dropna = df.copy()
    df.dropna(inplace=True)

    logger.info(f"Data shape after adding all features and dropping NaNs: {df.shape}")
    if df.empty:
        logger.error("❌ Error: No complete rows remain after feature engineering and NaN handling.")
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        df_before_final_dropna.reset_index().to_csv(os.path.join(config.PROCESSED_OUTPUT_FOLDER, f'{base_name}_debug_before_dropna.csv'), index=False)
        return None

    cols_to_scale = df.select_dtypes(include=np.number).columns.tolist()
    if target_col_name in cols_to_scale: cols_to_scale.remove(target_col_name)
    
    # Example: Exclude known categorical/binary (already 0/1 or specific range) if needed
    # time_cols = ['hour', 'dayofweek', 'dayofmonth', 'weekofyear', 'month']
    # for t_col in time_cols:
    #     if t_col in cols_to_scale: cols_to_scale.remove(t_col)

    if not cols_to_scale:
        logger.warning("⚠️ No numeric columns found for scaling (excluding target).")
    else:
        logger.info(f"Scaling {len(cols_to_scale)} columns: {cols_to_scale[:5]}...")
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        scaler_filename = os.path.join(config.PROCESSED_OUTPUT_FOLDER, f'{config.SCALER_FILENAME_PREFIX}_{file_basename_for_scaler}.joblib')
        joblib.dump(scaler, scaler_filename)
        logger.info(f"✅ Scaler saved to {scaler_filename}. Data Normalized.")

    final_out_file_path = os.path.join(config.PROCESSED_OUTPUT_FOLDER, config.FINAL_FEATURE_FILENAME)
    df.reset_index(inplace=True)
    df.to_csv(final_out_file_path, index=False)
    logger.info(f"🎉 Final feature data saved: {final_out_file_path} (Shape: {df.shape})\n")

    return df

# ======= MAIN EXECUTION =======
if __name__ == "__main__":
    # --- Step 1: Choose and Load/Fetch Raw Data ---
    # Default to Yahoo Finance data
    raw_data_df, source_basename = fetch_yahoo_forex() # Uses defaults from config
    source_file_path_for_processing = os.path.join(config.RAW_OUTPUT_FOLDER, f'{source_basename}.csv')

    # --- OR ---
    # Example: Use data from a MetaTrader CSV export
    # mt4_csv_path = "path_to_your_mt4_data/EURUSD_1H_2020_2023.csv" # Replace with your actual path
    # if os.path.exists(mt4_csv_path):
    #     logger.info(f"MetaTrader CSV found at {mt4_csv_path}, attempting to use it.")
    #     raw_data_df, source_basename = load_metatrader_csv(mt4_csv_path)
    #     if raw_data_df is not None and not raw_data_df.empty and source_basename:
    #         source_file_path_for_processing = os.path.join(config.RAW_OUTPUT_FOLDER, f"{source_basename}.csv")
    #     else:
    #         logger.warning(f"Failed to load MetaTrader CSV. Falling back to Yahoo Finance if previously fetched/defaulted.")
    # else:
    #     logger.info(f"MetaTrader CSV not found at {mt4_csv_path}. Using default Yahoo Finance data.")


    # --- Step 2: Preprocess the Chosen Data ---
    if raw_data_df is not None and not raw_data_df.empty and source_basename and os.path.exists(source_file_path_for_processing):
        processed_dataframe = preprocess_forex_data(
            source_file_path_for_processing,
            file_basename_for_scaler=source_basename,
            future_period_for_target=config.CONFIG_FUTURE_PERIOD,
            target_col_name=config.CONFIG_TARGET_COL_NAME
        )
        if processed_dataframe is None:
            logger.error(f"CRITICAL: Preprocessing failed for {source_file_path_for_processing}.")
        else:
            logger.info(f"Preprocessing successful for {source_file_path_for_processing}.")
    elif not source_basename:
        logger.error("CRITICAL: source_basename not determined, cannot proceed with preprocessing.")
    else:
        logger.warning(f"Source file {source_file_path_for_processing} not found or raw data is empty. Skipping preprocessing.")

    logger.info("🏁 Data collection and preprocessing script finished!")