import pandas as pd
import numpy as np
import talib as ta
import logging
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PROCESSED_OUTPUT_FOLDER = "forex_preprocessed"
os.makedirs(PROCESSED_OUTPUT_FOLDER, exist_ok=True)

def create_target_column(df, future_period, price_column='Close', target_type='binary'):
    """
    Create target column based on price movement in the future.
    
    Args:
        df: DataFrame with price data
        future_period: Number of periods to look ahead
        price_column: Column name for price data
        target_type: 'binary' (1=up, 0=down) or 'ternary' (1=up, 0=neutral, -1=down)
        
    Returns:
        DataFrame with target column added
    """
    logger.info(f"Creating target column with future_period={future_period}")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate future price and percentage change
    df['future_price'] = df[price_column].shift(-future_period)
    df['future_pct_change'] = (df['future_price'] - df[price_column]) / df[price_column] * 100
    
    if target_type == 'binary':
        # Binary: 1 if price goes up, 0 if price goes down
        df['Target'] = (df['future_price'] > df[price_column]).astype(int)
    
    elif target_type == 'ternary':
        # Define neutral zone threshold (e.g., ±0.05%)
        threshold = 0.05
        
        # Initialize with neutral class (0)
        df['Target'] = 0
        
        # Assign 1 for significant up moves
        df.loc[df['future_pct_change'] > threshold, 'Target'] = 1
        
        # Assign -1 for significant down moves
        df.loc[df['future_pct_change'] < -threshold, 'Target'] = -1
    
    else:
        raise ValueError(f"Invalid target_type: {target_type}. Use 'binary' or 'ternary'.")
    
    logger.info(f"Target column created. Distribution: \n{df['Target'].value_counts(normalize=True)}")
    
    return df

def add_price_features(df):
    """Add basic price-derived features."""
    df = df.copy()
    
    # Simple returns over different periods
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'return_{period}'] = df['Close'].pct_change(period)
    
    # Log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Price differences
    df['price_diff'] = df['Close'].diff()
    df['high_low_diff'] = df['High'] - df['Low']
    df['close_open_diff'] = df['Close'] - df['Open']
    
    # Normalized price by bollinger band width
    df['norm_price'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    
    # Candle features
    df['candle_size'] = df['High'] - df['Low']
    df['candle_body'] = np.abs(df['Close'] - df['Open'])
    df['candle_wick_upper'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['candle_wick_lower'] = np.minimum(df['Close'], df['Open']) - df['Low']
    df['candle_body_ratio'] = df['candle_body'] / df['candle_size'].replace(0, np.nan)
    
    # Volatility metrics
    df['atr_14'] = ta.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    
    logger.info("Added price-based features")
    return df

def add_technical_indicators(df):
    """Add technical indicators from TA-Lib."""
    df = df.copy()
    
    logger.info("Adding technical indicators...")
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = ta.SMA(df['Close'].values, timeperiod=period)
        df[f'ema_{period}'] = ta.EMA(df['Close'].values, timeperiod=period)
    
    # Momentum Indicators
    df['rsi_14'] = ta.RSI(df['Close'].values, timeperiod=14)
    df['rsi_7'] = ta.RSI(df['Close'].values, timeperiod=7)
    df['rsi_21'] = ta.RSI(df['Close'].values, timeperiod=21)
    
    # MACD
    macd, macd_signal, macd_hist = ta.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    
    # Bollinger Bands
    upper, middle, lower = ta.BBANDS(df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / middle
    df['bb_pct'] = (df['Close'] - lower) / (upper - lower)
    
    # Stochastic Oscillator
    slowk, slowd = ta.STOCH(df['High'].values, df['Low'].values, df['Close'].values, 
                           fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['stoch_k'] = slowk
    df['stoch_d'] = slowd
    
    # Average Directional Index
    df['adx'] = ta.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    
    # Commodity Channel Index
    df['cci'] = ta.CCI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    
    # Williams %R
    df['willr'] = ta.WILLR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    
    # Relative strength across multiple periods
    for period in [3, 5, 10, 20]:
        df[f'rs_{period}'] = df['Close'] / df['Close'].shift(period)
    
    # Rate of Change
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = ta.ROC(df['Close'].values, timeperiod=period)
    
    # On-Balance Volume (OBV)
    df['obv'] = ta.OBV(df['Close'].values, df['Volume'].values)
    
    # Parabolic SAR
    df['sar'] = ta.SAR(df['High'].values, df['Low'].values, acceleration=0.02, maximum=0.2)
    
    # Awesome Oscillator (simplified calculation)
    df['ao'] = ta.SMA(df['High'].values + df['Low'].values, timeperiod=5) / 2 - ta.SMA(df['High'].values + df['Low'].values, timeperiod=34) / 2
    
    # Ichimoku Cloud components
    tenkan_sen = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    kijun_sen = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['tenkan_sen'] = tenkan_sen
    df['kijun_sen'] = kijun_sen
    df['senkou_span_a'] = ((tenkan_sen + kijun_sen) / 2).shift(26)
    df['senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['chikou_span'] = df['Close'].shift(-26)
    
    logger.info("Added technical indicators")
    return df

def add_volatility_features(df):
    """Add volatility-based features."""
    df = df.copy()
    
    # Volatility over different periods
    for period in [5, 10, 20, 50]:
        df[f'volatility_{period}'] = df['Close'].pct_change().rolling(period).std()
    
    # True Range
    df['true_range'] = ta.TRANGE(df['High'].values, df['Low'].values, df['Close'].values)
    
    # Normalized True Range (dividing by Close)
    df['norm_true_range'] = df['true_range'] / df['Close']
    
    # Historical volatility (annualized)
    df['hist_vol_14_annualized'] = df['Close'].pct_change().rolling(14).std() * np.sqrt(252)
    
    # Rolling High-Low Range
    for period in [5, 10, 20]:
        rolling_high = df['High'].rolling(period).max()
        rolling_low = df['Low'].rolling(period).min()
        df[f'high_low_range_{period}'] = (rolling_high - rolling_low) / rolling_low * 100
    
    # ATR Percentage
    df['atr_pct'] = df['atr_14'] / df['Close'] * 100
    
    # Bollinger Band Width
    df['bbands_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    logger.info("Added volatility features")
    return df

def add_pattern_recognition(df):
    """Add pattern recognition signals from TA-Lib."""
    df = df.copy()
    
    # Candlestick patterns
    pattern_functions = {
        'cdl_doji': ta.CDLDOJI,
        'cdl_hammer': ta.CDLHAMMER,
        'cdl_shooting_star': ta.CDLSHOOTINGSTAR,
        'cdl_engulfing': ta.CDLENGULFING,
        'cdl_evening_star': ta.CDLEVENINGSTAR,
        'cdl_morning_star': ta.CDLMORNINGSTAR,
        'cdl_harami': ta.CDLHARAMI,
        'cdl_three_white_soldiers': ta.CDL3WHITESOLDIERS,
        'cdl_three_black_crows': ta.CDL3BLACKCROWS,
        'cdl_spinning_top': ta.CDLSPINNINGTOP,
        'cdl_belt_hold': ta.CDLBELTHOLD
    }
    
    for name, func in pattern_functions.items():
        df[name] = func(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
    
    logger.info("Added pattern recognition features")
    return df

def add_time_features(df):
    """Add time-based features from datetime index."""
    df = df.copy()
    
    if 'Datetime' in df.columns:
        # Ensure Datetime is datetime type
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Extract time components
        df['hour'] = df['Datetime'].dt.hour
        df['dayofweek'] = df['Datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day'] = df['Datetime'].dt.day
        df['week'] = df['Datetime'].dt.isocalendar().week
        df['month'] = df['Datetime'].dt.month
        df['quarter'] = df['Datetime'].dt.quarter
        df['year'] = df['Datetime'].dt.year
        
        # Cyclical encoding of hour (converts to sine and cosine components)
        df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
        df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
        
        # Cyclical encoding of day of week
        df['dayofweek_sin'] = np.sin(df['dayofweek'] * (2 * np.pi / 7))
        df['dayofweek_cos'] = np.cos(df['dayofweek'] * (2 * np.pi / 7))
        
        # Cyclical encoding of month
        df['month_sin'] = np.sin(df['month'] * (2 * np.pi / 12))
        df['month_cos'] = np.cos(df['month'] * (2 * np.pi / 12))
        
        # Market session flags (simplified, adjust time ranges as needed)
        # Sydney: 22-7 UTC, Tokyo: 0-9 UTC, London: 8-17 UTC, New York: 13-22 UTC
        df['sydney_session'] = ((df['hour'] >= 22) | (df['hour'] < 7)).astype(int)
        df['tokyo_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
        df['newyork_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        
        # Overlap sessions (high liquidity periods)
        df['london_newyork_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 17)).astype(int)
        df['tokyo_london_overlap'] = ((df['hour'] >= 8) & (df['hour'] < 9)).astype(int)
        
        # Weekend flag (low liquidity, high spreads)
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        logger.info("Added time-based features")
    else:
        logger.warning("No 'Datetime' column found, skipping time features")
    
    return df

def add_custom_indicators(df):
    """Add custom indicators and features."""
    df = df.copy()
    
    # Elder's Force Index
    df['elder_force_index'] = df['Close'].diff(1) * df['Volume']
    df['elder_force_index_13'] = ta.EMA(df['elder_force_index'].values, timeperiod=13)
    
    # Ease of Movement
    df['eom'] = ((df['High'] + df['Low']) / 2 - (df['High'].shift(1) + df['Low'].shift(1)) / 2) / (df['Volume'] / (df['High'] - df['Low']))
    df['eom_14'] = df['eom'].rolling(14).mean()
    
    # Accumulation/Distribution Line
    df['adl'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['adl'] = df['adl'].cumsum()
    
    # Volume Weighted Average Price (intraday)
    df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Price momentum across multiple timeframes
    for period in [3, 5, 10, 15, 20]:
        df[f'momentum_{period}'] = df['Close'] - df['Close'].shift(period)
    
    # Convergence/Divergence of indicators
    df['sma_converge_5_10'] = df['sma_5'] - df['sma_10']
    df['sma_converge_10_20'] = df['sma_10'] - df['sma_20']
    df['sma_converge_20_50'] = df['sma_20'] - df['sma_50']
    
    # Support and resistance regions
    for period in [10, 20, 50]:
        df[f'support_{period}'] = df['Low'].rolling(period).min()
        df[f'resistance_{period}'] = df['High'].rolling(period).max()
        
        # Distance from current price to support/resistance as percentage
        df[f'dist_to_support_{period}'] = (df['Close'] - df[f'support_{period}']) / df['Close'] * 100
        df[f'dist_to_resistance_{period}'] = (df[f'resistance_{period}'] - df['Close']) / df['Close'] * 100
    
    # Bullish/bearish signals
    df['is_sma_5_above_sma_20'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['is_sma_20_above_sma_50'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['is_sma_50_above_sma_200'] = (df['sma_50'] > df['sma_200']).astype(int)
    df['is_macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
    
    # Trend strength
    df['adx_trend_strength'] = np.where(df['adx'] < 20, 0,  # No trend
                                      np.where(df['adx'] < 40, 1,  # Moderate trend
                                             np.where(df['adx'] < 60, 2, 3)))  # Strong/Very strong trend
    
    # Money Flow Index
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    # Positive and Negative Money Flow
    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    
    # Calculate MFI with 14-period window
    pf_14 = pd.Series(positive_flow).rolling(14).sum()
    nf_14 = pd.Series(negative_flow).rolling(14).sum()
    money_ratio = pf_14 / nf_14
    df['mfi_14'] = 100 - (100 / (1 + money_ratio))
    
    logger.info("Added custom indicators")
    return df

def prepare_features(df, future_period=2, target_type='binary', include_patterns=True):
    """
    Main function to create all features and prepare data for model training.
    
    Args:
        df: DataFrame with OHLCV data
        future_period: Number of periods to look ahead for target creation
        target_type: Type of target variable ('binary' or 'ternary')
        include_patterns: Whether to include candlestick pattern features
        
    Returns:
        DataFrame with all features and target
    """
    logger.info("Starting feature preparation...")
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Make sure Datetime is handled properly
    if 'Datetime' not in df.columns and df.index.name == 'Datetime':
        df = df.reset_index()
    
    # Add all feature groups
    df = add_price_features(df)
    df = add_technical_indicators(df)
    df = add_volatility_features(df)
    if include_patterns:
        df = add_pattern_recognition(df)
    df = add_time_features(df)
    df = add_custom_indicators(df)
    
    # Create target column last (to ensure all features are available for target creation)
    df = create_target_column(df, future_period, price_column='Close', target_type=target_type)
    
    # Handle NaN values
    nan_before = df.isna().sum().sum()
    if nan_before > 0:
        logger.info(f"Found {nan_before} NaN values across all columns before cleaning")
    
    # Forward fill some NaN values (for time series consistency)
    df_filled = df.fillna(method='ffill')
    
    # Any remaining NaNs filled with column median for numeric, most frequent for categorical
    numeric_cols = df_filled.select_dtypes(include=['float64', 'int64']).columns
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].median())
    
    # Drop rows still containing NaNs
    df_clean = df_filled.dropna()
    rows_lost = len(df) - len(df_clean)
    if rows_lost > 0:
        logger.warning(f"Dropped {rows_lost} rows containing NaN values")
    
    logger.info(f"Feature preparation complete. Final shape: {df_clean.shape}")
    return df_clean

def scale_features(df, target_col='Target', exclude_cols=None, scaler_file=None):
    """
    Scale features using MinMaxScaler.
    
    Args:
        df: DataFrame with features
        target_col: Name of target column to exclude from scaling
        exclude_cols: List of additional columns to exclude from scaling
        scaler_file: Path to save scaler for later use
    
    Returns:
        DataFrame with scaled features, scaler object
    """
    df = df.copy()
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Add target to excluded columns
    if target_col and target_col in df.columns:
        exclude_cols.append(target_col)
    
    # Also exclude datetime and any binary/categorical columns
    if 'Datetime' in df.columns:
        exclude_cols.append('Datetime')
        
    # Add any columns with _session or is_ prefix which are likely binary
    binary_cols = [col for col in df.columns if col.endswith('_session') or col.startswith('is_')]
    exclude_cols.extend(binary_cols)
    
    # Unique list
    exclude_cols = list(set(exclude_cols))
    
    # Columns to scale
    cols_to_scale = [col for col in df.columns if col not in exclude_cols]
    
    # Scale features
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    # Save scaler if path is provided
    if scaler_file:
        joblib.dump(scaler, scaler_file)
        logger.info(f"Scaler saved to {scaler_file}")
    
    return df, scaler

def plot_feature_distributions(df, feature_cols=None, output_dir=None, max_cols=20):
    """
    Plot distributions of features.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature columns to plot (if None, uses all)
        output_dir: Directory to save plots
        max_cols: Maximum number of columns to plot
    """
    if feature_cols is None:
        # Exclude target and datetime
        feature_cols = [col for col in df.columns if col not in ['Target', 'Datetime']]
    
    # Limit number of features to plot
    if len(feature_cols) > max_cols:
        logger.info(f"Limiting plots to {max_cols} features")
        feature_cols = feature_cols[:max_cols]
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for col in feature_cols:
        try:
            plt.figure(figsize=(10, 6))
            
            # Histogram for numeric data
            if df[col].dtype in ['float64', 'int64']:
                plt.hist(df[col].dropna(), bins=50, alpha=0.7)
                plt.axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {df[col].mean():.2f}')
                plt.axvline(df[col].median(), color='green', linestyle='dashed', linewidth=1, label=f'Median: {df[col].median():.2f}')
            
            # For binary/categorical columns, use value counts
            else:
                counts = df[col].value_counts()
                plt.bar(counts.index.astype(str), counts.values)
            
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'dist_{col}.png'))
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"Error plotting {col}: {e}")
    
    logger.info(f"Feature distribution plots created")

def process_and_save_features(input_file_path, output_file_path=None, future_period=2, 
                             target_type='binary', include_patterns=True, output_dir=None):
    """
    Process raw data, engineer features, and save the results.
    
    Args:
        input_file_path: Path to input raw data CSV
        output_file_path: Path to save processed features CSV
        future_period: Number of periods to look ahead for target creation
        target_type: Type of target variable
        include_patterns: Whether to include candlestick pattern features
        output_dir: Directory to save plots and scaler
        
    Returns:
        DataFrame with processed features
    """
    # Ensure output directory exists
    if output_dir is None:
        output_dir = PROCESSED_OUTPUT_FOLDER
    os.makedirs(output_dir, exist_ok=True)
    
    # Default output filename if not specified
    if output_file_path is None:
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_file_path = os.path.join(output_dir, f"{base_name}_processed.csv")
    
    # Load data
    logger.info(f"Loading raw data from {input_file_path}")
    df = pd.read_csv(input_file_path)
    
    # Process and prepare features
    processed_df = prepare_features(df, future_period, target_type, include_patterns)
    
    # Scale features
    scaler_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(output_file_path))[0]}_scaler.joblib")
    scaled_df, _ = scale_features(processed_df, scaler_file=scaler_file)
    
    # Save processed data
    scaled_df.to_csv(output_file_path, index=False)
    logger.info(f"Processed data saved to {output_file_path}")
    
    # Plot feature distributions
    plot_feature_distributions(scaled_df, output_dir=output_dir)
    
    return scaled_df

if __name__ == "__main__":
    # Example usage
    input_file = "forex_data/mt5_EURUSD_M5.csv"
    output_file = "forex_preprocessed/EURUSD_M5_processed.csv"
    
    processed_data = process_and_save_features(
        input_file_path=input_file,
        output_file_path=output_file,
        future_period=5,  # 5-period lookahead for target
        target_type='binary',
        include_patterns=True
    )
    
    logger.info("Feature engineering completed successfully")
