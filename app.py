import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import ta
from sklearn.preprocessing import MinMaxScaler

# ======= CONFIG =======
RAW_OUTPUT_FOLDER = "forex_data"
PROCESSED_OUTPUT_FOLDER = "forex_preprocessed"
os.makedirs(RAW_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_OUTPUT_FOLDER, exist_ok=True)

# ======= Fetch Yahoo Forex =======
def fetch_yahoo_forex(pair_symbol='EURUSD=X', interval='60m', period='7d'):
    print(f"Fetching Yahoo Finance data for {pair_symbol}...")
    data = yf.download(tickers=pair_symbol, interval=interval, period=period)
    data.reset_index(inplace=True)
    filename = os.path.join(RAW_OUTPUT_FOLDER, f'yahoo_{pair_symbol}_{interval}.csv')
    data.to_csv(filename, index=False)
    print(f"✅ Yahoo data saved: {filename}\n")
    return data

# ======= Fetch Binance Forex =======
def fetch_binance_forex(pair_symbol='EURUSDT', interval='1h', limit=1000):
    print(f"Fetching Binance data for {pair_symbol}...")
    interval_mapping = {
        '1h': '1h',
        '1d': '1d',
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
    }
    url = f"https://api.binance.com/api/v3/klines?symbol={pair_symbol}&interval={interval_mapping[interval]}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    if isinstance(data, dict) and 'code' in data:
        print(f"❌ Binance API error: {data}")
        return None

    df = pd.DataFrame(data, columns=[
        'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades',
        'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
    ])

    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col])

    filename = os.path.join(RAW_OUTPUT_FOLDER, f'binance_{pair_symbol}_{interval}.csv')
    df.to_csv(filename, index=False)
    print(f"✅ Binance data saved: {filename}\n")
    return df

# ======= Preprocess Data =======
def preprocess_forex_data(input_file):
    print(f"\n🔄 Preprocessing {input_file}...")
    df = pd.read_csv(input_file)
    
    if df.empty:
        print("❌ Error: Input file contains no data")
        return
        
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    print(f"Data shape after loading: {df.shape}")

    # Clean Data
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Close'])
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"Data shape after cleaning: {df.shape}")
    print("✅ Cleaned Data")
    
    if df.empty:
        print("❌ Error: No data remains after cleaning")
        return

    # Technical Indicators
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['bollinger_h'] = bollinger.bollinger_hband()
        df['bollinger_l'] = bollinger.bollinger_lband()
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        if len(df) >= 50:
            df['sma_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        else:
            print(f"⚠️ Warning: Not enough data for SMA 50 (only {len(df)} points)")
            df['sma_50'] = np.nan
            
        if len(df) >= 200:
            df['sma_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
        else:
            print(f"⚠️ Warning: Not enough data for SMA 200 (only {len(df)} points)")
            df['sma_200'] = np.nan
            
        print("✅ Technical Indicators Added")
    except Exception as e:
        print(f"❌ Error adding technical indicators: {e}")

    for lag in range(1, 6):
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
    print("✅ Lagged Features Added")

    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    print("✅ Target Added")

    df_complete = df.copy()
    df = df.dropna()
    
    print(f"Data shape after dropping NaNs: {df.shape}")
    
    if df.empty:
        print("❌ Error: No complete rows remain after creating features")
        out_file = os.path.join(PROCESSED_OUTPUT_FOLDER, 'incomplete_features.csv')
        df_complete.to_csv(out_file)
        print(f"⚠️ Incomplete feature data saved for inspection: {out_file}")
        return

    scaler = MinMaxScaler()
    
    cols_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'macd_signal',
                       'bollinger_h', 'bollinger_l', 'atr']
    
    if 'sma_50' in df.columns and not df['sma_50'].isna().all():
        cols_to_normalize.append('sma_50')
    if 'sma_200' in df.columns and not df['sma_200'].isna().all():
        cols_to_normalize.append('sma_200')
    
    lag_cols = [f'close_lag_{i}' for i in range(1, 6)]
    cols_to_normalize.extend(lag_cols)
    
    cols_to_normalize = [col for col in cols_to_normalize if col in df.columns]
    
    print(f"Columns to normalize: {cols_to_normalize}")
    print(f"Sample data before normalization:\n{df[cols_to_normalize].head()}")
    
    try:
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        print("✅ Normalized Data")
    except Exception as e:
        print(f"❌ Error during normalization: {e}")
        print("⚠️ Continuing without normalization")

    out_file = os.path.join(PROCESSED_OUTPUT_FOLDER, 'finalfeature.csv')
    df.to_csv(out_file)
    print(f"🎉 Final feature data saved: {out_file}\n")
    
    recent_file = os.path.join(PROCESSED_OUTPUT_FOLDER, 'recent_data.csv')
    df.tail(5).to_csv(recent_file)
    print(f"📊 Recent data saved for predictions: {recent_file}\n")

# ======= Main Run =======
if __name__ == "__main__":
    yahoo_df = fetch_yahoo_forex(pair_symbol='EURUSD=X', interval='60m', period='30d')

    binance_df = fetch_binance_forex(pair_symbol='EURUSDT', interval='1h', limit=1000)

    print("✅ All raw data fetched and saved!\n")

    yahoo_file = os.path.join(RAW_OUTPUT_FOLDER, 'yahoo_EURUSD=X_60m.csv')
    preprocess_forex_data(yahoo_file)
    
    binance_file = os.path.join(RAW_OUTPUT_FOLDER, 'binance_EURUSDT_1h.csv')
    preprocess_forex_data(binance_file)

    print("🏁 DONE: All data fetched and preprocessed!")