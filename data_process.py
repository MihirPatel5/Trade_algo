import pandas as pd
import numpy as np
import os, ta
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = 'forex_data/yahoo_EURUSD=X_1h.csv'
OUTPUT_FOLDER = 'forex_preprocessed'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(INPUT_FILE, parse_dates=['Datetime'])
df.set_index('Datetime', inplace=True)
print("✅ Original Data Loaded:")
print(df.head())

df = df.dropna()
df = df[~df.index.duplicated(keep='first')]
df = df[pd.to_numeric(df['Close'], errors='coerce').notnull()]

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n✅ Cleaned Numeric Data:")
print(df.head())

df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
macd = ta.trend.MACD(df['Close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()

bollinger = ta.volatility.BollingerBands(df['Close'])
df['bollinger_h'] = bollinger.bollinger_hband()
df['bollinger_l'] = bollinger.bollinger_lband()

df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
df['sma_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
df['sma_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()

print("\n✅ Technical Indicators Added:")
print(df[['rsi', 'macd', 'bollinger_h', 'atr']].tail())

for lag in range(1, 6):
    df[f'close_lag_{lag}'] = df['Close'].shift(lag)
print("\n✅ Lagged Features Added:")
print(df[[f'close_lag_{i}' for i in range(1, 6)]].tail())

scaler = MinMaxScaler()
cols_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'macd_signal', 
                     'bollinger_h', 'bollinger_l', 'atr', 'sma_50', 'sma_200'] + \
                    [f'close_lag_{i}' for i in range(1, 6)]

df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

print("\n✅ Normalized Data Sample:")
print(df[cols_to_normalize].head())

pair_name = os.path.basename(INPUT_FILE).replace('.csv', '').replace('yahoo_', '').replace('alpha_', '')
out_file = os.path.join(OUTPUT_FOLDER, f'{pair_name}_preprocessed.csv')
df.dropna().to_csv(out_file)

print(f"\n🎉 Preprocessed data saved to '{out_file}'")
