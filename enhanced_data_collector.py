import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import logging
import pytz
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
RAW_DATA_FOLDER = "forex_data"
os.makedirs(RAW_DATA_FOLDER, exist_ok=True)

def initialize_mt5(username=None, password=None, server=None):
    """Initialize connection to MetaTrader 5 terminal with optional login credentials."""
    if not mt5.initialize():
        logger.error("❌ Failed to initialize MT5")
        return False
    
    # If login credentials are provided, attempt to login
    if username and password and server:
        authorized = mt5.login(username, password, server)
        if not authorized:
            logger.error(f"❌ Failed to login to MT5. Error: {mt5.last_error()}")
            mt5.shutdown()
            return False
        logger.info("✅ Successfully logged in to MT5")
    
    logger.info("✅ MetaTrader 5 initialized successfully")
    return True

def get_symbols():
    """Get all available symbols from MT5."""
    if not mt5.initialize():
        logger.error("❌ Failed to initialize MT5")
        return []
    
    try:
        symbols = mt5.symbols_get()
        symbol_names = [symbol.name for symbol in symbols]
        logger.info(f"✅ Retrieved {len(symbol_names)} symbols from MT5")
        return symbol_names
    except Exception as e:
        logger.error(f"❌ Error retrieving symbols: {e}")
        return []
    finally:
        mt5.shutdown()

def fetch_mt5_historical_data(symbol, timeframe, start_date=None, end_date=None, num_bars=None):
    """
    Fetch historical data directly from MetaTrader 5.
    
    Args:
        symbol (str): Symbol to fetch data for (e.g., 'EURUSD')
        timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_M5)
        start_date (datetime, optional): Start date for data
        end_date (datetime, optional): End date for data
        num_bars (int, optional): Number of bars to fetch if dates not specified
    
    Returns:
        pd.DataFrame: DataFrame with historical data
    """
    logger.info(f"Fetching {symbol} data from MetaTrader 5...")
    
    if not initialize_mt5():
        return pd.DataFrame()
    
    try:
        # Map timeframe to human-readable string for logging/filenames
        timeframe_map = {
            mt5.TIMEFRAME_M1: "M1",
            mt5.TIMEFRAME_M5: "M5",
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "H1",
            mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1",
        }
        timeframe_str = timeframe_map.get(timeframe, str(timeframe))
        
        # Fetch data based on parameters provided
        if start_date and end_date:
            # Convert dates to UTC timestamp
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            rates = mt5.copy_rates_range(symbol, timeframe, start_timestamp, end_timestamp)
        else:
            # Default: fetch the specified number of bars or 5000 if not specified
            num_bars = num_bars or 5000
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        
        if rates is None or len(rates) == 0:
            logger.error(f"❌ Failed to get rates from MT5. Error: {mt5.last_error()}")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Standardize column names
        df.rename(columns={
            'time': 'Datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume',
            'spread': 'Spread',
            'real_volume': 'RealVolume'
        }, inplace=True)
        
        # Select and reorder columns (include spread and real_volume if available)
        columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        if 'Spread' in df.columns:
            columns.append('Spread')
        if 'RealVolume' in df.columns:
            columns.append('RealVolume')
        
        df = df[columns]
        
        # Save to file
        filename = os.path.join(RAW_DATA_FOLDER, f'mt5_{symbol}_{timeframe_str}.csv')
        df.to_csv(filename, index=False)
        logger.info(f"✅ MT5 data saved: {filename} (Shape: {df.shape})")
        
        return df
    
    except Exception as e:
        logger.error(f"❌ Error fetching MT5 data: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        mt5.shutdown()

def fetch_multiple_symbols(symbols, timeframe, start_date=None, end_date=None, num_bars=None):
    """Fetch historical data for multiple symbols."""
    all_data = {}
    
    for symbol in symbols:
        try:
            df = fetch_mt5_historical_data(symbol, timeframe, start_date, end_date, num_bars)
            if not df.empty:
                all_data[symbol] = df
                logger.info(f"✅ Successfully fetched data for {symbol}")
            else:
                logger.warning(f"⚠️ Empty data returned for {symbol}")
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol}: {e}")
    
    return all_data

def fetch_multi_timeframe_data(symbol, timeframes, start_date=None, end_date=None, num_bars=None):
    """Fetch data for a single symbol across multiple timeframes."""
    all_data = {}
    
    for timeframe in timeframes:
        try:
            df = fetch_mt5_historical_data(symbol, timeframe, start_date, end_date, num_bars)
            if not df.empty:
                all_data[timeframe] = df
                logger.info(f"✅ Successfully fetched {symbol} data for timeframe {timeframe}")
            else:
                logger.warning(f"⚠️ Empty data returned for {symbol} on timeframe {timeframe}")
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol} on timeframe {timeframe}: {e}")
    
    return all_data

def get_account_info():
    """Get detailed account information from MT5."""
    if not initialize_mt5():
        return None
    
    try:
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"❌ Failed to get account info. Error: {mt5.last_error()}")
            return None
        
        # Convert account info to dictionary
        account_dict = {
            "Login": account_info.login,
            "Server": account_info.server,
            "Currency": account_info.currency,
            "Balance": account_info.balance,
            "Equity": account_info.equity,
            "Margin": account_info.margin,
            "FreeMargin": account_info.margin_free,
            "MarginLevel": account_info.margin_level,
            "LeverageValue": account_info.leverage
        }
        
        logger.info(f"✅ Successfully retrieved account information")
        logger.info(f"Account Balance: {account_dict['Balance']} {account_dict['Currency']}")
        logger.info(f"Account Equity: {account_dict['Equity']} {account_dict['Currency']}")
        
        return account_dict
    
    except Exception as e:
        logger.error(f"❌ Error retrieving account info: {e}")
        return None
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    # Example usage
    initialize_mt5()
    
    # Get account info
    account_info = get_account_info()
    
    # Example: Fetch last 5000 bars of EURUSD 5-minute data
    eurusd_m5 = fetch_mt5_historical_data("EURUSD", mt5.TIMEFRAME_M5, num_bars=5000)
    
    # Example: Fetch multiple timeframes
    timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]
    multi_tf_data = fetch_multi_timeframe_data("EURUSD", timeframes, num_bars=2000)
    
    # Example: Fetch last 1000 bars for multiple pairs
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    multi_symbol_data = fetch_multiple_symbols(symbols, mt5.TIMEFRAME_M5, num_bars=1000)
    
    logger.info("✅ Data collection completed")
