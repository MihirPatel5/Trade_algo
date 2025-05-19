import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import time
import logging
import json
import os
from datetime import datetime, timedelta
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForexTradingBot:
    """Multi-trade forex trading bot using ML predictions."""
    
    def __init__(self, model_path, timeframe=mt5.TIMEFRAME_M5, 
                 symbols=None, lot_size=0.01, risk_per_trade=0.02,
                 stop_loss_atr_multiplier=2.0, take_profit_atr_multiplier=3.0,
                 max_spread_pips=5, max_trades_per_symbol=1, max_total_trades=10):
        """
        Initialize the trading bot.
        
        Args:
            model_path: Path to trained model
            timeframe: MT5 timeframe constant
            symbols: List of symbols to trade
            lot_size: Position size in lots (0.01 = micro lot)
            risk_per_trade: Risk per trade as fraction of account balance
            stop_loss_atr_multiplier: Multiplier for ATR to set stop loss
            take_profit_atr_multiplier: Multiplier for ATR to set take profit
            max_spread_pips: Maximum allowed spread in pips
            max_trades_per_symbol: Maximum trades allowed per symbol
            max_total_trades: Maximum total trades allowed
        """
        self.model_path = model_path
        self.timeframe = timeframe
        # Default to major forex pairs if none provided
        self.symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
        self.lot_size = lot_size
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier
        self.max_spread_pips = max_spread_pips
        self.max_trades_per_symbol = max_trades_per_symbol
        self.max_total_trades = max_total_trades
        
        # Trading state
        self.model = None
        self.feature_names = None
        self.is_trading = False
        self.active_trades = {}
        self.trade_count = 0
        self.trade_history = []
        
        # Initialize lock for thread safety
        self.lock = threading.Lock()
        self.trade_queue = queue.Queue()
        
        # Log initialization
        logger.info(f"Forex Trading Bot initialized with {len(self.symbols)} symbols")
        logger.info(f"Max trades per symbol: {max_trades_per_symbol}, Max total trades: {max_total_trades}")
    
    def load_model(self):
        """Load the trained ML model."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded: {type(self.model)}")
            
            # Try to load feature names from model metadata
            metadata_path = os.path.join(os.path.dirname(self.model_path), "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.feature_names = metadata[0]['feature_names']
                logger.info(f"Loaded feature names from metadata: {len(self.feature_names)} features")
            else:
                logger.warning("Model metadata not found, will need to provide feature names manually")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def connect_to_mt5(self, username=None, password=None, server=None):
        """Connect to MetaTrader 5 terminal."""
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return False
        
        # Login if credentials provided
        if username and password and server:
            if not mt5.login(username, password, server):
                logger.error(f"MT5 login failed. Error: {mt5.last_error()}")
                mt5.shutdown()
                return False
            logger.info("Successfully logged in to MT5")
        
        # Get account info to confirm connection
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            mt5.shutdown()
            return False
        
        logger.info(f"Connected to MT5: Account #{account_info.login}")
        logger.info(f"Balance: {account_info.balance} {account_info.currency}")
        logger.info(f"Leverage: 1:{account_info.leverage}")
        
        return True
    
    def fetch_and_prepare_data(self, symbol, num_bars=500):
        """Fetch data for a symbol and prepare features."""
        if not mt5.initialize():
            logger.error("MT5 not initialized")
            return None
        
        logger.info(f"Fetching data for {symbol} ({num_bars} bars)...")
        
        # Get historical data
        rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, num_bars)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates for {symbol}. Error: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
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
        
        try:
            # Import feature engineering module
            from feature_engineering import prepare_features, scale_features
            
            # Prepare features
            df_features = prepare_features(df, future_period=1, target_type='binary')
            
            # Scale features (excluding target and datetime)
            df_scaled, _ = scale_features(df_features, target_col='Target', 
                                       exclude_cols=['Datetime', 'future_price', 'future_pct_change'])
            
            return df_scaled
            
        except ImportError:
            logger.error("Failed to import feature_engineering module")
            return None
        except Exception as e:
            logger.error(f"Error preparing features for {symbol}: {e}")
            return None
    
    def predict_trade_signal(self, prepared_data, symbol):
        """Generate trade signal from prepared data using the ML model."""
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        if prepared_data is None or prepared_data.empty:
            logger.error(f"No data available for {symbol}")
            return None
        
        try:
            # Extract features for prediction
            if self.feature_names:
                X = prepared_data[self.feature_names].iloc[-1:].copy()
            else:
                # If feature names not provided, exclude known non-feature columns
                exclude_cols = ['Target', 'Datetime', 'future_price', 'future_pct_change', 
                               'Open', 'High', 'Low', 'Close', 'Volume']
                X = prepared_data.iloc[-1:].drop([col for col in exclude_cols if col in prepared_data.columns], axis=1)
                self.feature_names = X.columns.tolist()
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Get prediction probability if available
            probability = 0.5
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(X)[0][1]
            
            signal = {
                'symbol': symbol,
                'action': 'BUY' if prediction == 1 else 'SELL',
                'confidence': float(probability),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'price': prepared_data['Close'].iloc[-1],
                'atr': prepared_data.get('atr_14', pd.Series([0])).iloc[-1]
            }
            
            logger.info(f"Signal for {symbol}: {signal['action']} (Confidence: {signal['confidence']:.4f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, signal, account_info):
        """Calculate position size based on risk management parameters."""
        try:
            symbol_info = mt5.symbol_info(signal['symbol'])
            if symbol_info is None:
                logger.error(f"Symbol info not available for {signal['symbol']}")
                return self.lot_size
            
            # Calculate ATR-based stop loss in points
            atr = signal['atr']
            if atr == 0:  # If ATR not available or zero
                stop_loss_points = 100  # Default value
            else:
                # Convert ATR to points
                stop_loss_points = int(atr * self.stop_loss_atr_multiplier * 10000)
            
            # Ensure minimum stop loss
            stop_loss_points = max(stop_loss_points, 50)
            
            # Calculate position size based on risk
            risk_amount = account_info.balance * self.risk_per_trade
            pip_value = symbol_info.trade_tick_value * (10 / symbol_info.point)  # Value of 1 pip
            
            # Position size formula: Risk amount / (Stop loss in pips * Pip value)
            position_size = risk_amount / (stop_loss_points * pip_value / 10000)
            
            # Round to nearest lot step
            lot_step = symbol_info.volume_step
            position_size = round(position_size / lot_step) * lot_step
            
            # Ensure minimum and maximum lot size
            position_size = max(position_size, symbol_info.volume_min)
            position_size = min(position_size, symbol_info.volume_max, 1.0)  # Cap at 1.0 lot for safety
            
            logger.info(f"Calculated position size for {signal['symbol']}: {position_size} lots")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.lot_size
    
    def execute_trade(self, signal):
        """Execute a trade based on the signal."""
        if not mt5.initialize():
            logger.error("MT5 not initialized")
            return False
        
        # Check if we already have a trade for this symbol
        with self.lock:
            symbol_trades = sum(1 for trade in self.active_trades.values() if trade['symbol'] == signal['symbol'])
            if symbol_trades >= self.max_trades_per_symbol:
                logger.warning(f"Maximum trades ({self.max_trades_per_symbol}) reached for {signal['symbol']}")
                return False
            
            # Check total trades
            if len(self.active_trades) >= self.max_total_trades:
                logger.warning(f"Maximum total trades ({self.max_total_trades}) reached")
                return False
        
        # Check spread
        symbol_info = mt5.symbol_info(signal['symbol'])
        if symbol_info is None:
            logger.error(f"Symbol info not available for {signal['symbol']}")
            return False
        
        # Check if spread is too high
        current_spread = symbol_info.spread
        if current_spread > self.max_spread_pips * 10:  # Convert pips to points
            logger.warning(f"Spread too high for {signal['symbol']}: {current_spread} points")
            return False
        
        # Get account info for position sizing
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return False
        
        # Calculate position size
        position_size = self.calculate_position_size(signal, account_info)
        
        # Prepare trade request
        action = mt5.ORDER_TYPE_BUY if signal['action'] == 'BUY' else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(signal['symbol']).ask if action == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(signal['symbol']).bid
        
        # Calculate stop loss and take profit levels
        atr = signal['atr']
        if atr == 0:
            sl_points = 100
            tp_points = 150
        else:
            sl_points = int(atr * self.stop_loss_atr_multiplier * 10000)
            tp_points = int(atr * self.take_profit_atr_multiplier * 10000)
        
        # Ensure minimum values
        sl_points = max(sl_points, 50)
        tp_points = max(tp_points, 75)
        
        # Calculate actual price levels for SL and TP
        if action == mt5.ORDER_TYPE_BUY:
            sl = price - sl_points * symbol_info.point
            tp = price + tp_points * symbol_info.point
        else:
            sl = price + sl_points * symbol_info.point
            tp = price - tp_points * symbol_info.point
        
        # Create trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal['symbol'],
            "volume": position_size,
            "type": action,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,  # Identifier for this bot
            "comment": f"ML Signal: {signal['confidence']:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send trade
        logger.info(f"Sending {signal['action']} order for {signal['symbol']} at {price}...")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed, retcode={result.retcode}")
            logger.error(f"Error message: {mt5.last_error()}")
            return False
        
        # Record trade
        trade_id = result.order
        
        # Store trade info
        with self.lock:
            self.active_trades[trade_id] = {
                "id": trade_id,
                "symbol": signal['symbol'],
                "type": "BUY" if action == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": position_size,
                "price": price,
                "sl": sl,
                "tp": tp,
                "time": datetime.now(),
                "signal_confidence": signal['confidence']
            }
            self.trade_count += 1
        
        logger.info(f"✅ Trade {trade_id} executed successfully: {signal['action']} {signal['symbol']} at {price}")
        return True
    
    def close_trade(self, trade_id):
        """Close a specific trade by its ID."""
        if not mt5.initialize():
            logger.error("MT5 not initialized")
            return False
        
        with self.lock:
            if trade_id not in self.active_trades:
                logger.warning(f"Trade {trade_id} not found in active trades")
                return False
            
            trade = self.active_trades[trade_id]
        
        # Get current position
        positions = mt5.positions_get(ticket=trade_id)
        if not positions:
            logger.warning(f"Position {trade_id} not found in MT5")
            
            # Clean up our records even if position not found
            with self.lock:
                if trade_id in self.active_trades:
                    del self.active_trades[trade_id]
                    
            return False
        
        position = positions[0]
        
        # Prepare close request
        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(position.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Close by trading bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close order
        logger.info(f"Closing trade {trade_id} for {position.symbol}...")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close trade {trade_id}, retcode={result.retcode}")
            logger.error(f"Error message: {mt5.last_error()}")
            return False
        
        # Update records
        with self.lock:
            if trade_id in self.active_trades:
                # Calculate profit/loss
                trade_info = self.active_trades[trade_id]
                trade_info["close_price"] = price
                trade_info["close_time"] = datetime.now()
                trade_info["profit"] = position.profit
                
                # Move to history
                self.trade_history.append(trade_info)
                
                # Remove from active trades
                del self.active_trades[trade_id]
        
        logger.info(f"✅ Successfully closed trade {trade_id} with profit: {position.profit}")
        return True
    
    def manage_open_trades(self):
        """Check and manage open trades (modify stops, partial close, etc.)"""
        if not mt5.initialize():
            logger.error("MT5 not initialized")
            return
        
        # Get all open positions
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            return
        
        # Process each position
        for position in positions:
            # Check if it's our trade
            if position.magic != 123456:
                continue
            
            trade_id = position.ticket
            
            # Skip if not in our active trades
            with self.lock:
                if trade_id not in self.active_trades:
                    continue
                
                trade = self.active_trades[trade_id]
            
            # Get current price
            symbol_info = mt5.symbol_info(position.symbol)
            current_price = symbol_info.ask if position.type == mt5.ORDER_TYPE_SELL else symbol_info.bid
            
            # Calculate profit in pips
            entry_price = position.price_open
            point_value = symbol_info.point
            
            if position.type == mt5.ORDER_TYPE_BUY:
                profit_points = (current_price - entry_price) / point_value
            else:
                profit_points = (entry_price - current_price) / point_value
            
            # Implement trailing stop if in profit
            if profit_points > 200:  # 20 pips
                # Calculate new stop loss level
                if position.type == mt5.ORDER_TYPE_BUY:
                    new_sl = current_price - 100 * point_value  # 10 pips below current price
                    if new_sl > position.sl + 50 * point_value:  # Only move if significant improvement
                        self.modify_trade_sl_tp(trade_id, new_sl=new_sl)
                else:
                    new_sl = current_price + 100 * point_value  # 10 pips above current price
                    if new_sl < position.sl - 50 * point_value:  # Only move if significant improvement
                        self.modify_trade_sl_tp(trade_id, new_sl=new_sl)
    
    def modify_trade_sl_tp(self, trade_id, new_sl=None, new_tp=None):
        """Modify stop loss or take profit for an open trade."""
        if not mt5.initialize():
            logger.error("MT5 not initialized")
            return False
        
        # Get position
        positions = mt5.positions_get(ticket=trade_id)
        if not positions:
            logger.warning(f"Position {trade_id} not found")
            return False
        
        position = positions[0]
        
        # Prepare modify request
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "symbol": position.symbol,
            "position": trade_id,
            "magic": 123456,
        }
        
        # Set new SL/TP if provided, otherwise keep existing
        if new_sl is not None:
            request["sl"] = new_sl
        else:
            request["sl"] = position.sl
            
        if new_tp is not None:
            request["tp"] = new_tp
        else:
            request["tp"] = position.tp
        
        # Send modification request
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to modify trade {trade_id}, retcode={result.retcode}")
            logger.error(f"Error message: {mt5.last_error()}")
            return False
        
        logger.info(f"✅ Successfully modified trade {trade_id} (SL: {request['sl']}, TP: {request['tp']})")
        return True
    
    def process_symbol(self, symbol):
        """Process a single symbol for trading."""
        try:
            # Fetch and prepare data
            data = self.fetch_and_prepare_data(symbol)
            if data is None or data.empty:
                logger.warning(f"No data for {symbol}, skipping")
                return
            
            # Generate signal
            signal = self.predict_trade_signal(data, symbol)
            if signal is None:
                logger.warning(f"No signal generated for {symbol}")
                return
            
            # Execute trade if signal is strong enough
            confidence_threshold = 0.65  # Adjust based on model performance
            if (signal['action'] == 'BUY' and signal['confidence'] > confidence_threshold) or \
               (signal['action'] == 'SELL' and signal['confidence'] < (1 - confidence_threshold)):
                self.trade_queue.put(signal)
            else:
                logger.info(f"Signal for {symbol} not strong enough: {signal['confidence']:.4f}")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    def trade_processor(self):
        """Process trade signals from the queue."""
        while self.is_trading:
            try:
                signal = self.trade_queue.get(timeout=1)
                self.execute_trade(signal)
            except queue.Empty:
                # No trades in queue, continue
                pass
            except Exception as e:
                logger.error(f"Error processing trade: {e}")
            finally:
                # Always mark task as done
                try:
                    self.trade_queue.task_done()
                except:
                    pass
    
    def start_trading(self, username=None, password=None, server=None):
        """Start the trading bot."""
        # Load model
        if not self.load_model():
            logger.error("Failed to load model, cannot start trading")
            return False
        
        # Connect to MT5
        if not self.connect_to_mt5(username, password, server):
            logger.error("Failed to connect to MT5, cannot start trading")
            return False
        
        # Start trading
        self.is_trading = True
        
        # Start trade processor thread
        trade_processor_thread = threading.Thread(target=self.trade_processor)
        trade_processor_thread.daemon = True
        trade_processor_thread.start()
        
        logger.info("Trading bot started")
        
        try:
            while self.is_trading:
                # Process each symbol in a thread pool
                with ThreadPoolExecutor(max_workers=len(self.symbols)) as executor:
                    executor.map(self.process_symbol, self.symbols)
                
                # Manage open trades
                self.manage_open_trades()
                
                # Wait before next cycle
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            self.stop_trading()
        
        return True
    
    def stop_trading(self):
        """Stop the trading bot."""
        self.is_trading = False
        
        # Shutdown MT5 connection
        mt5.shutdown()
        
        # Save trading history
        self.save_trading_history()
        
        logger.info("Trading bot stopped")
        return True
    
    def close_all_trades(self):
        """Close all open trades."""
        if not mt5.initialize():
            logger.error("MT5 not initialized")
            return False
        
        logger.info("Closing all open trades...")
        
        # Get all trade IDs
        with self.lock:
            trade_ids = list(self.active_trades.keys())
        
        # Close each trade
        for trade_id in trade_ids:
            self.close_trade(trade_id)
        
        return True
    
    def save_trading_history(self, filename="trading_history.json"):
        """Save trading history to file."""
        output_path = filename
        
        with self.lock:
            history = copy.deepcopy(self.trade_history)
            active = copy.deepcopy(self.active_trades)
        
        # Convert datetime objects to strings
        for trade in history:
            if isinstance(trade.get('time'), datetime):
                trade['time'] = trade['time'].strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(trade.get('close_time'), datetime):
                trade['close_time'] = trade['close_time'].strftime('%Y-%m-%d %H:%M:%S')
        
        for trade in active.values():
            if isinstance(trade.get('time'), datetime):
                trade['time'] = trade['time'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Combine into single structure
        trade_data = {
            "history": history,
            "active": list(active.values()),
            "total_trades": self.trade_count,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to file
        try:
            with open(output_path, 'w') as f:
                json.dump(trade_data, f, indent=4)
            logger.info(f"Trading history saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving trading history: {e}")
            return False

def main():
    """Main function to run the trading bot."""
    import argparse
    parser = argparse.ArgumentParser(description="Forex Trading Bot")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--timeframe", type=int, default=mt5.TIMEFRAME_M5, help="MT5 timeframe")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to trade")
    parser.add_argument("--lot_size", type=float, default=0.01, help="Lot size")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk per trade (fraction of account)")
    parser.add_argument("--username", type=int, help="MT5 username")
    parser.add_argument("--password", type=str, help="MT5 password")
    parser.add_argument("--server", type=str, help="MT5 server")
    
    args = parser.parse_args()
    
    # Create trading bot
    bot = ForexTradingBot(
        model_path=args.model,
        timeframe=args.timeframe,
        symbols=args.symbols,
        lot_size=args.lot_size,
        risk_per_trade=args.risk
    )
    
    # Start trading
    bot.start_trading(args.username, args.password, args.server)

if __name__ == "__main__":
    main()
