#!/usr/bin/env python3
"""
Forex Trading Algorithm - Main Script
====================================
This script orchestrates the complete forex trading pipeline:
1. Data collection from MT5
2. Feature engineering and preprocessing
3. Model training with comprehensive metrics
4. Backtesting with multiple strategies
5. Trading execution with multiple symbols

Run this script to execute the complete workflow or specific components.
"""

import os
import argparse
import logging
import sys
import datetime
import pandas as pd
import MetaTrader5 as mt5
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forex_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Ensure all necessary directories exist."""
    directories = [
        "forex_data",
        "forex_preprocessed",
        "forex_models",
        "forex_backtest_results"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")

def collect_data(symbols, timeframe, num_bars):
    """Collect data from MetaTrader 5."""
    logger.info("=== Starting Data Collection ===")
    
    from enhanced_data_collector import initialize_mt5, fetch_mt5_historical_data
    
    if not initialize_mt5():
        logger.error("Failed to initialize MT5, aborting data collection")
        return False
    
    # Convert timeframe string to MT5 constant
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
    
    success = True
    collected_files = []
    
    for symbol in symbols:
        try:
            logger.info(f"Collecting data for {symbol} ({timeframe}, {num_bars} bars)")
            df = fetch_mt5_historical_data(symbol, mt5_timeframe, num_bars=num_bars)
            
            if df is not None and not df.empty:
                file_path = os.path.join("forex_data", f"mt5_{symbol}_{timeframe}.csv")
                collected_files.append(file_path)
                logger.info(f"Successfully collected data for {symbol}")
            else:
                logger.error(f"Failed to collect data for {symbol}")
                success = False
                
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            success = False
    
    mt5.shutdown()
    
    if success:
        logger.info(f"Data collection completed successfully for {len(collected_files)} symbols")
    else:
        logger.warning("Data collection completed with some errors")
    
    logger.info("=== Data Collection Finished ===")
    return collected_files

def process_features(input_files, future_period=2, target_type='binary'):
    """Process raw data files and engineer features."""
    logger.info("=== Starting Feature Engineering ===")
    
    from feature_engineering import process_and_save_features
    
    processed_files = []
    
    for input_file in input_files:
        try:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join("forex_preprocessed", f"{base_name}_processed.csv")
            
            logger.info(f"Processing features for {input_file}")
            
            df = process_and_save_features(
                input_file_path=input_file,
                output_file_path=output_file,
                future_period=future_period,
                target_type=target_type,
                include_patterns=True
            )
            
            if df is not None and not df.empty:
                processed_files.append(output_file)
                logger.info(f"Successfully processed features for {input_file}")
            else:
                logger.error(f"Failed to process features for {input_file}")
                
        except Exception as e:
            logger.error(f"Error processing features for {input_file}: {e}")
    
    logger.info(f"Feature engineering completed for {len(processed_files)} files")
    logger.info("=== Feature Engineering Finished ===")
    return processed_files

def train_model(data_file, n_trials=50):
    """Train ML model on processed features."""
    logger.info("=== Starting Model Training ===")
    
    from enhanced_ml_pipeline import MLModelTrainer
    
    try:
        # Extract symbol info for model naming
        file_basename = os.path.basename(data_file)
        symbol_info = file_basename.split('_')[1:3]  # e.g., ['EURUSD', 'M5']
        symbol_str = '_'.join(symbol_info)
        
        model_filename = f"{symbol_str}_model.pkl"
        
        logger.info(f"Training model on {data_file} with {n_trials} optimization trials")
        
        trainer = MLModelTrainer(
            data_path=data_file,
            target_col="Target",
            test_size=0.2,
            model_output_path="forex_models",
            n_trials=n_trials,
            cv_splits=5
        )
        
        model_path, metrics = trainer.run_full_pipeline(
            data_path=data_file,
            model_filename=model_filename,
            time_based_split=True
        )
        
        logger.info(f"Model trained successfully: {model_path}")
        logger.info(f"Model metrics: {metrics}")
        
        logger.info("=== Model Training Finished ===")
        return model_path
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        logger.info("=== Model Training Failed ===")
        return None

def run_backtest(data_file, model_path):
    """Run backtesting on trained model."""
    logger.info("=== Starting Backtesting ===")
    
    from advanced_backtester import ForexBacktester
    
    try:
        # Default parameters
        price_col = "Close"
        target_col = "Target"
        datetime_col = "Datetime"
        
        logger.info(f"Running backtest on {data_file} with model {model_path}")
        
        backtester = ForexBacktester(
            data_path=data_file,
            model_path=model_path,
            price_col=price_col,
            target_col=target_col,
            datetime_col=datetime_col,
            commission=0.0001,
            slippage=0.0001
        )
        
        # Load data and model
        backtester.load_data_and_model()
        
        # Run standard backtest
        standard_results = backtester.standard_backtest()
        
        # Plot equity curve
        backtester.plot_equity_curve(output_file="standard_backtest_equity.png")
        
        # Run walk-forward backtest
        walk_forward_results = backtester.walk_forward_backtest(
            initial_train_size=0.5,
            retrain_frequency=20
        )
        
        # Plot walk-forward equity curve
        backtester.plot_equity_curve(walk_forward_results, output_file="walk_forward_equity.png")
        
        # Run Monte Carlo simulation (lighter version)
        mc_results = backtester.monte_carlo_backtest(
            n_simulations=100,
            window_fraction=0.5
        )
        
        logger.info(f"Backtesting completed successfully")
        logger.info(f"Standard backtest metrics: {backtester.metrics}")
        
        logger.info("=== Backtesting Finished ===")
        return True
        
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        logger.info("=== Backtesting Failed ===")
        return False

def run_trading(model_path, symbols, timeframe, lot_size=0.01, risk=0.02):
    """Run live trading with the trained model."""
    logger.info("=== Starting Trading Execution ===")
    
    from trade_executor import ForexTradingBot
    
    try:
        # Convert timeframe string to MT5 constant
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
        
        logger.info(f"Starting trading bot with model {model_path}")
        logger.info(f"Trading symbols: {symbols}")
        logger.info(f"Timeframe: {timeframe}, Lot size: {lot_size}, Risk: {risk}")
        
        # Create trading bot
        bot = ForexTradingBot(
            model_path=model_path,
            timeframe=mt5_timeframe,
            symbols=symbols,
            lot_size=lot_size,
            risk_per_trade=risk,
            stop_loss_atr_multiplier=2.0,
            take_profit_atr_multiplier=3.0,
            max_spread_pips=5,
            max_trades_per_symbol=1,
            max_total_trades=10
        )
        
        # Start trading (this will block until Ctrl+C or error)
        bot.start_trading()
        
        logger.info("=== Trading Execution Finished ===")
        return True
        
    except Exception as e:
        logger.error(f"Error in trading execution: {e}")
        logger.info("=== Trading Execution Failed ===")
        return False

def run_full_pipeline(symbols, timeframe, num_bars, future_period, n_trials, run_live=False):
    """Run the complete trading pipeline."""
    logger.info("=== Starting Full Trading Pipeline ===")
    
    # Setup directories
    setup_directories()
    
    # 1. Collect data
    collected_files = collect_data(symbols, timeframe, num_bars)
    if not collected_files:
        logger.error("Data collection failed, aborting pipeline")
        return False
    
    # 2. Process features
    processed_files = process_features(collected_files, future_period=future_period)
    if not processed_files:
        logger.error("Feature processing failed, aborting pipeline")
        return False
    
    # Process each symbol
    results = {}
    
    for processed_file in processed_files:
        symbol = os.path.basename(processed_file).split('_')[1]
        logger.info(f"Processing pipeline for {symbol}")
        
        # 3. Train model
        model_path = train_model(processed_file, n_trials=n_trials)
        if not model_path:
            logger.error(f"Model training failed for {symbol}, skipping to next symbol")
            continue
        
        # 4. Run backtest
        backtest_success = run_backtest(processed_file, model_path)
        if not backtest_success:
            logger.error(f"Backtesting failed for {symbol}, skipping to next symbol")
            continue
        
        results[symbol] = {
            "data_file": processed_file,
            "model_path": model_path,
            "backtest_complete": backtest_success
        }
    
    # 5. Run live trading if requested
    if run_live and results:
        # Use the first successful model for trading all symbols
        first_model = next(iter(results.values()))["model_path"]
        trading_success = run_trading(first_model, symbols, timeframe)
        logger.info(f"Live trading completed with status: {trading_success}")
    
    logger.info("=== Full Trading Pipeline Completed ===")
    return results

def main():
    """Main entry point for the trading system."""
    parser = argparse.ArgumentParser(description="Forex Trading Algorithm System")
    parser.add_argument("--action", type=str, choices=["full", "collect", "process", "train", "backtest", "trade"],
                      default="full", help="Action to perform")
    parser.add_argument("--symbols", type=str, nargs="+", default=["EURUSD", "GBPUSD", "USDJPY"],
                      help="Symbols to trade")
    parser.add_argument("--timeframe", type=str, default="M5", 
                      choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                      help="Timeframe for data")
    parser.add_argument("--bars", type=int, default=5000, help="Number of bars to collect")
    parser.add_argument("--future_period", type=int, default=2, help="Future period for target creation")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--live", action="store_true", help="Run live trading (use with caution)")
    parser.add_argument("--input_file", type=str, help="Input file for process/train/backtest actions")
    parser.add_argument("--model", type=str, help="Model path for backtest/trade actions")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Forex Trading System with action: {args.action}")
    
    # Setup directories
    setup_directories()
    
    try:
        if args.action == "full":
            run_full_pipeline(args.symbols, args.timeframe, args.bars, 
                            args.future_period, args.trials, args.live)
        
        elif args.action == "collect":
            collect_data(args.symbols, args.timeframe, args.bars)
        
        elif args.action == "process":
            if args.input_file:
                process_features([args.input_file], future_period=args.future_period)
            else:
                logger.error("Input file required for process action")
        
        elif args.action == "train":
            if args.input_file:
                train_model(args.input_file, n_trials=args.trials)
            else:
                logger.error("Input file required for train action")
        
        elif args.action == "backtest":
            if args.input_file and args.model:
                run_backtest(args.input_file, args.model)
            else:
                logger.error("Input file and model path required for backtest action")
        
        elif args.action == "trade":
            if args.model:
                run_trading(args.model, args.symbols, args.timeframe)
            else:
                logger.error("Model path required for trade action")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
