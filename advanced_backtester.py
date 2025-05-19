import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging
import json
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global configurations
PROCESSED_DATA_PATH = "forex_preprocessed"
MODEL_OUTPUT_PATH = "forex_models"
BACKTEST_OUTPUT_PATH = "forex_backtest_results"
os.makedirs(BACKTEST_OUTPUT_PATH, exist_ok=True)

class ForexBacktester:
    """Advanced backtesting framework for forex trading models."""
    
    def __init__(self, data_path, model_path, price_col='Close', 
                 target_col='Target', datetime_col='Datetime', 
                 commission=0.0001, slippage=0.0001):
        """
        Initialize the backtester.
        
        Args:
            data_path: Path to processed data CSV
            model_path: Path to trained model
            price_col: Column name for price data
            target_col: Column name for target variable
            datetime_col: Column name for datetime
            commission: Commission per trade (as decimal)
            slippage: Slippage per trade (as decimal)
        """
        self.data_path = data_path
        self.model_path = model_path
        self.price_col = price_col
        self.target_col = target_col
        self.datetime_col = datetime_col
        self.commission = commission
        self.slippage = slippage
        self.data = None
        self.model = None
        self.feature_names = None
        self.backtest_results = None
        self.metrics = {}
        
        logger.info(f"Initialized ForexBacktester with model: {model_path}")
    
    def load_data_and_model(self):
        """Load the processed data and trained model."""
        # Load data
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        
        # Convert datetime column if available
        if self.datetime_col in self.data.columns:
            self.data[self.datetime_col] = pd.to_datetime(self.data[self.datetime_col])
        
        logger.info(f"Data loaded. Shape: {self.data.shape}")
        
        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        
        # Try to load feature names from model metadata
        metadata_path = os.path.join(os.path.dirname(self.model_path), "model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.feature_names = metadata[0]['feature_names']
                logger.info(f"Loaded {len(self.feature_names)} feature names from metadata")
            except Exception as e:
                logger.warning(f"Could not load feature names from metadata: {e}")
        
        # If feature names not loaded, infer from data columns
        if self.feature_names is None:
            # Exclude known non-feature columns
            exclude_cols = [self.target_col, self.price_col, self.datetime_col]
            for col in ['future_price', 'future_pct_change']:
                if col in self.data.columns:
                    exclude_cols.append(col)
            self.feature_names = [col for col in self.data.columns if col not in exclude_cols]
            logger.info(f"Inferred {len(self.feature_names)} feature names from data columns")
        
        logger.info(f"Data and model loaded successfully")
        return self.data, self.model
    
    def standard_backtest(self):
        """
        Perform a standard backtest using the full dataset.
        
        Returns:
            DataFrame with backtest results
        """
        if self.data is None or self.model is None:
            self.load_data_and_model()
        
        logger.info("Starting standard backtest...")
        
        # Copy data to avoid modifying original
        data = self.data.copy()
        
        # Ensure data is sorted by datetime
        if self.datetime_col in data.columns:
            data = data.sort_values(by=self.datetime_col)
        
        # Get features for prediction
        X = data[self.feature_names]
        
        # Make predictions
        try:
            y_pred = self.model.predict(X)
            data['prediction'] = y_pred
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X)
                # For binary classification
                if y_prob.shape[1] == 2:
                    data['prediction_probability'] = y_prob[:, 1]
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
        
        # Calculate trade signals (1 = buy, 0 = hold/sell)
        data['signal'] = data['prediction']
        
        # Calculate returns
        # 1. Price change for next period
        data['next_price'] = data[self.price_col].shift(-1)
        data['price_change'] = (data['next_price'] - data[self.price_col]) / data[self.price_col]
        
        # 2. Strategy returns (apply signal to price change)
        data['strategy_return'] = data['signal'] * data['price_change']
        
        # 3. Apply costs (commission and slippage)
        # Only apply costs when signal changes (trade is executed)
        data['signal_change'] = data['signal'].diff().fillna(0) != 0
        data.loc[data['signal_change'], 'strategy_return'] -= (self.commission + self.slippage)
        
        # 4. Cumulative returns
        data['cumulative_price_change'] = (1 + data['price_change']).cumprod() - 1
        data['cumulative_strategy_return'] = (1 + data['strategy_return']).cumprod() - 1
        
        # Calculate metrics
        self.calculate_performance_metrics(data)
        
        # Save results
        self.backtest_results = data
        self.save_backtest_results()
        
        logger.info("Standard backtest completed")
        return data
    
    def walk_forward_backtest(self, initial_train_size=0.5, retrain_frequency=20):
        """
        Perform a walk-forward backtest with periodic model retraining.
        
        Args:
            initial_train_size: Fraction of data to use for initial training
            retrain_frequency: Retrain model every N samples
            
        Returns:
            DataFrame with backtest results
        """
        if self.data is None:
            self.load_data_and_model()
        
        logger.info(f"Starting walk-forward backtest with retraining every {retrain_frequency} periods...")
        
        # Copy data to avoid modifying original
        data = self.data.copy()
        
        # Ensure data is sorted by datetime
        if self.datetime_col in data.columns:
            data = data.sort_values(by=self.datetime_col)
        
        # Initialize arrays for predictions and actual target
        y_true = data[self.target_col].values
        y_pred = np.zeros_like(y_true)
        
        # Get features and target
        X = data[self.feature_names]
        y = data[self.target_col]
        
        # Initial training set size
        initial_train_idx = int(len(data) * initial_train_size)
        
        # Perform walk-forward backtest
        for i in range(initial_train_idx, len(data), retrain_frequency):
            # Training indices: all data up to current point
            train_indices = list(range(0, i))
            
            # Test indices: next retrain_frequency points (or remaining data)
            test_indices = list(range(i, min(i + retrain_frequency, len(data))))
            
            if not test_indices:
                break
            
            # Create train/test sets
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            
            # Clone and retrain model
            model_clone = joblib.load(self.model_path)  # Load fresh model
            model_clone.fit(X_train, y_train)
            
            # Make predictions
            try:
                predictions = model_clone.predict(X_test)
                y_pred[test_indices] = predictions
                
                logger.info(f"Walk-forward step {i//retrain_frequency+1}: trained on {len(X_train)} samples, "
                          f"predicted {len(test_indices)} samples")
            except Exception as e:
                logger.error(f"Error in walk-forward step at index {i}: {e}")
                continue
        
        # Add predictions to data
        data['prediction'] = y_pred
        
        # Calculate trade signals (1 = buy, 0 = hold/sell)
        data['signal'] = data['prediction']
        
        # Calculate returns (same as in standard backtest)
        data['next_price'] = data[self.price_col].shift(-1)
        data['price_change'] = (data['next_price'] - data[self.price_col]) / data[self.price_col]
        data['strategy_return'] = data['signal'] * data['price_change']
        
        # Apply costs (commission and slippage)
        data['signal_change'] = data['signal'].diff().fillna(0) != 0
        data.loc[data['signal_change'], 'strategy_return'] -= (self.commission + self.slippage)
        
        # Cumulative returns
        data['cumulative_price_change'] = (1 + data['price_change']).cumprod() - 1
        data['cumulative_strategy_return'] = (1 + data['strategy_return']).cumprod() - 1
        
        # Skip initial training period for final metrics
        test_data = data.iloc[initial_train_idx:]
        self.calculate_performance_metrics(test_data)
        
        # Save results
        self.backtest_results = data
        self.save_backtest_results("walk_forward_backtest_results.csv")
        
        logger.info("Walk-forward backtest completed")
        return data
    
    def monte_carlo_backtest(self, n_simulations=100, window_fraction=0.5, random_seed=42):
        """
        Perform Monte Carlo backtesting by sampling random windows of data.
        
        Args:
            n_simulations: Number of Monte Carlo simulations to run
            window_fraction: Fraction of data to use in each simulation
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with Monte Carlo simulation results
        """
        if self.data is None or self.model is None:
            self.load_data_and_model()
        
        logger.info(f"Starting Monte Carlo backtest with {n_simulations} simulations...")
        
        # Copy data to avoid modifying original
        data = self.data.copy()
        
        # Ensure data is sorted by datetime
        if self.datetime_col in data.columns:
            data = data.sort_values(by=self.datetime_col)
        
        # Get features for prediction
        X = data[self.feature_names]
        y = data[self.target_col]
        
        # Initialize random number generator
        np.random.seed(random_seed)
        
        # Initialize arrays for simulation results
        simulation_returns = []
        simulation_metrics = []
        
        # Run simulations
        for sim in tqdm(range(n_simulations), desc="Monte Carlo Simulations"):
            # Select random window
            window_size = int(len(data) * window_fraction)
            start_idx = np.random.randint(0, len(data) - window_size)
            end_idx = start_idx + window_size
            
            # Extract window data
            window_data = data.iloc[start_idx:end_idx].copy()
            window_X = X.iloc[start_idx:end_idx]
            window_y = y.iloc[start_idx:end_idx]
            
            # Make predictions
            try:
                window_data['prediction'] = self.model.predict(window_X)
            except Exception as e:
                logger.error(f"Error in simulation {sim}: {e}")
                continue
            
            # Calculate trade signals and returns
            window_data['signal'] = window_data['prediction']
            window_data['next_price'] = window_data[self.price_col].shift(-1)
            window_data['price_change'] = (window_data['next_price'] - window_data[self.price_col]) / window_data[self.price_col]
            window_data['strategy_return'] = window_data['signal'] * window_data['price_change']
            
            # Apply costs (commission and slippage)
            window_data['signal_change'] = window_data['signal'].diff().fillna(0) != 0
            window_data.loc[window_data['signal_change'], 'strategy_return'] -= (self.commission + self.slippage)
            
            # Cumulative returns
            window_data['cumulative_strategy_return'] = (1 + window_data['strategy_return']).cumprod() - 1
            
            # Extract final return
            final_return = window_data['cumulative_strategy_return'].iloc[-1]
            simulation_returns.append(final_return)
            
            # Calculate metrics
            metrics = {
                'final_return': final_return,
                'sharpe': self.calculate_sharpe_ratio(window_data['strategy_return']),
                'max_drawdown': self.calculate_max_drawdown(window_data['cumulative_strategy_return']),
                'win_rate': (window_data['strategy_return'] > 0).mean(),
                'accuracy': accuracy_score(window_y, window_data['prediction'])
            }
            simulation_metrics.append(metrics)
        
        # Compile results
        mc_results = {
            'returns': simulation_returns,
            'metrics': simulation_metrics,
            'mean_return': np.mean(simulation_returns),
            'std_return': np.std(simulation_returns),
            'median_return': np.median(simulation_returns),
            '5th_percentile': np.percentile(simulation_returns, 5),
            '95th_percentile': np.percentile(simulation_returns, 95)
        }
        
        # Plot histogram of returns
        plt.figure(figsize=(12, 6))
        sns.histplot(simulation_returns, kde=True)
        plt.axvline(mc_results['mean_return'], color='r', linestyle='--', label=f"Mean: {mc_results['mean_return']:.2%}")
        plt.axvline(mc_results['5th_percentile'], color='g', linestyle='--', label=f"5th Percentile: {mc_results['5th_percentile']:.2%}")
        plt.axvline(mc_results['95th_percentile'], color='g', linestyle='--', label=f"95th Percentile: {mc_results['95th_percentile']:.2%}")
        plt.title("Monte Carlo Simulation - Distribution of Returns")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(BACKTEST_OUTPUT_PATH, "monte_carlo_returns.png"))
        plt.close()
        
        # Save MC results
        pd.DataFrame(simulation_metrics).to_csv(os.path.join(BACKTEST_OUTPUT_PATH, "monte_carlo_metrics.csv"), index=False)
        
        logger.info(f"Monte Carlo backtesting completed. Mean return: {mc_results['mean_return']:.2%}")
        
        return mc_results
    
    def calculate_performance_metrics(self, backtest_data):
        """
        Calculate performance metrics from backtest results.
        
        Args:
            backtest_data: DataFrame with backtest results
            
        Returns:
            Dictionary with performance metrics
        """
        # Trading metrics
        strategy_returns = backtest_data['strategy_return'].dropna()
        price_changes = backtest_data['price_change'].dropna()
        
        # Trading performance
        self.metrics = {
            # Returns
            'total_return': (1 + strategy_returns).prod() - 1,
            'annualized_return': self.calculate_annualized_return(strategy_returns),
            'benchmark_return': (1 + price_changes).prod() - 1,
            
            # Risk metrics
            'volatility': strategy_returns.std() * np.sqrt(252),  # Annualized
            'sharpe_ratio': self.calculate_sharpe_ratio(strategy_returns),
            'max_drawdown': self.calculate_max_drawdown(backtest_data['cumulative_strategy_return']),
            
            # Trade metrics
            'num_trades': backtest_data['signal_change'].sum(),
            'win_rate': (strategy_returns > 0).mean(),
            'avg_win': strategy_returns[strategy_returns > 0].mean() if any(strategy_returns > 0) else 0,
            'avg_loss': strategy_returns[strategy_returns < 0].mean() if any(strategy_returns < 0) else 0,
            'profit_factor': abs(strategy_returns[strategy_returns > 0].sum() / 
                               strategy_returns[strategy_returns < 0].sum()) if any(strategy_returns < 0) else float('inf'),
            
            # Model performance
            'accuracy': accuracy_score(backtest_data[self.target_col], backtest_data['prediction']),
            'precision': precision_score(backtest_data[self.target_col], backtest_data['prediction'], zero_division=0),
            'recall': recall_score(backtest_data[self.target_col], backtest_data['prediction'], zero_division=0),
            'f1': f1_score(backtest_data[self.target_col], backtest_data['prediction'], zero_division=0),
        }
        
        # Log metrics
        logger.info("Backtest Performance Metrics:")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        # Create confusion matrix
        cm = confusion_matrix(backtest_data[self.target_col], backtest_data['prediction'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Model Predictions Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(BACKTEST_OUTPUT_PATH, "backtest_confusion_matrix.png"))
        plt.close()
        
        return self.metrics
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio from returns."""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess_returns = returns - risk_free_rate
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    
    def calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown from equity curve."""
        equity_curve = equity_curve.dropna()
        if len(equity_curve) == 0:
            return 0
        
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Return maximum drawdown
        return drawdown.min()
    
    def calculate_annualized_return(self, returns):
        """Calculate annualized return from period returns."""
        if len(returns) == 0:
            return 0
        
        total_return = (1 + returns).prod() - 1
        # Assuming daily returns
        trading_days_per_year = 252
        years = len(returns) / trading_days_per_year
        
        return (1 + total_return) ** (1 / years) - 1
    
    def plot_equity_curve(self, backtest_data=None, output_file="equity_curve.png"):
        """
        Plot equity curve from backtest results.
        
        Args:
            backtest_data: DataFrame with backtest results
            output_file: Filename for saved plot
            
        Returns:
            Path to saved plot
        """
        if backtest_data is None:
            backtest_data = self.backtest_results
        
        if backtest_data is None:
            logger.warning("No backtest results available for plotting")
            return None
        
        plt.figure(figsize=(12, 6))
        
        # Plot equity curves
        plt.plot(backtest_data['cumulative_price_change'], label='Buy & Hold', color='blue', alpha=0.5)
        plt.plot(backtest_data['cumulative_strategy_return'], label='Strategy', color='green')
        
        # Add labels and title
        plt.title("Equity Curve - Strategy vs. Buy & Hold")
        plt.xlabel("Trade #")
        plt.ylabel("Return (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = os.path.join(BACKTEST_OUTPUT_PATH, output_file)
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Equity curve saved to {output_path}")
        return output_path
    
    def save_backtest_results(self, filename="backtest_results.csv"):
        """
        Save backtest results to CSV.
        
        Args:
            filename: Filename for saved results
            
        Returns:
            Path to saved file
        """
        if self.backtest_results is None:
            logger.warning("No backtest results available to save")
            return None
        
        output_path = os.path.join(BACKTEST_OUTPUT_PATH, filename)
        self.backtest_results.to_csv(output_path, index=False)
        
        # Save metrics
        metrics_path = os.path.join(BACKTEST_OUTPUT_PATH, "backtest_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        logger.info(f"Backtest results saved to {output_path}")
        logger.info(f"Backtest metrics saved to {metrics_path}")
        
        return output_path

def main():
    """Main function to run the backtester."""
    import argparse
    parser = argparse.ArgumentParser(description="Backtest forex trading strategy")
    parser.add_argument("--data", type=str, required=True, help="Path to processed features CSV")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--price_col", type=str, default="Close", help="Price column name")
    parser.add_argument("--target_col", type=str, default="Target", help="Target column name")
    parser.add_argument("--datetime_col", type=str, default="Datetime", help="Datetime column name")
    parser.add_argument("--commission", type=float, default=0.0001, help="Commission per trade")
    parser.add_argument("--slippage", type=float, default=0.0001, help="Slippage per trade")
    parser.add_argument("--mode", type=str, default="standard", 
                        choices=["standard", "walk_forward", "monte_carlo"],
                        help="Backtest mode")
    args = parser.parse_args()
    
    # Create backtester
    backtester = ForexBacktester(
        data_path=args.data,
        model_path=args.model,
        price_col=args.price_col,
        target_col=args.target_col,
        datetime_col=args.datetime_col,
        commission=args.commission,
        slippage=args.slippage
    )
    
    # Run backtest based on mode
    if args.mode == "standard":
        results = backtester.standard_backtest()
    elif args.mode == "walk_forward":
        results = backtester.walk_forward_backtest()
    elif args.mode == "monte_carlo":
        results = backtester.monte_carlo_backtest()
    
    # Plot equity curve
    backtester.plot_equity_curve()
    
    logger.info("Backtesting completed!")

if __name__ == "__main__":
    main()
