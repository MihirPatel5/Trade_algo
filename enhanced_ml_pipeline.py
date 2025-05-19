import pandas as pd
import numpy as np
import os
import joblib
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global configurations
PROCESSED_DATA_PATH = "forex_preprocessed"
MODEL_OUTPUT_PATH = "forex_models"
FEATURE_IMPORTANCE_PLOT = "feature_importance.png"
CONFUSION_MATRIX_PLOT = "confusion_matrix.png"
ROC_CURVE_PLOT = "roc_curve.png"
PR_CURVE_PLOT = "precision_recall_curve.png"
MODEL_METADATA_FILE = "model_metadata.json"

# Ensure output directories exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

class MLModelTrainer:
    """Class for training and evaluating ML models for forex prediction."""
    
    def __init__(self, data_path=None, target_col="Target", test_size=0.2,
                 model_output_path=MODEL_OUTPUT_PATH, n_trials=50, cv_splits=5):
        """
        Initialize the ML model trainer.
        
        Args:
            data_path: Path to the processed features CSV file
            target_col: Name of the target column
            test_size: Proportion of data to use for testing
            model_output_path: Path to save model artifacts
            n_trials: Number of Optuna trials for hyperparameter optimization
            cv_splits: Number of cross-validation splits
        """
        self.data_path = data_path
        self.target_col = target_col
        self.test_size = test_size
        self.model_output_path = model_output_path
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.model_name = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_params = None
        self.feature_importances = None
        self.metrics = {}
        
        logger.info(f"Initialized ML model trainer with {n_trials} optimization trials")
    
    def load_data(self, data_path=None):
        """
        Load and prepare data for model training.
        
        Args:
            data_path: Path to the processed features CSV file
        
        Returns:
            DataFrame with loaded data
        """
        if data_path is not None:
            self.data_path = data_path
        
        if self.data_path is None:
            raise ValueError("Data path must be provided")
        
        logger.info(f"Loading data from {self.data_path}")
        data = pd.read_csv(self.data_path)
        
        if self.target_col not in data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
        
        # Handle datetime column if present
        if 'Datetime' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            
        # Check for NaN values
        if data.isnull().values.any():
            logger.warning("Data contains NaN values. Filling NaNs with appropriate values.")
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            data.fillna(method='ffill', inplace=True)
            data.dropna(inplace=True)
        
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    
    def prepare_train_test_split(self, data=None, time_based=True):
        """
        Prepare train/test split for model evaluation.
        
        Args:
            data: DataFrame with features and target
            time_based: Whether to use a time-based split (True) or random split (False)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if data is None:
            data = self.load_data()
        
        # Separate features and target
        exclude_cols = [self.target_col]
        
        # Also exclude datetime column and any non-feature columns
        for col in ['Datetime', 'future_price', 'future_pct_change']:
            if col in data.columns:
                exclude_cols.append(col)
        
        # Get feature names
        self.feature_names = [col for col in data.columns if col not in exclude_cols]
        
        X = data[self.feature_names]
        y = data[self.target_col]
        
        logger.info(f"Number of features: {len(self.feature_names)}")
        logger.info(f"Target distribution: \n{y.value_counts(normalize=True)}")
        
        # Time-based split (for time series data) or random split
        if time_based:
            logger.info("Using time-based train/test split")
            split_idx = int(len(data) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            from sklearn.model_selection import train_test_split
            logger.info("Using random train/test split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42, stratify=y
            )
        
        # Save for later use
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        
        logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Mean cross-validation F1 score
        """
        # Select model type
        model_type = trial.suggest_categorical("model_type", [
            "RandomForest", "XGBoost", "LightGBM", "CatBoost", 
            "LogisticRegression", "GradientBoosting"
        ])
        
        # Model-specific hyperparameters
        if model_type == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 25),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42
            }
            model = RandomForestClassifier(**params)
        
        elif model_type == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "random_state": 42,
                "eval_metric": "logloss",
                "use_label_encoder": False
            }
            model = xgb.XGBClassifier(**params)
        
        elif model_type == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "verbose": -1
            }
            model = lgb.LGBMClassifier(**params)
        
        elif model_type == "CatBoost":
            params = {
                "iterations": trial.suggest_int("iterations", 50, 500),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "random_strength": trial.suggest_float("random_strength", 0.1, 10),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
                "random_seed": 42,
                "verbose": False
            }
            model = cb.CatBoostClassifier(**params)
        
        elif model_type == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 0.001, 10.0, log=True),
                "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
                "max_iter": 5000,
                "random_state": 42
            }
            model = LogisticRegression(**params)
        
        elif model_type == "GradientBoosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42
            }
            model = GradientBoostingClassifier(**params)
        
        # TimeSeriesSplit for cross-validation
        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        f1_scores = []
        
        for train_idx, val_idx in cv.split(self.X_train):
            X_train_cv, X_val_cv = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_cv, y_val_cv = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_val_cv)
            
            # Use weighted F1 score for imbalanced data
            f1 = f1_score(y_val_cv, y_pred_cv, average='weighted')
            f1_scores.append(f1)
        
        mean_f1 = np.mean(f1_scores)
        return mean_f1
    
    def optimize_hyperparameters(self):
        """
        Run hyperparameter optimization using Optuna.
        
        Returns:
            Best model, best parameters
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        # Create and run the study
        study = optuna.create_study(direction="maximize", 
                                   pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        
        start_time = time.time()
        study.optimize(self.objective, n_trials=self.n_trials)
        end_time = time.time()
        
        # Get best parameters and model
        self.best_params = study.best_params
        model_type = self.best_params.pop("model_type")
        self.model_name = model_type
        
        logger.info(f"Best model: {model_type}")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Optimization completed in {(end_time - start_time)/60:.2f} minutes")
        
        # Create best model with optimized parameters
        if model_type == "RandomForest":
            self.best_model = RandomForestClassifier(**self.best_params)
        elif model_type == "XGBoost":
            self.best_model = xgb.XGBClassifier(**self.best_params, eval_metric="logloss", use_label_encoder=False)
        elif model_type == "LightGBM":
            self.best_model = lgb.LGBMClassifier(**self.best_params, verbose=-1)
        elif model_type == "CatBoost":
            self.best_model = cb.CatBoostClassifier(**self.best_params, verbose=False)
        elif model_type == "LogisticRegression":
            self.best_model = LogisticRegression(**self.best_params, max_iter=5000)
        elif model_type == "GradientBoosting":
            self.best_model = GradientBoostingClassifier(**self.best_params)
        
        # Save optimization history plots
        try:
            fig = plot_optimization_history(study)
            fig.write_image(os.path.join(self.model_output_path, "optimization_history.png"))
            
            fig = plot_param_importances(study)
            fig.write_image(os.path.join(self.model_output_path, "param_importances.png"))
        except Exception as e:
            logger.warning(f"Could not save optimization plots: {e}")
        
        return self.best_model, self.best_params
    
    def train_model(self, model=None):
        """
        Train the best model on the full training set.
        
        Args:
            model: Model to train (if None, uses best_model from optimization)
            
        Returns:
            Trained model
        """
        if model is not None:
            self.best_model = model
        
        if self.best_model is None:
            raise ValueError("No model specified and no best model available from optimization")
        
        logger.info(f"Training {type(self.best_model).__name__} on full training set...")
        
        start_time = time.time()
        self.best_model.fit(self.X_train, self.y_train)
        end_time = time.time()
        
        logger.info(f"Model training completed in {(end_time - start_time):.2f} seconds")
        
        return self.best_model
    
    def evaluate_model(self, model=None, X_test=None, y_test=None):
        """
        Evaluate model performance on test data.
        
        Args:
            model: Model to evaluate (if None, uses best_model)
            X_test: Test features (if None, uses self.X_test)
            y_test: Test targets (if None, uses self.y_test)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model is None:
            model = self.best_model
        
        if X_test is None:
            X_test = self.X_test
        
        if y_test is None:
            y_test = self.y_test
        
        if model is None or X_test is None or y_test is None:
            raise ValueError("Model, X_test, and y_test must be provided or available as instance variables")
        
        logger.info("Evaluating model on test data...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # For ROC and PR curves, we need probability predictions
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = None
        
        # Calculate metrics
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Add AUC if probabilities are available
        if y_pred_proba is not None:
            self.metrics["auc_roc"] = roc_auc_score(y_test, y_pred_proba)
            self.metrics["average_precision"] = average_precision_score(y_test, y_pred_proba)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Log metrics
        logger.info(f"Evaluation metrics:")
        for metric, value in self.metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {type(model).__name__}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_output_path, CONFUSION_MATRIX_PLOT))
        plt.close()
        
        # Plot ROC curve if probabilities are available
        if y_pred_proba is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"AUC = {self.metrics['auc_roc']:.4f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"ROC Curve - {type(model).__name__}")
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(self.model_output_path, ROC_CURVE_PLOT))
            plt.close()
            
            # Plot Precision-Recall curve
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.plot(recall, precision, label=f"AP = {self.metrics['average_precision']:.4f}")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f"Precision-Recall Curve - {type(model).__name__}")
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(self.model_output_path, PR_CURVE_PLOT))
            plt.close()
        
        # Log classification report
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        return self.metrics
    
    def extract_feature_importance(self, model=None):
        """
        Extract and visualize feature importances.
        
        Args:
            model: Model to extract feature importances from (if None, uses best_model)
            
        Returns:
            DataFrame with feature importances
        """
        if model is None:
            model = self.best_model
        
        if model is None:
            raise ValueError("Model must be provided or available as best_model")
        
        logger.info("Extracting feature importances...")
        
        # Extract feature importances based on model type
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            logger.warning("Model does not provide feature importances")
            return None
        
        # Create DataFrame of feature importances
        feature_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        self.feature_importances = feature_imp
        
        # Plot top N features
        top_n = min(20, len(feature_imp))
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Importance", y="Feature", data=feature_imp.head(top_n))
        plt.title(f"Top {top_n} Feature Importances - {type(model).__name__}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_output_path, FEATURE_IMPORTANCE_PLOT))
        plt.close()
        
        # Save feature importances to CSV
        feature_imp.to_csv(os.path.join(self.model_output_path, "feature_importances.csv"), index=False)
        
        logger.info(f"Top 10 features:\n{feature_imp.head(10)}")
        
        return feature_imp
    
    def save_model(self, model=None, model_filename="best_forex_model.pkl"):
        """
        Save the trained model and metadata.
        
        Args:
            model: Model to save (if None, uses best_model)
            model_filename: Filename for the saved model
            
        Returns:
            Path to saved model
        """
        if model is None:
            model = self.best_model
        
        if model is None:
            raise ValueError("Model must be provided or available as best_model")
        
        # Full path to save model
        model_path = os.path.join(self.model_output_path, model_filename)
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata (parameters, metrics, feature names)
        metadata = {
            "model_name": self.model_name,
            "parameters": self.best_params,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_path": self.data_path
        }
        
        metadata_path = os.path.join(self.model_output_path, MODEL_METADATA_FILE)
        pd.DataFrame([metadata]).to_json(metadata_path, orient="records")
        logger.info(f"Model metadata saved to {metadata_path}")
        
        return model_path
    
    def run_full_pipeline(self, data_path=None, model_filename="best_forex_model.pkl", time_based_split=True):
        """
        Run the full model training pipeline.
        
        Args:
            data_path: Path to processed features CSV
            model_filename: Filename for the saved model
            time_based_split: Whether to use time-based train/test split
            
        Returns:
            Path to saved model, evaluation metrics
        """
        logger.info("Starting full model training pipeline...")
        
        # Load and prepare data
        data = self.load_data(data_path)
        
        # Prepare train/test split
        self.prepare_train_test_split(data, time_based=time_based_split)
        
        # Optimize hyperparameters
        self.optimize_hyperparameters()
        
        # Train best model
        self.train_model()
        
        # Evaluate model
        metrics = self.evaluate_model()
        
        # Extract feature importances
        self.extract_feature_importance()
        
        # Save model and metadata
        model_path = self.save_model(model_filename=model_filename)
        
        logger.info("Model training pipeline completed successfully!")
        
        return model_path, metrics

def main():
    """Main function to run the model training pipeline."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train forex trading prediction model")
    parser.add_argument("--data", type=str, required=True, help="Path to processed features CSV")
    parser.add_argument("--output", type=str, default=MODEL_OUTPUT_PATH, help="Path to save model artifacts")
    parser.add_argument("--target", type=str, default="Target", help="Target column name")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data for testing")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of cross-validation splits")
    parser.add_argument("--time_based", action="store_true", help="Use time-based train/test split")
    parser.add_argument("--model_name", type=str, default="best_forex_model.pkl", help="Filename for the saved model")
    
    args = parser.parse_args()
    
    # Create trainer and run pipeline
    trainer = MLModelTrainer(
        data_path=args.data,
        target_col=args.target,
        test_size=args.test_size,
        model_output_path=args.output,
        n_trials=args.trials,
        cv_splits=args.cv_splits
    )
    
    model_path, metrics = trainer.run_full_pipeline(
        data_path=args.data,
        model_filename=args.model_name,
        time_based_split=args.time_based
    )
    
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Final metrics: {metrics}")

if __name__ == "__main__":
    main()
