"""
Model Training Module

Handles training of XGBoost and Prophet models for crime prediction.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from prophet import Prophet
from config import Config

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training operations"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.xgb_model = None
        self.prophet_models = {}
        self.feature_cols = None
        self.scaler = None
    
    def train_xgboost_model(self, X, y, feature_cols):
        """
        Train XGBoost model for crime prediction
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target variable
            feature_cols (list): Feature column names
            
        Returns:
            XGBRegressor: Trained XGBoost model
        """
        logger.info("Training XGBoost model...")
        
        # Prepare data
        X_clean = X[feature_cols].dropna()
        y_clean = y.loc[X_clean.index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, 
            test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE,
            shuffle=False  # Maintain temporal order
        )
        
        # Update base_score in parameters
        params = self.config.XGBOOST_PARAMS.copy()
        params['base_score'] = y_train.mean()
        params['missing'] = np.nan
        
        # Train model
        self.xgb_model = XGBRegressor(**params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=self.config.XGBOOST_PARAMS['early_stopping_rounds'],
            verbose=True
        )
        
        # Evaluate model
        y_pred = self.xgb_model.predict(X_test)
        self._evaluate_model(y_test, y_pred, "XGBoost")
        
        self.feature_cols = feature_cols
        logger.info("XGBoost model training completed")
        return self.xgb_model
    
    def train_prophet_models(self, df, community_areas=None):
        """
        Train Prophet models for time series forecasting
        
        Args:
            df (pd.DataFrame): Time series data
            community_areas (list): List of community areas to model
            
        Returns:
            dict: Dictionary of trained Prophet models
        """
        logger.info("Training Prophet models...")
        
        if community_areas is None:
            community_areas = df['Community Area'].unique()
        
        for community_area in community_areas:
            if pd.isna(community_area):
                continue
            
            try:
                # Get data for this community area
                area_data = df[df['Community Area'] == community_area].copy()
                
                if len(area_data) < 10:
                    logger.warning(f"Insufficient data for Community Area {community_area}")
                    continue
                
                # Prepare data for Prophet
                area_data = area_data.groupby('Date').size().reset_index(name='incidents')
                area_data.columns = ['ds', 'y']  # Prophet requires 'ds' for dates, 'y' for values
                
                # Initialize Prophet model
                model = Prophet(**self.config.PROPHET_PARAMS)
                
                # Add custom seasonality
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
                
                # Fit model
                model.fit(area_data)
                
                # Store model
                self.prophet_models[community_area] = {
                    'model': model,
                    'data': area_data
                }
                
                logger.info(f"Prophet model fitted for Community Area {community_area}")
                
            except Exception as e:
                logger.error(f"Error fitting Prophet for Community Area {community_area}: {e}")
                continue
        
        logger.info(f"Prophet model training completed. {len(self.prophet_models)} models created")
        return self.prophet_models
    
    def _evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance"""
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        logger.info(f"{model_name} Model Performance:")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"R² Score: {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def predict_xgboost(self, X):
        """Make predictions using XGBoost model"""
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained")
        
        X_clean = X[self.feature_cols].dropna()
        predictions = self.xgb_model.predict(X_clean)
        
        return predictions
    
    def predict_prophet(self, community_area, periods=180):
        """
        Make predictions using Prophet model
        
        Args:
            community_area (int): Community area number
            periods (int): Number of periods to forecast
            
        Returns:
            pd.DataFrame: Forecast results
        """
        if community_area not in self.prophet_models:
            raise ValueError(f"No Prophet model for Community Area {community_area}")
        
        model_data = self.prophet_models[community_area]
        model = model_data['model']
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        return forecast
    
    def save_models(self, save_dir=None):
        """Save trained models to disk"""
        if save_dir is None:
            save_dir = self.config.MODELS_DIR
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save XGBoost model
        if self.xgb_model is not None:
            xgb_path = os.path.join(save_dir, 'xgboost_model.pkl')
            with open(xgb_path, 'wb') as f:
                pickle.dump({
                    'model': self.xgb_model,
                    'feature_cols': self.feature_cols,
                    'config': self.config.XGBOOST_PARAMS
                }, f)
            logger.info(f"XGBoost model saved to {xgb_path}")
        
        # Save Prophet models
        if self.prophet_models:
            prophet_dir = os.path.join(save_dir, 'prophet_models')
            os.makedirs(prophet_dir, exist_ok=True)
            
            for area, model_data in self.prophet_models.items():
                prophet_path = os.path.join(prophet_dir, f'prophet_model_area_{area}.pkl')
                with open(prophet_path, 'wb') as f:
                    pickle.dump(model_data, f)
            
            logger.info(f"Prophet models saved to {prophet_dir}")
    
    def load_models(self, load_dir=None):
        """Load trained models from disk"""
        if load_dir is None:
            load_dir = self.config.MODELS_DIR
        
        # Load XGBoost model
        xgb_path = os.path.join(load_dir, 'xgboost_model.pkl')
        if os.path.exists(xgb_path):
            with open(xgb_path, 'rb') as f:
                model_data = pickle.load(f)
                self.xgb_model = model_data['model']
                self.feature_cols = model_data['feature_cols']
            logger.info(f"XGBoost model loaded from {xgb_path}")
        
        # Load Prophet models
        prophet_dir = os.path.join(load_dir, 'prophet_models')
        if os.path.exists(prophet_dir):
            for filename in os.listdir(prophet_dir):
                if filename.startswith('prophet_model_area_') and filename.endswith('.pkl'):
                    area = int(filename.split('_')[-1].split('.')[0])
                    prophet_path = os.path.join(prophet_dir, filename)
                    
                    with open(prophet_path, 'rb') as f:
                        self.prophet_models[area] = pickle.load(f)
            
            logger.info(f"Prophet models loaded from {prophet_dir}")
    
    def get_feature_importance(self):
        """Get feature importance from XGBoost model"""
        if self.xgb_model is None or self.feature_cols is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df 