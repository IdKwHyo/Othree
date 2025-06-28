"""
Feature Engineering Module

Handles feature creation, selection, and engineering for the crime analysis system.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from config import Config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature engineering operations"""
    
    def __init__(self, config=None):
        self.config = config or Config()
    
    def create_temporal_features(self, df):
        """
        Create temporal features from date information
        
        Args:
            df (pd.DataFrame): Data with date information
            
        Returns:
            pd.DataFrame: Data with temporal features
        """
        logger.info("Creating temporal features...")
        
        if 'Date' not in df.columns:
            logger.warning("No Date column found. Skipping temporal features.")
            return df
        
        # Basic temporal features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['week_of_year'] = df['Date'].dt.isocalendar().week
        df['month_of_year'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        
        # Cyclical encoding for periodic features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month_of_year'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_of_year'] / 12)
        
        # Season features
        df['season'] = df['month_of_year'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Holiday indicators (simplified)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        
        logger.info("Temporal features created")
        return df
    
    def create_spatial_features(self, df):
        """
        Create spatial features from location information
        
        Args:
            df (pd.DataFrame): Data with location information
            
        Returns:
            pd.DataFrame: Data with spatial features
        """
        logger.info("Creating spatial features...")
        
        # Community area features
        if 'Community Area' in df.columns:
            # Area density (if population data available)
            if 'population' in df.columns and 'area_km2' in df.columns:
                df['population_density'] = df['population'] / df['area_km2']
            
            # Area category based on size
            if 'area_km2' in df.columns:
                df['area_category'] = pd.cut(
                    df['area_km2'], 
                    bins=[0, 2, 3, 4, 10], 
                    labels=['small', 'medium', 'large', 'very_large']
                )
        
        # Coordinate-based features (if available)
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            # Distance from city center (approximate Chicago center)
            chicago_center_lat, chicago_center_lon = 41.8781, -87.6298
            df['distance_from_center'] = np.sqrt(
                (df['Latitude'] - chicago_center_lat)**2 + 
                (df['Longitude'] - chicago_center_lon)**2
            )
        
        logger.info("Spatial features created")
        return df
    
    def create_crime_type_features(self, df):
        """
        Create features based on crime type information
        
        Args:
            df (pd.DataFrame): Data with crime type information
            
        Returns:
            pd.DataFrame: Data with crime type features
        """
        logger.info("Creating crime type features...")
        
        if 'Primary Type' not in df.columns:
            logger.warning("No Primary Type column found. Skipping crime type features.")
            return df
        
        # Crime type encoding
        crime_type_counts = df['Primary Type'].value_counts()
        df['crime_type_frequency'] = df['Primary Type'].map(crime_type_counts)
        
        # Crime category (simplified)
        violent_crimes = ['HOMICIDE', 'CRIMINAL SEXUAL ASSAULT', 'ROBBERY', 'AGGRAVATED ASSAULT', 'BATTERY']
        property_crimes = ['BURGLARY', 'THEFT', 'CRIMINAL DAMAGE']
        
        df['crime_category'] = df['Primary Type'].apply(
            lambda x: 'violent' if x in violent_crimes else 
                     'property' if x in property_crimes else 'other'
        )
        
        # One-hot encoding for crime categories
        crime_categories = pd.get_dummies(df['crime_category'], prefix='crime_category')
        df = pd.concat([df, crime_categories], axis=1)
        
        logger.info("Crime type features created")
        return df
    
    def create_lag_features(self, df, group_cols=['Community Area'], target_col='incidents', lags=[1, 2, 3]):
        """
        Create lag features for time series analysis
        
        Args:
            df (pd.DataFrame): Time series data
            group_cols (list): Columns to group by
            target_col (str): Target column for lagging
            lags (list): Lag periods to create
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        logger.info("Creating lag features...")
        
        # Sort by time
        if 'Date' in df.columns:
            df = df.sort_values(['Community Area', 'Date'])
        
        # Create lag features
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
        
        # Create rolling statistics
        for window in [3, 7, 30]:
            df[f'{target_col}_rolling_mean_{window}'] = df.groupby(group_cols)[target_col].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df[f'{target_col}_rolling_std_{window}'] = df.groupby(group_cols)[target_col].rolling(
                window=window, min_periods=1
            ).std().reset_index(0, drop=True)
        
        logger.info("Lag features created")
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between different variables
        
        Args:
            df (pd.DataFrame): Data with features
            
        Returns:
            pd.DataFrame: Data with interaction features
        """
        logger.info("Creating interaction features...")
        
        # Time-space interactions
        if 'day_of_week' in df.columns and 'Community Area' in df.columns:
            df['area_day_interaction'] = df['Community Area'].astype(str) + '_' + df['day_of_week'].astype(str)
        
        # Season-crime type interactions
        if 'season' in df.columns and 'crime_category' in df.columns:
            df['season_crime_interaction'] = df['season'] + '_' + df['crime_category']
        
        # Population-crime rate interactions
        if 'population' in df.columns and 'incidents_per_1k' in df.columns:
            df['population_crime_interaction'] = df['population'] * df['incidents_per_1k'] / 1000
        
        logger.info("Interaction features created")
        return df
    
    def select_features(self, df, target_col='incidents', exclude_cols=None):
        """
        Select relevant features for model training
        
        Args:
            df (pd.DataFrame): Data with all features
            target_col (str): Target variable column
            exclude_cols (list): Columns to exclude
            
        Returns:
            tuple: (feature_columns, target_series)
        """
        logger.info("Selecting features for model training...")
        
        if exclude_cols is None:
            exclude_cols = []
        
        # Define columns to exclude
        exclude_cols.extend([
            'Date', 'rag_text', 'crime_category', 'season', 'area_category',
            'area_day_interaction', 'season_crime_interaction'
        ])
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        missing_counts = df[feature_cols].isnull().sum() / len(df)
        feature_cols = [col for col in feature_cols if missing_counts[col] < missing_threshold]
        
        # Remove constant columns
        constant_cols = []
        for col in feature_cols:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        feature_cols = [col for col in feature_cols if col not in constant_cols]
        
        logger.info(f"Selected {len(feature_cols)} features for model training")
        logger.info(f"Feature columns: {feature_cols}")
        
        return feature_cols, df[target_col] if target_col in df.columns else None
    
    def handle_missing_values(self, df, feature_cols):
        """
        Handle missing values in feature columns
        
        Args:
            df (pd.DataFrame): Data with features
            feature_cols (list): Feature columns to process
            
        Returns:
            pd.DataFrame: Data with handled missing values
        """
        logger.info("Handling missing values...")
        
        for col in feature_cols:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Numeric columns: fill with median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Categorical columns: fill with mode
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        logger.info("Missing values handled")
        return df
    
    def scale_features(self, df, feature_cols, scaler=None):
        """
        Scale numerical features
        
        Args:
            df (pd.DataFrame): Data with features
            feature_cols (list): Numerical feature columns
            scaler: Scaler object (if None, will create new one)
            
        Returns:
            tuple: (scaled_data, scaler)
        """
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Scaling numerical features...")
        
        if scaler is None:
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = scaler.transform(df[feature_cols])
        
        logger.info("Features scaled")
        return df, scaler
    
    def get_feature_importance(self, model, feature_cols):
        """
        Get feature importance from trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_cols (list): Feature column names
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Feature importance calculated")
            return importance_df
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None 