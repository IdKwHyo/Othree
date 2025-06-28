"""
Data Preprocessing Module

Handles data cleaning, validation, and preparation for the crime analysis system.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing operations"""
    
    def __init__(self, config=None):
        self.config = config or Config()
    
    def clean_crime_data(self, df):
        """
        Clean and preprocess crime data
        
        Args:
            df (pd.DataFrame): Raw crime data
            
        Returns:
            pd.DataFrame: Cleaned crime data
        """
        logger.info("Starting data cleaning process...")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Clean community area data
        df = self._clean_community_areas(df)
        
        # Clean coordinate data
        df = self._clean_coordinates(df)
        
        # Clean and parse date data
        df = self._clean_dates(df)
        
        # Add severity mapping
        df = self._add_severity_mapping(df)
        
        logger.info(f"Data cleaning completed. Final dataset: {len(df)} rows")
        return df
    
    def _clean_community_areas(self, df):
        """Clean community area data"""
        if 'Community Area' in df.columns:
            # Remove empty community areas
            df = df[df['Community Area'] != '']
            
            # Convert to numeric
            df['Community Area'] = pd.to_numeric(df['Community Area'], errors='coerce')
            
            # Remove null values
            df = df[df['Community Area'].notnull()]
            
            # Convert to integer
            df['Community Area'] = df['Community Area'].astype(int)
            
            logger.info(f"Cleaned community areas. Valid areas: {df['Community Area'].nunique()}")
        
        return df
    
    def _clean_coordinates(self, df):
        """Clean coordinate data"""
        for col in ['Latitude', 'Longitude']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"Cleaned {col} coordinates")
        
        return df
    
    def _clean_dates(self, df):
        """Clean and parse date data"""
        if 'Date' in df.columns:
            # Convert to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Remove rows with invalid dates
            df = df[df['Date'].notnull()]
            
            # Extract date components
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day
            df['hour'] = df['Date'].dt.hour
            df['minute'] = df['Date'].dt.minute
            df['second'] = df['Date'].dt.second
            df['week'] = df['Date'].dt.isocalendar().week
            
            logger.info(f"Cleaned dates. Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    
    def _add_severity_mapping(self, df):
        """Add severity mapping for crime types"""
        if 'Primary Type' in df.columns:
            df['severity'] = df['Primary Type'].map(
                lambda x: self.config.SEVERITY_MAP.get(x, 0)
            )
            logger.info("Added severity mapping")
        else:
            df['severity'] = 0
        
        return df
    
    def aggregate_data(self, df):
        """
        Aggregate crime data by community area and time periods
        
        Args:
            df (pd.DataFrame): Cleaned crime data
            
        Returns:
            tuple: (monthly_data, weekly_data)
        """
        logger.info("Starting data aggregation...")
        
        # Monthly aggregation
        monthly = df.groupby(['Community Area', 'year', 'month']).size().reset_index(name='incidents')
        logger.info(f"Created monthly aggregation: {len(monthly)} records")
        
        # Weekly aggregation
        weekly = df.groupby(['Community Area', 'year', 'week']).size().reset_index(name='incidents')
        logger.info(f"Created weekly aggregation: {len(weekly)} records")
        
        return monthly, weekly
    
    def normalize_data(self, monthly_data, weekly_data, community_data):
        """
        Normalize crime data using population and area information
        
        Args:
            monthly_data (pd.DataFrame): Monthly aggregated data
            weekly_data (pd.DataFrame): Weekly aggregated data
            community_data (pd.DataFrame): Community area data
            
        Returns:
            tuple: (normalized_monthly, normalized_weekly)
        """
        logger.info("Starting data normalization...")
        
        # Merge community data
        monthly = monthly_data.merge(community_data, on='Community Area', how='left')
        weekly = weekly_data.merge(community_data, on='Community Area', how='left')
        
        # Calculate normalized metrics
        monthly['incidents_per_1k'] = monthly['incidents'] / monthly['population'] * 1000
        monthly['incidents_per_km2'] = monthly['incidents'] / monthly['area_km2']
        weekly['incidents_per_1k'] = weekly['incidents'] / weekly['population'] * 1000
        weekly['incidents_per_km2'] = weekly['incidents'] / weekly['area_km2']
        
        # Clean normalized data
        monthly = self._clean_normalized_data(monthly)
        weekly = self._clean_normalized_data(weekly)
        
        logger.info("Data normalization completed")
        return monthly, weekly
    
    def _clean_normalized_data(self, df):
        """Clean normalized data by handling outliers and invalid values"""
        # Replace infinite values with NaN
        df['incidents_per_1k'] = df['incidents_per_1k'].replace([np.inf, -np.inf], np.nan)
        df['incidents_per_km2'] = df['incidents_per_km2'].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with 0
        df['incidents_per_1k'] = df['incidents_per_1k'].fillna(0)
        df['incidents_per_km2'] = df['incidents_per_km2'].fillna(0)
        
        # Remove outliers (values beyond 3 standard deviations)
        for col in ['incidents_per_1k', 'incidents_per_km2']:
            mean_val = df[col].mean()
            std_val = df[col].std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def add_zscore_thresholding(self, monthly_data):
        """
        Add z-score thresholding to identify high-risk areas
        
        Args:
            monthly_data (pd.DataFrame): Monthly normalized data
            
        Returns:
            pd.DataFrame: Data with z-score and threshold indicators
        """
        logger.info("Adding z-score thresholding...")
        
        # Calculate z-score for incidents per 1k population
        monthly_data['zscore'] = monthly_data.groupby(['year', 'month'])['incidents_per_1k'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        # Identify areas above threshold (z-score > 2)
        monthly_data['above_threshold'] = monthly_data['zscore'] > 2
        
        # Identify top community areas
        top_areas = monthly_data[monthly_data['above_threshold']].copy()
        
        logger.info(f"Z-score thresholding completed. Found {len(top_areas)} high-risk records")
        
        return monthly_data, top_areas
    
    def create_rag_text(self, df):
        """
        Create text field for RAG (Retrieval-Augmented Generation) preprocessing
        
        Args:
            df (pd.DataFrame): Crime data
            
        Returns:
            pd.DataFrame: Data with RAG text field
        """
        logger.info("Creating RAG text field...")
        
        df['rag_text'] = df.apply(
            lambda row: f"{row.get('Date','')} {row.get('Primary Type','')} "
                       f"{row.get('Location Description','')} {row.get('Community Area','')}", 
            axis=1
        )
        
        logger.info("RAG text field created")
        return df
    
    def validate_data(self, df, data_type="crime"):
        """
        Validate data quality and completeness
        
        Args:
            df (pd.DataFrame): Data to validate
            data_type (str): Type of data for validation
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            'total_rows': len(df),
            'null_counts': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        if data_type == "crime":
            validation_results.update({
                'date_range': {
                    'start': df['Date'].min().strftime('%Y-%m-%d') if 'Date' in df.columns else None,
                    'end': df['Date'].max().strftime('%Y-%m-%d') if 'Date' in df.columns else None
                },
                'unique_areas': df['Community Area'].nunique() if 'Community Area' in df.columns else 0,
                'unique_crime_types': df['Primary Type'].nunique() if 'Primary Type' in df.columns else 0
            })
        
        logger.info(f"Data validation completed for {data_type} data")
        return validation_results 