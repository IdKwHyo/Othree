import os
from datetime import datetime

class Config:
    """Configuration settings for the Crime Hotspot Detection system"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'chicago-crime-detection-secret-key-2024'
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # Data Sources
    CHICAGO_CRIME_URL = "https://data.cityofchicago.org/Public-Safety/Crimes-2025/t7ek-mgzi/data_preview"
    CHICAGO_COMMUNITY_AREAS_URL = "https://data.cityofchicago.org/resource/igwz-8jzy.json"
    
    # File Paths
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    PROPHET_MODELS_DIR = os.path.join(MODELS_DIR, 'prophet_models')
    
    # Data Files
    RAW_DATA_FILE = os.path.join(DATA_DIR, 'scraped_chicago_crime_sample.csv')
    MONTHLY_DATA_FILE = os.path.join(DATA_DIR, 'monthly_crime_rates.csv')
    WEEKLY_DATA_FILE = os.path.join(DATA_DIR, 'weekly_crime_rates.csv')
    TOP_AREAS_FILE = os.path.join(DATA_DIR, 'top_community_areas.csv')
    
    # Web Scraping Configuration
    SCRAPING_TIMEOUT = 30
    MAX_ROWS_TO_SCRAPE = 100000
    PAGE_LOAD_WAIT = 2
    
    # XGBoost Model Parameters
    XGBOOST_PARAMS = {
        'booster': 'gbtree',
        'nthread': 4,
        'verbosity': 1,
        'seed': 42,
        'max_depth': 8,
        'min_child_weight': 6,
        'gamma': 0.1,
        'max_delta_step': 1,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'subsample': 0.9,
        'reg_lambda': 1.5,
        'reg_alpha': 0.2,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'early_stopping_rounds': 200,
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'missing': None
    }
    
    # Prophet Model Configuration
    PROPHET_PARAMS = {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        'seasonality_mode': 'additive'
    }
    
    # Feature Engineering
    SEVERITY_MAP = {
        'HOMICIDE': 10,
        'CRIMINAL SEXUAL ASSAULT': 9,
        'ROBBERY': 8,
        'AGGRAVATED ASSAULT': 7,
        'BURGLARY': 6,
        'THEFT': 5,
        'BATTERY': 4,
        'NARCOTICS': 3,
        'CRIMINAL DAMAGE': 2,
        'OTHER OFFENSE': 1
    }
    
    # Community Areas Data (Fallback)
    CHICAGO_COMMUNITY_AREAS = {
        'total_areas': 77,
        'population_range': (50000, 150000),
        'area_range': (2.0, 5.0)
    }
    
    # Model Training
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TIME_SERIES_SPLITS = 5
    
    # Visualization
    CHART_COLORS = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8'
    }
    
    # API Configuration
    API_TIMEOUT = 30
    MAX_PREDICTION_DAYS = 180
    
    @staticmethod
    def ensure_directories():
        """Create necessary directories if they don't exist"""
        directories = [
            Config.DATA_DIR,
            Config.MODELS_DIR,
            Config.PROPHET_MODELS_DIR
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 