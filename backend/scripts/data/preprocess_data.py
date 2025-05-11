import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def categorize_source(source):
    """Simple function to categorize news sources"""
    tech_sources = ['TechCrunch', 'Wired', 'The Verge', 'Ars Technica', 'CNET']
    business_sources = ['Bloomberg', 'Forbes', 'Business Insider', 'Wall Street Journal', 'CNBC']
    
    if source in tech_sources:
        return 1
    elif source in business_sources:
        return 2
    else:
        return 0

def preprocess_data():
    """
    Preprocess raw news data:
    - Handle missing values
    - Convert date strings to datetime
    - Extract features from datetime using cyclical encoding
    - Extract additional features from text
    - Normalize numerical fields
    - Remove low variance features
    - Save the processed data
    """
    input_path = os.path.join('data', 'raw_data.csv')
    output_path = os.path.join('data', 'processed_data.csv')
    
    try:
        # Read the raw data
        logger.info(f"Reading raw data from {input_path}")
        df = pd.read_csv(input_path)
        
        if df.empty:
            logger.error("Raw data file is empty.")
            return False
        
        # Handle missing values
        logger.info("Handling missing values")
        df['author'] = df['author'].fillna('Unknown')
        df['source'] = df['source'].fillna('Unknown')
        df['content_length'] = df['content_length'].fillna(0)
        df['description_length'] = df['description_length'].fillna(0)
        
        # Convert published_at to datetime
        logger.info("Converting date strings to datetime")
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['published_at'])
        
        # Extract better features
        logger.info("Extracting additional features")
        
        # Basic text features
        df['title_length'] = df['title'].apply(len)
        df['has_author'] = df['author'].apply(lambda x: 0 if x == 'Unknown' else 1)
        df['source_category'] = df['source'].apply(categorize_source)
        
        # Time features using cyclical encoding (better than linear scaling for time)
        df['publish_hour'] = df['published_at'].dt.hour
        df['publish_day'] = df['published_at'].dt.day
        df['publish_month'] = df['published_at'].dt.month
        df['publish_weekday'] = df['published_at'].dt.weekday
        
        # Convert time features to cyclical representation
        df['hour_sin'] = np.sin(2 * np.pi * df['publish_hour']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['publish_hour']/24.0)
        df['day_sin'] = np.sin(2 * np.pi * df['publish_day']/31.0)
        df['day_cos'] = np.cos(2 * np.pi * df['publish_day']/31.0)
        df['month_sin'] = np.sin(2 * np.pi * df['publish_month']/12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['publish_month']/12.0)
        df['weekday_sin'] = np.sin(2 * np.pi * df['publish_weekday']/7.0)
        df['weekday_cos'] = np.cos(2 * np.pi * df['publish_weekday']/7.0)
        
        # Drop original time columns
        df = df.drop(['publish_hour', 'publish_day', 'publish_month', 'publish_weekday'], axis=1)
        
        # Check for and remove features with very low variance
        logger.info("Checking for low variance features")
        numerical_cols = ['content_length', 'description_length', 'title_length', 
                         'has_author', 'source_category', 
                         'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                         'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos']
        
        # Calculate variance for each feature
        variance = df[numerical_cols].var()
        low_variance_cols = variance[variance < 0.01].index.tolist()
        
        if low_variance_cols:
            logger.warning(f"Removing low variance columns: {low_variance_cols}")
            df = df.drop(low_variance_cols, axis=1)
            # Remove the dropped columns from our list
            numerical_cols = [col for col in numerical_cols if col not in low_variance_cols]
        
        # Normalize remaining numerical fields
        logger.info("Normalizing numerical fields")
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        # Add preprocessing timestamp
        df['processed_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Create content length categories
# Create content length categories
                # Create content length categories
        logger.info("Creating content length categories")
        # Define boundaries for short, medium, and long articles
        q25 = df['content_length'].quantile(0.33)
        q75 = df['content_length'].quantile(0.66)
        
        # Check if quantiles are unique
        if q25 == q75:
            # If quantiles are identical, use fixed bins
            logger.warning("Content length values are too similar for quantile binning. Using fixed bins instead.")
            df['content_length_category'] = pd.cut(
                df['content_length'], 
                bins=[0, 0.33, 0.66, 1.0], 
                labels=[0, 1, 2]
            )
        else:
            # Use quantile-based bins
            df['content_length_category'] = pd.cut(
                df['content_length'], 
                bins=[float('-inf'), q25, q75, float('inf')], 
                labels=[0, 1, 2],
                duplicates='drop'  # Handle duplicate bin edges
            )
                # Handle NaN values in content_length_category
        if 0 not in df['content_length_category'].cat.categories:
            df['content_length_category'] = df['content_length_category'].cat.add_categories([0])
        
        df['content_length_category'] = df['content_length_category'].fillna(0).astype(int)

        logger.info(f"Content length categories distribution: {df['content_length_category'].value_counts()}")
        # Save processed data
        logger.info(f"Saving processed data to {output_path}")
        df.to_csv(output_path, index=False)
        
        logger.info(f"Successfully preprocessed {len(df)} news articles")
        return True
        
    except FileNotFoundError:
        logger.error(f"Raw data file not found at {input_path}")
        return False
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return False


if __name__ == "__main__":
    preprocess_data()