import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_news_data():
    """
    Collect news data from News API for the last 5 days
    and save it to a CSV file.
    """
    # Get the API key from environment variables
    api_key = os.getenv('NEWS_API_KEY')
    
    if not api_key:
        logger.error("API key not found. Please set the NEWS_API_KEY environment variable.")
        return False
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    # Format dates for the API
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Endpoint URL
    url = os.getenv('NEWS_API_ENDPOINT', 'https://newsapi.org/v2/everything')
    
    # Parameters for the API request
    params = {
    'q': 'technology OR business OR science OR health',
    'from': from_date,
    'to': to_date,
    'sortBy': 'publishedAt',
    'language': 'en',
    'apiKey': api_key,
    'pageSize': 100  # Get more articles per request
    }
    try:
        # Make the request
        logger.info(f"Fetching news data from {from_date} to {to_date}")
        all_articles = []
        for i in range(5):  # Get data from 5 different days
            # Adjust dates for each request
            adjusted_from = (start_date - timedelta(days=i*2)).strftime('%Y-%m-%d')
            adjusted_to = (end_date - timedelta(days=i*2)).strftime('%Y-%m-%d')
            params['from'] = adjusted_from
            params['to'] = adjusted_to
            
            response = requests.get(url, params=params)
            data = response.json()
            if data['status'] == 'ok' and data['articles']:
                all_articles.extend(data['articles'])
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        if not all_articles:
            logger.warning("No articles found for the given parameters.")
            return False        
        if data['status'] != 'ok':
            logger.error(f"API returned error: {data.get('message', 'Unknown error')}")
            return False
            
        articles = all_articles  # Use all_articles instead of data['articles']
        
        if not articles: 
            logger.warning("No articles found for the given parameters.")
            return False
            
        # Extract relevant fields from each article
        processed_articles = []
        for article in articles:
            processed_article = {
                'title': article.get('title', ''),
                'source': article.get('source', {}).get('name', ''),
                'author': article.get('author', ''),
                'published_at': article.get('publishedAt', ''),
                'content_length': len(article.get('content', '')) if article.get('content') else 0,
                'description_length': len(article.get('description', '')) if article.get('description') else 0
            }
            processed_articles.append(processed_article)
        
        # Create DataFrame
        df = pd.DataFrame(processed_articles)
        
        # Add timestamp column (collection date)
        df['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to CSV
        output_path = os.path.join('data', 'raw_data.csv')
        df.to_csv(output_path, index=False)
        
        logger.info(f"Successfully collected {len(df)} news articles and saved to {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to News API: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    collect_news_data()