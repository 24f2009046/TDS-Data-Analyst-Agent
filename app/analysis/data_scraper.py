"""
Data Scraper - Handles web scraping and data extraction
Optimized for speed and common data formats
"""

import asyncio
from typing import Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

from app.utils.utils import get_logger

logger = get_logger(__name__)

class DataScraper:
    """Fast web scraper optimized for common data formats"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DataAnalystAgent/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # Cache for repeated URLs
        self.cache = {}
        
        # Configuration
        self.max_rows = 200  # Limit rows for speed
        self.timeout = 30
    
    async def initialize(self):
        """Initialize scraper"""
        logger.info("Data scraper initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.session.close()
        logger.info("Data scraper cleaned up")
    
    async def scrape_url(self, url: str) -> pd.DataFrame:
        """
        Scrape data from URL and return as DataFrame
        
        Args:
            url: URL to scrape
            
        Returns:
            A pandas DataFrame containing the scraped data.
        """
        logger.info(f"Scraping data from {url}")
        
        if url in self.cache:
            logger.info("Using cached data for URL.")
            return self.cache[url]
        
        try:
            # Use asyncio to make the request non-blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.session.get(url, timeout=self.timeout)
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Find all tables on the page
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            
            if not tables:
                logger.warning("No tables found on the page.")
                return pd.DataFrame()
            
            # Heuristic to find the main data table
            # We assume the most relevant table is the largest one by row count.
            best_table = max(tables, key=lambda t: len(t.find_all('tr')))
            
            # Read the HTML table into a pandas DataFrame
            dfs = pd.read_html(str(best_table))
            if not dfs:
                return pd.DataFrame()
                
            df = dfs[0]
            
            # Perform basic data cleaning
            df = self._clean_dataframe(df)

            # Cache the result
            self.cache[url] = df
            
            logger.info(f"Successfully scraped data. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            return pd.DataFrame()

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic cleaning on the DataFrame.
        This includes removing multi-level headers and cleaning numerical columns.
        """
        # Handle multi-level headers, which are common on Wikipedia.
        # This collapses them into a single, clean row.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        # Clean column names
        df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True).str.lower()

        # Convert columns to appropriate types
        for col in df.columns:
            if self._is_numeric_like(df[col]):
                df[col] = self._convert_to_numeric(df[col])
            # Other conversions (e.g., dates) can be added here
        
        return df
    
    def _is_numeric_like(self, series: pd.Series) -> bool:
        """Check if a series appears to be numeric, even with text"""
        # Check a sample of the data to avoid performance issues on large datasets
        sample = series.dropna().sample(min(len(series.dropna()), 100))
        numeric_count = 0
        for value in sample:
            clean_value = str(value).replace(',', '').replace('$', '').replace('%', '')
            try:
                float(clean_value)
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        # If most values look numeric, convert the column
        return numeric_count / len(sample) > 0.6
    
    def _convert_to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert series to numeric, handling common formats"""
        
        def clean_numeric(value):
            if pd.isna(value):
                return value
            
            # Convert to string and clean
            clean_str = str(value).replace(',', '').replace('$', '').replace('%', '')
            
            # Handle parentheses (negative numbers)
            if '(' in clean_str and ')' in clean_str:
                clean_str = '-' + clean_str.replace('(', '').replace(')', '')
            
            # Extract numbers
            number_match = re.search(r'-?\d+(?:\.\d+)?', clean_str)
            if number_match:
                try:
                    return float(number_match.group())
                except:
                    return value
            
            return value
        
        return series.apply(clean_numeric)

