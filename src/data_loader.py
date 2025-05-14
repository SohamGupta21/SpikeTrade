"""
Data loading and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import time
from pathlib import Path
from typing import Dict, Optional
import json

class DataLoader:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "market_data_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from file if it exists"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def _get_cache_key(self, symbol: str, start_date: str, end_date: str) -> str:
        """Generate cache key for data request"""
        return f"{symbol}_{start_date}_{end_date}"
    
    def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        max_retries: int = 5,
        retry_delay: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        Get stock data with caching and improved rate limit handling
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date
            end_date (str): End date
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Delay between retries in seconds
            
        Returns:
            Optional[pd.DataFrame]: Stock data or None if failed
        """
        cache_key = self._get_cache_key(symbol, start_date, end_date)
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = pd.read_csv(self.cache_dir / f"{cache_key}.csv", index_col=0, parse_dates=True)
            print(f"Using cached data for {symbol}")
            return cached_data
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1} for {symbol}...")
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if not data.empty:
                    # Save to cache
                    data.to_csv(self.cache_dir / f"{cache_key}.csv")
                    self.cache[cache_key] = {
                        'symbol': symbol,
                        'start_date': start_date,
                        'end_date': end_date,
                        'timestamp': datetime.now().isoformat()
                    }
                    self._save_cache()
                    return data
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Max retries reached for {symbol}")
        
        return None

def preprocess_market_data(market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Preprocess market data by calculating technical indicators
    
    Args:
        market_data (Dict[str, pd.DataFrame]): Raw market data
        
    Returns:
        Dict[str, pd.DataFrame]: Processed market data
    """
    processed_data = {}
    
    for symbol, data in market_data.items():
        if data is None or data.empty:
            continue
            
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        
        # Calculate moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        
        # Calculate ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        
        # Drop NaN values
        data = data.dropna()
        
        processed_data[symbol] = data
    
    return processed_data 