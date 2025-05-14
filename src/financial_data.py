"""
Financial news and earnings data handling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import json
from pathlib import Path

class FinancialDataLoader:
    """
    Loads and processes financial news and earnings data
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # FNSPID dataset URLs
        self.fnspid_urls = {
            'stock_price': "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip",
            'stock_news': "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
        }
    
    def download_fnspid_data(self):
        """
        Download FNSPID dataset if not already present
        """
        for name, url in self.fnspid_urls.items():
            file_path = self.data_dir / f"{name}.csv"
            if not file_path.exists():
                print(f"Downloading {name} data...")
                response = requests.get(url)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    print(f"Failed to download {name} data")
    
    def load_news_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load news data for a specific symbol and date range
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: News data with sentiment scores
        """
        news_file = self.data_dir / "stock_news.csv"
        if not news_file.exists():
            self.download_fnspid_data()
        
        # Load and filter news data
        news_data = pd.read_csv(news_file)
        news_data['date'] = pd.to_datetime(news_data['date'])
        
        # Filter by symbol and date range
        mask = (
            (news_data['symbol'] == symbol) &
            (news_data['date'] >= start_date) &
            (news_data['date'] <= end_date)
        )
        
        return news_data[mask]
    
    def get_earnings_data(self, symbol: str) -> pd.DataFrame:
        """
        Get earnings data for a symbol using yfinance
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Earnings data
        """
        import yfinance as yf
        
        stock = yf.Ticker(symbol)
        earnings = stock.earnings
        
        if earnings is not None:
            earnings.index = pd.to_datetime(earnings.index)
            return earnings
        return pd.DataFrame()
    
    def combine_market_data(
        self,
        market_data: pd.DataFrame,
        news_data: pd.DataFrame,
        earnings_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine market data with news and earnings data
        
        Args:
            market_data (pd.DataFrame): Market price data
            news_data (pd.DataFrame): News data with sentiment
            earnings_data (pd.DataFrame): Earnings data
            
        Returns:
            pd.DataFrame: Combined dataset
        """
        # Ensure all data has datetime index
        market_data.index = pd.to_datetime(market_data.index)
        news_data.index = pd.to_datetime(news_data['date'])
        earnings_data.index = pd.to_datetime(earnings_data.index)
        
        # Resample news sentiment to daily
        daily_sentiment = news_data['sentiment_score'].resample('D').mean()
        
        # Create earnings indicators
        earnings_indicator = pd.Series(0, index=market_data.index)
        for date in earnings_data.index:
            # Mark earnings dates and next 5 days
            earnings_indicator[date:date + timedelta(days=5)] = 1
        
        # Combine all data
        combined_data = market_data.copy()
        combined_data['news_sentiment'] = daily_sentiment
        combined_data['earnings_indicator'] = earnings_indicator
        
        # Fill missing values
        combined_data = combined_data.fillna(method='ffill')
        
        return combined_data 