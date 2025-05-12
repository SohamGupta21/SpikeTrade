"""
Data loading and preprocessing utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import yfinance as yf
from datetime import datetime, timedelta

def load_market_data(symbols: List[str], start_date: str, end_date: str = None) -> Dict[str, pd.DataFrame]:
    """
    Load market data for multiple symbols.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (defaults to today)
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    market_data = {}
    for symbol in symbols:
        try:
            # Download data
            data = yf.download(symbol, start=start_date, end=end_date)
            
            # Rename columns
            data.columns = [f'{col}_{symbol}' for col in data.columns]
            
            # Store in dictionary
            market_data[symbol] = data
        except Exception as e:
            print(f"Error loading data for {symbol}: {str(e)}")
            continue
    
    return market_data

def preprocess_market_data(market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Preprocess market data with enhanced features.
    
    Args:
        market_data: Dictionary of market data DataFrames
        
    Returns:
        Dictionary of preprocessed DataFrames
    """
    processed_data = {}
    
    for symbol, data in market_data.items():
        try:
            # Calculate returns
            for col in ['Open', 'High', 'Low', 'Close']:
                col_name = f'{col}_{symbol}'
                if col_name in data.columns:
                    data[f'Returns_{col}_{symbol}'] = data[col_name].pct_change()
            
            # Calculate volatility
            data[f'Volatility_{symbol}'] = data[f'Returns_Close_{symbol}'].rolling(20).std()
            
            # Calculate moving averages
            for window in [5, 10, 20, 50]:
                data[f'MA_{window}_{symbol}'] = data[f'Close_{symbol}'].rolling(window).mean()
            
            # Calculate RSI
            delta = data[f'Close_{symbol}'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data[f'RSI_{symbol}'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = data[f'Close_{symbol}'].ewm(span=12, adjust=False).mean()
            exp2 = data[f'Close_{symbol}'].ewm(span=26, adjust=False).mean()
            data[f'MACD_{symbol}'] = exp1 - exp2
            data[f'MACD_Signal_{symbol}'] = data[f'MACD_{symbol}'].ewm(span=9, adjust=False).mean()
            
            # Calculate Bollinger Bands
            data[f'BB_Middle_{symbol}'] = data[f'Close_{symbol}'].rolling(20).mean()
            data[f'BB_Std_{symbol}'] = data[f'Close_{symbol}'].rolling(20).std()
            data[f'BB_Upper_{symbol}'] = data[f'BB_Middle_{symbol}'] + 2 * data[f'BB_Std_{symbol}']
            data[f'BB_Lower_{symbol}'] = data[f'BB_Middle_{symbol}'] - 2 * data[f'BB_Std_{symbol}']
            
            # Calculate volume indicators
            data[f'Volume_MA_{symbol}'] = data[f'Volume_{symbol}'].rolling(20).mean()
            data[f'Volume_Ratio_{symbol}'] = data[f'Volume_{symbol}'] / data[f'Volume_MA_{symbol}']
            
            # Calculate price momentum
            data[f'Momentum_{symbol}'] = data[f'Close_{symbol}'].pct_change(20)
            
            # Fill missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            processed_data[symbol] = data
        except Exception as e:
            print(f"Error preprocessing data for {symbol}: {str(e)}")
            continue
    
    return processed_data

def align_market_data(market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align market data across symbols.
    
    Args:
        market_data: Dictionary of market data DataFrames
        
    Returns:
        Dictionary of aligned DataFrames
    """
    # Get common index
    common_index = None
    for data in market_data.values():
        if common_index is None:
            common_index = data.index
        else:
            common_index = common_index.intersection(data.index)
    
    # Align data
    aligned_data = {}
    for symbol, data in market_data.items():
        aligned_data[symbol] = data.loc[common_index]
    
    return aligned_data

def split_market_data(market_data: Dict[str, pd.DataFrame],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Tuple[Dict, Dict, Dict]:
    """
    Split market data into train, validation, and test sets.
    
    Args:
        market_data: Dictionary of market data DataFrames
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    train_data = {}
    val_data = {}
    test_data = {}
    
    for symbol, data in market_data.items():
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data[symbol] = data.iloc[:train_end]
        val_data[symbol] = data.iloc[train_end:val_end]
        test_data[symbol] = data.iloc[val_end:]
    
    return train_data, val_data, test_data

def calculate_market_returns(market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate market returns across all symbols.
    
    Args:
        market_data: Dictionary of market data DataFrames
        
    Returns:
        DataFrame of market returns
    """
    returns = pd.DataFrame()
    
    for symbol, data in market_data.items():
        returns[symbol] = data[f'Returns_Close_{symbol}']
    
    return returns

def calculate_market_correlation(market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate correlation matrix across all symbols.
    
    Args:
        market_data: Dictionary of market data DataFrames
        
    Returns:
        Correlation matrix DataFrame
    """
    returns = calculate_market_returns(market_data)
    return returns.corr() 