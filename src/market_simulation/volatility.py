"""
Volatility calculations and stock grouping functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans

def calculate_volatility_characteristics(returns: np.ndarray, window: int = 20) -> Tuple[float, float, float]:
    """
    Calculate volatility characteristics with enhanced accuracy.
    
    Args:
        returns: Array of returns
        window: Rolling window size
        
    Returns:
        Tuple of (volatility, skewness, kurtosis)
    """
    # Convert to numpy array
    returns = np.array(returns)
    
    # Calculate rolling volatility
    volatility = pd.Series(returns).rolling(
        window=window,
        min_periods=1,
        win_type='gaussian'
    ).std(std=3).fillna(0.01)
    
    # Calculate rolling skewness
    skewness = pd.Series(returns).rolling(
        window=window,
        min_periods=1,
        win_type='gaussian'
    ).skew().fillna(0)
    
    # Calculate rolling kurtosis
    kurtosis = pd.Series(returns).rolling(
        window=window,
        min_periods=1,
        win_type='gaussian'
    ).kurt().fillna(3)
    
    return volatility.iloc[-1], skewness.iloc[-1], kurtosis.iloc[-1]

def group_stocks_by_volatility(market_data: Dict, n_groups: int = 3) -> Dict[str, List[str]]:
    """
    Group stocks by volatility characteristics using K-means clustering.
    
    Args:
        market_data: Dictionary of stock data DataFrames
        n_groups: Number of volatility groups
        
    Returns:
        Dictionary mapping group names to lists of stock symbols
    """
    # Calculate volatility metrics for each stock
    volatility_metrics = []
    stock_symbols = []
    
    for symbol, data in market_data.items():
        try:
            # Calculate returns
            returns = data[f'Close_{symbol}'].pct_change().dropna()
            
            # Calculate volatility characteristics
            vol, skew, kurt = calculate_volatility_characteristics(returns)
            
            # Store metrics
            volatility_metrics.append([vol, skew, kurt])
            stock_symbols.append(symbol)
        except:
            continue
    
    if not volatility_metrics:
        return {'Low': [], 'Medium': [], 'High': []}
    
    # Convert to numpy array
    X = np.array(volatility_metrics)
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_groups, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Group stocks by cluster
    groups = {}
    for i in range(n_groups):
        group_stocks = [stock_symbols[j] for j in range(len(stock_symbols)) if labels[j] == i]
        group_name = f'Group_{i+1}'
        groups[group_name] = group_stocks
    
    return groups

def calculate_cross_asset_correlation(returns_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Calculate cross-asset correlation matrix with enhanced accuracy.
    
    Args:
        returns_dict: Dictionary mapping asset symbols to return arrays
        
    Returns:
        Correlation matrix as DataFrame
    """
    # Convert to DataFrame
    returns_df = pd.DataFrame(returns_dict)
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr(method='spearman')
    
    # Apply shrinkage to reduce noise
    n = len(returns_df)
    shrinkage = 0.5
    corr_matrix = shrinkage * corr_matrix + (1 - shrinkage) * np.eye(len(corr_matrix))
    
    return corr_matrix

def calculate_volatility_regime(returns: np.ndarray, window: int = 60) -> str:
    """
    Determine the current volatility regime.
    
    Args:
        returns: Array of returns
        window: Rolling window size
        
    Returns:
        Regime classification ('low', 'medium', 'high')
    """
    # Calculate rolling volatility
    volatility = pd.Series(returns).rolling(
        window=window,
        min_periods=1,
        win_type='gaussian'
    ).std(std=3)
    
    # Calculate volatility percentiles
    vol_percentiles = volatility.quantile([0.33, 0.66])
    
    # Determine regime
    current_vol = volatility.iloc[-1]
    if current_vol < vol_percentiles[0.33]:
        return 'low'
    elif current_vol < vol_percentiles[0.66]:
        return 'medium'
    else:
        return 'high' 