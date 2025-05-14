"""
Market analysis functions and metrics calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range
    
    Args:
        data (pd.DataFrame): Price data with High, Low, Close columns
        period (int): ATR period
    
    Returns:
        pd.Series: ATR values
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        data (pd.DataFrame): Price data
        period (int): Moving average period
        std_dev (float): Number of standard deviations
    
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: Middle, Upper, and Lower bands
    """
    middle_band = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return middle_band, upper_band, lower_band

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    
    Args:
        data (pd.DataFrame): Price data
        period (int): RSI period
    
    Returns:
        pd.Series: RSI values
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data (pd.DataFrame): Price data
        fast (int): Fast period
        slow (int): Slow period
        signal (int): Signal period
    
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: MACD line, Signal line, and Histogram
    """
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def calculate_volatility_metrics(data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    """
    Calculate volatility metrics
    
    Args:
        data (pd.DataFrame): Price data
        window (int): Rolling window size
    
    Returns:
        Dict[str, float]: Dictionary of volatility metrics
    """
    returns = data['Close'].pct_change()
    
    metrics = {
        'daily_volatility': returns.std(),
        'annualized_volatility': returns.std() * np.sqrt(252),
        'rolling_volatility': returns.rolling(window=window).std().mean(),
        'max_drawdown': (data['Close'] / data['Close'].cummax() - 1).min(),
        'var_95': returns.quantile(0.05),
        'var_99': returns.quantile(0.01)
    }
    
    return metrics

def analyze_market_regime(data: pd.DataFrame, window: int = 20) -> str:
    """
    Analyze current market regime
    
    Args:
        data (pd.DataFrame): Price data
        window (int): Rolling window size
    
    Returns:
        str: Market regime ('trending', 'ranging', or 'volatile')
    """
    # Calculate metrics
    returns = data['Close'].pct_change()
    volatility = returns.rolling(window=window).std()
    trend = data['Close'].rolling(window=window).mean().pct_change()
    
    # Current values
    current_volatility = volatility.iloc[-1]
    current_trend = trend.iloc[-1]
    
    # Determine regime
    if abs(current_trend) > 0.001:  # Strong trend
        return 'trending'
    elif current_volatility > 0.02:  # High volatility
        return 'volatile'
    else:
        return 'ranging' 