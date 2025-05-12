"""
Liquidity metrics calculation including Kyle's lambda and Amihud's illiquidity measure.
"""

import numpy as np
import pandas as pd
from typing import Union, List

def calculate_kyle_lambda(price_changes: np.ndarray, volume: np.ndarray) -> float:
    """
    Calculate Kyle's lambda (price impact coefficient) with enhanced accuracy.
    
    Args:
        price_changes: Array of price changes
        volume: Array of trading volumes
        
    Returns:
        Kyle's lambda coefficient
    """
    # Convert inputs to numpy arrays
    price_changes = np.array(price_changes)
    volume = np.array(volume)
    
    # Ensure matching lengths
    min_len = min(len(price_changes), len(volume))
    if min_len == 0:
        return 0.0001
        
    price_changes = price_changes[:min_len]
    volume = volume[:min_len]
    
    # Remove zero volumes
    mask = volume != 0
    if not np.any(mask):
        return 0.0001
    
    price_changes = price_changes[mask]
    volume = volume[mask]
    
    # Calculate signed volume
    signed_volume = np.sign(price_changes) * volume
    
    try:
        # Calculate weighted covariance
        weights = 1 / (volume + 1e-6)
        weights = weights / np.sum(weights)
        
        mean_price_change = np.average(price_changes, weights=weights)
        mean_signed_volume = np.average(signed_volume, weights=weights)
        
        cov = np.sum(weights * (price_changes - mean_price_change) * 
                    (signed_volume - mean_signed_volume))
        var = np.sum(weights * (signed_volume - mean_signed_volume)**2)
        
        lambda_value = cov / var if var != 0 else 0.0001
        return np.clip(lambda_value, 0.00001, 0.01)
    except:
        return 0.0001

def calculate_amihud_illiquidity(prices: np.ndarray, volumes: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate Amihud's illiquidity measure with enhanced accuracy.
    
    Args:
        prices: Array of prices
        volumes: Array of trading volumes
        window: Rolling window size
        
    Returns:
        Array of illiquidity measures
    """
    # Convert inputs to numpy arrays
    prices = np.array(prices)
    volumes = np.array(volumes)
    
    # Validate inputs
    if len(prices) < 2 or len(volumes) < 2:
        return np.array([0.0001])
    
    # Calculate absolute returns
    returns = np.abs(np.diff(prices) / prices[:-1])
    
    # Calculate dollar volume
    dollar_volume = prices[:-1] * volumes[:-1]
    
    # Calculate illiquidity ratio
    illiquidity = returns / (dollar_volume + 1e-6)
    
    # Apply exponential weighting
    weights = np.exp(np.linspace(-1, 0, len(illiquidity)))
    weights = weights / np.sum(weights)
    
    # Calculate weighted rolling average
    illiquidity_ma = pd.Series(illiquidity).rolling(
        window=window,
        min_periods=1,
        win_type='gaussian'
    ).mean(std=3).fillna(0.0001)
    
    # Add market impact adjustment
    market_impact = 0.1 * np.std(returns) / np.mean(dollar_volume)
    illiquidity_ma = illiquidity_ma * (1 + market_impact)
    
    return np.clip(illiquidity_ma.values, 0.00001, 0.01)

def calculate_market_impact(price: float, volume: float, base_volume: float,
                          kyle_lambda: float, amihud: float) -> float:
    """
    Calculate market impact with enhanced accuracy.
    
    Args:
        price: Current price
        volume: Current volume
        base_volume: Base volume (e.g., 20-day average)
        kyle_lambda: Kyle's lambda coefficient
        amihud: Amihud's illiquidity measure
        
    Returns:
        Market impact on price
    """
    # Calculate volume ratio
    volume_ratio = volume / base_volume
    
    # Calculate temporary impact (Kyle's lambda)
    temp_impact = kyle_lambda * np.sqrt(volume_ratio)
    
    # Calculate permanent impact (Amihud)
    perm_impact = amihud * volume_ratio
    
    # Combine impacts
    total_impact = 0.7 * temp_impact + 0.3 * perm_impact
    
    # Add non-linear scaling for large trades
    if volume_ratio > 2:
        total_impact *= np.sqrt(volume_ratio)
    
    return total_impact * price 