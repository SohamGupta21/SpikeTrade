"""
Market simulation functions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from trading_strategies import TradingStrategy

def simulate_market_multiple(
    traders: List[TradingStrategy],
    market_data: Dict[str, pd.DataFrame],
    stock_groups: Dict[str, List[str]],
    days: int = 100
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, float]]:
    """
    Simulate market for multiple stocks with cross-asset interactions
    
    Args:
        traders (List[TradingStrategy]): List of trading strategies
        market_data (Dict[str, pd.DataFrame]): Historical market data
        stock_groups (Dict[str, List[str]]): Groups of related stocks
        days (int): Number of days to simulate
    
    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, float]]:
            Simulated prices, volumes, and group correlations
    """
    # Initialize results
    simulated_prices = {}
    simulated_volumes = {}
    group_correlations = {}
    
    # Get latest date from market data
    latest_date = max(data.index[-1] for data in market_data.values())
    
    # Simulate each group
    for group_name, symbols in stock_groups.items():
        print(f"\nSimulating group: {group_name}")
        
        # Initialize group data
        group_prices = {}
        group_volumes = {}
        
        # Simulate each stock in the group
        for symbol in symbols:
            if symbol not in market_data:
                continue
                
            print(f"Simulating {symbol}")
            
            # Get historical data
            data = market_data[symbol].copy()
            
            # Simulate prices and volumes
            prices, volumes = simulate_stock(
                data=data,
                traders=traders,
                days=days,
                start_date=latest_date
            )
            
            group_prices[symbol] = prices
            group_volumes[symbol] = volumes
        
        # Calculate group correlations
        if len(group_prices) > 1:
            prices_df = pd.DataFrame(group_prices)
            correlation = prices_df.pct_change().corr().mean().mean()
            group_correlations[group_name] = correlation
        
        # Store results
        simulated_prices.update(group_prices)
        simulated_volumes.update(group_volumes)
    
    return simulated_prices, simulated_volumes, group_correlations

def simulate_stock(
    data: pd.DataFrame,
    traders: List[TradingStrategy],
    days: int,
    start_date: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a single stock's price and volume
    
    Args:
        data (pd.DataFrame): Historical market data
        traders (List[TradingStrategy]): List of trading strategies
        days (int): Number of days to simulate
        start_date (datetime): Start date for simulation
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Simulated prices and volumes
    """
    # Get initial values
    last_price = data['Close'].iloc[-1]
    last_volume = data['Volume'].iloc[-1]
    
    # Calculate historical volatility
    returns = data['Close'].pct_change()
    volatility = returns.std()
    
    # Initialize simulation arrays
    dates = pd.date_range(start=start_date, periods=days+1)
    prices = np.zeros(days+1)
    volumes = np.zeros(days+1)
    
    # Set initial values
    prices[0] = last_price
    volumes[0] = last_volume
    
    # Simulate each day
    for i in range(1, days+1):
        # Generate random return
        daily_return = np.random.normal(0, volatility)
        
        # Calculate new price
        prices[i] = prices[i-1] * (1 + daily_return)
        
        # Calculate new volume
        volume_change = np.random.normal(0, 0.2)  # 20% standard deviation
        volumes[i] = volumes[i-1] * (1 + volume_change)
        
        # Ensure positive values
        prices[i] = max(prices[i], 0.01)
        volumes[i] = max(volumes[i], 1000)
    
    # Create DataFrames
    price_df = pd.DataFrame({
        'Close': prices,
        'Open': prices * (1 + np.random.normal(0, 0.001, size=len(prices))),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.002, size=len(prices)))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.002, size=len(prices))))
    }, index=dates)
    
    volume_df = pd.DataFrame({
        'Volume': volumes
    }, index=dates)
    
    return price_df, volume_df

def plot_group_results(
    actual_data: Dict[str, pd.DataFrame],
    simulated_prices: Dict[str, pd.DataFrame],
    simulated_volumes: Dict[str, pd.DataFrame],
    group_name: str,
    dates: pd.DatetimeIndex
):
    """
    Plot results for a group of stocks
    
    Args:
        actual_data (Dict[str, pd.DataFrame]): Historical market data
        simulated_prices (Dict[str, pd.DataFrame]): Simulated prices
        simulated_volumes (Dict[str, pd.DataFrame]): Simulated volumes
        group_name (str): Name of the stock group
        dates (pd.DatetimeIndex): Dates to plot
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot prices
    for symbol in simulated_prices.keys():
        if symbol in actual_data:
            # Plot actual data
            ax1.plot(actual_data[symbol].index, actual_data[symbol]['Close'],
                    label=f'{symbol} (Actual)', alpha=0.5)
            
            # Plot simulated data
            ax1.plot(dates, simulated_prices[symbol]['Close'],
                    label=f'{symbol} (Simulated)', linestyle='--')
    
    ax1.set_title(f'Price Simulation - {group_name}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot volumes
    for symbol in simulated_volumes.keys():
        if symbol in actual_data:
            # Plot actual data
            ax2.plot(actual_data[symbol].index, actual_data[symbol]['Volume'],
                    label=f'{symbol} (Actual)', alpha=0.5)
            
            # Plot simulated data
            ax2.plot(dates, simulated_volumes[symbol]['Volume'],
                    label=f'{symbol} (Simulated)', linestyle='--')
    
    ax2.set_title(f'Volume Simulation - {group_name}')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_group_performance(
    actual_data: Dict[str, pd.DataFrame],
    simulated_prices: Dict[str, pd.DataFrame],
    simulated_volumes: Dict[str, pd.DataFrame],
    group_name: str
) -> Dict[str, float]:
    """
    Analyze performance metrics for a group of stocks
    
    Args:
        actual_data (Dict[str, pd.DataFrame]): Historical market data
        simulated_prices (Dict[str, pd.DataFrame]): Simulated prices
        simulated_volumes (Dict[str, pd.DataFrame]): Simulated volumes
        group_name (str): Name of the stock group
    
    Returns:
        Dict[str, float]: Performance metrics
    """
    metrics = {}
    
    for symbol in simulated_prices.keys():
        if symbol in actual_data:
            # Calculate price metrics
            actual_returns = actual_data[symbol]['Close'].pct_change()
            simulated_returns = simulated_prices[symbol]['Close'].pct_change()
            
            # Calculate correlation
            correlation = actual_returns.corr(simulated_returns)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((actual_returns - simulated_returns) ** 2))
            
            # Calculate volume metrics
            volume_correlation = actual_data[symbol]['Volume'].corr(simulated_volumes[symbol]['Volume'])
            
            metrics[symbol] = {
                'price_correlation': correlation,
                'price_rmse': rmse,
                'volume_correlation': volume_correlation
            }
    
    return metrics 