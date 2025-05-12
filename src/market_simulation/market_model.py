"""
Market simulation model implementing price dynamics and cross-asset interactions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .liquidity import calculate_kyle_lambda, calculate_amihud_illiquidity
from .volatility import calculate_volatility_characteristics, group_stocks_by_volatility

def simulate_market_multiple(traders: List, market_data: Dict, stock_groups: Dict, days: int = 100) -> Tuple[Dict, Dict, Dict]:
    """
    Simulate market for multiple stocks with cross-asset interactions and improved price tracking.
    
    Args:
        traders: List of trader instances
        market_data: Dictionary of stock data DataFrames
        stock_groups: Dictionary of stock groups and their constituent stocks
        days: Number of days to simulate
        
    Returns:
        Tuple containing:
        - Dictionary of simulated prices by group and stock
        - Dictionary of simulated volumes by group and stock
        - Dictionary of group correlations
    """
    # Initialize results containers
    simulated_prices = {}
    simulated_volumes = {}
    group_correlations = {}
    
    # Calculate initial group betas and correlations
    group_betas = {}
    for group_name, stocks in stock_groups.items():
        group_returns = pd.DataFrame()
        for stock in stocks:
            if stock in market_data:
                returns = market_data[stock][f'Close_{stock}'].pct_change().dropna()
                group_returns[stock] = returns
        
        if not group_returns.empty:
            market_return = group_returns.mean(axis=1)
            group_betas[group_name] = group_returns.std().mean() / market_return.std()
            group_correlations[group_name] = group_returns.corr().mean().mean()
    
    # Simulate each group
    for group_name, stocks in stock_groups.items():
        group_prices = {}
        group_volumes = {}
        
        # Get group characteristics
        beta = group_betas.get(group_name, 1.0)
        correlation = group_correlations.get(group_name, 0.3)
        
        # Generate correlated market movements
        market_movement = np.random.normal(0, 0.01, days)
        random_walks = np.random.multivariate_normal(
            mean=[0] * len(stocks),
            cov=np.full((len(stocks), len(stocks)), correlation) + \
                np.eye(len(stocks)) * (1 - correlation),
            size=days
        )
        
        # Simulate each stock in the group
        for i, stock in enumerate(stocks):
            if stock in market_data:
                prices, volumes = _simulate_single_stock(
                    market_data[stock],
                    market_movement,
                    random_walks[:, i],
                    beta,
                    correlation
                )
                group_prices[stock] = prices
                group_volumes[stock] = volumes
        
        simulated_prices[group_name] = group_prices
        simulated_volumes[group_name] = group_volumes
    
    return simulated_prices, simulated_volumes, group_correlations

def _simulate_single_stock(stock_data: pd.DataFrame, market_movement: np.ndarray,
                         random_walk: np.ndarray, beta: float, correlation: float) -> Tuple[List[float], List[float]]:
    """
    Simulate a single stock's price and volume movements.
    
    Args:
        stock_data: DataFrame containing stock data
        market_movement: Array of market movements
        random_walk: Array of stock-specific random movements
        beta: Stock's beta to market
        correlation: Stock's correlation with group
        
    Returns:
        Tuple of (prices, volumes) lists
    """
    # Initialize with last actual price and volume
    last_price = stock_data[f'Close_{stock_data.name}'].iloc[-1]
    last_volume = stock_data[f'Volume_{stock_data.name}'].iloc[-1]
    
    # Calculate stock-specific metrics
    returns = stock_data[f'Close_{stock_data.name}'].pct_change().dropna()
    volatility = returns.std()
    mean_return = returns.mean()
    
    # Calculate historical metrics
    price_history = stock_data[f'Close_{stock_data.name}'].values[-20:]
    volume_history = stock_data[f'Volume_{stock_data.name}'].values[-20:]
    
    # Calculate momentum and trend
    momentum = (price_history[-1] - price_history[0]) / price_history[0]
    volume_trend = np.mean(volume_history[-5:]) / np.mean(volume_history)
    
    # Initialize state
    prices = [last_price]
    volumes = [last_volume]
    momentum_factor = momentum
    stock_volatility = volatility
    volume_factor = volume_trend
    
    # Calculate liquidity metrics
    kyle_lambda = calculate_kyle_lambda(
        np.diff(price_history) / price_history[:-1],
        volume_history[:-1]
    )
    amihud = calculate_amihud_illiquidity(price_history, volume_history)
    
    # Calculate trend and bias
    x = np.arange(len(price_history))
    slope, _ = np.polyfit(x, price_history, 1)
    trend_factor = slope / price_history[-1]
    recent_direction = np.sign(np.diff(price_history[-5:]))
    direction_bias = np.mean(recent_direction)
    price_momentum = np.mean(np.diff(price_history[-5:])) / price_history[-1]
    
    # Simulate each day
    for day in range(len(market_movement)):
        # Calculate price components
        market_impact = market_movement[day] * beta * (1 + 0.05 * np.sin(day/10))
        group_impact = random_walk[day] * correlation * (1 + 0.03 * np.cos(day/20))
        stock_specific = np.random.normal(mean_return, stock_volatility * 0.5)
        
        # Update volatility
        stock_volatility = np.sqrt(0.95 * stock_volatility**2 + 0.05 * stock_specific**2)
        
        # Update momentum
        momentum_factor = np.clip(
            0.95 * momentum_factor + 0.05 * (price_momentum + direction_bias),
            -volatility * 2,
            volatility * 2
        )
        
        # Calculate impacts
        volume_impact = kyle_lambda * (volumes[-1] - last_volume) / last_volume * 0.5
        illiquidity_impact = amihud[-1] * np.random.normal(0, 0.5)
        trend_impact = np.clip(
            trend_factor * (1 + 0.1 * np.sin(day/15)),
            -volatility * 2,
            volatility * 2
        )
        
        # Combine effects
        total_return = (
            0.10 * market_impact +
            0.10 * group_impact +
            0.15 * stock_specific +
            0.25 * momentum_factor +
            0.10 * volume_impact +
            0.05 * illiquidity_impact +
            0.25 * trend_impact
        )
        
        # Add directional bias
        total_return += 0.1 * np.clip(direction_bias, -volatility, volatility)
        
        # Apply mean reversion for extreme moves
        if abs(total_return) > 1.5 * volatility:
            total_return *= 0.6
        
        # Update price
        new_price = prices[-1] * (1 + total_return)
        max_move = min(0.05, 0.03 + abs(momentum_factor))
        new_price = max(prices[-1] * 0.95, min(new_price, prices[-1] * (1 + max_move)))
        
        # Update volume
        volume_change = np.exp(random_walk[day] * 0.3) * (1 + abs(total_return) * 0.5)
        volume_change *= (1 + 0.05 * np.sin(day/5))
        new_volume = last_volume * volume_change * volume_factor
        
        # Update state
        prices.append(new_price)
        volumes.append(new_volume)
        volume_factor = 0.98 * volume_factor + 0.02 * (new_volume / last_volume)
    
    return prices, volumes 