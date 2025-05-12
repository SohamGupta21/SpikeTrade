"""
Performance metrics calculation for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate returns from price series.
    
    Args:
        prices: Array of prices
        
    Returns:
        Array of returns
    """
    return np.diff(prices) / prices[:-1]

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    if len(excess_returns) < 2:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate
    if len(excess_returns) < 2:
        return 0.0
    
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0
    
    return np.mean(excess_returns) / downside_std * np.sqrt(252)

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Array of returns
        
    Returns:
        Maximum drawdown
    """
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = pd.Series(cumulative_returns).expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    return abs(drawdowns.min())

def calculate_calmar_ratio(returns: np.ndarray) -> float:
    """
    Calculate Calmar ratio.
    
    Args:
        returns: Array of returns
        
    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0
    
    annualized_return = np.mean(returns) * 252
    max_drawdown = calculate_max_drawdown(returns)
    
    if max_drawdown == 0:
        return 0.0
    
    return annualized_return / max_drawdown

def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculate Omega ratio.
    
    Args:
        returns: Array of returns
        threshold: Return threshold
        
    Returns:
        Omega ratio
    """
    if len(returns) < 2:
        return 0.0
    
    positive_returns = returns[returns > threshold]
    negative_returns = returns[returns <= threshold]
    
    if len(negative_returns) == 0:
        return np.inf
    
    return np.sum(positive_returns - threshold) / np.sum(threshold - negative_returns)

def calculate_treynor_ratio(returns: np.ndarray, market_returns: np.ndarray,
                          risk_free_rate: float = 0.0) -> float:
    """
    Calculate Treynor ratio.
    
    Args:
        returns: Array of returns
        market_returns: Array of market returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Treynor ratio
    """
    if len(returns) < 2 or len(market_returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    excess_market_returns = market_returns - risk_free_rate
    
    beta = np.cov(excess_returns, excess_market_returns)[0, 1] / np.var(excess_market_returns)
    if beta == 0:
        return 0.0
    
    return np.mean(excess_returns) / beta

def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate Information ratio.
    
    Args:
        returns: Array of returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        Information ratio
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return np.mean(active_returns) / tracking_error * np.sqrt(252)

def calculate_win_rate(trades: List[Dict]) -> float:
    """
    Calculate win rate from trades.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Win rate
    """
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade['quantity'] * trade['price'] > 0)
    return winning_trades / len(trades)

def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Calculate profit factor from trades.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Profit factor
    """
    if not trades:
        return 0.0
    
    gross_profit = sum(trade['quantity'] * trade['price']
                      for trade in trades
                      if trade['quantity'] * trade['price'] > 0)
    gross_loss = abs(sum(trade['quantity'] * trade['price']
                        for trade in trades
                        if trade['quantity'] * trade['price'] < 0))
    
    if gross_loss == 0:
        return np.inf
    
    return gross_profit / gross_loss

def calculate_metrics_summary(returns: np.ndarray, trades: List[Dict],
                            market_returns: np.ndarray = None) -> Dict:
    """
    Calculate comprehensive performance metrics summary.
    
    Args:
        returns: Array of returns
        trades: List of trade dictionaries
        market_returns: Array of market returns (optional)
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {
        'total_return': np.prod(1 + returns) - 1,
        'annualized_return': np.mean(returns) * 252,
        'volatility': np.std(returns) * np.sqrt(252),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'calmar_ratio': calculate_calmar_ratio(returns),
        'omega_ratio': calculate_omega_ratio(returns),
        'win_rate': calculate_win_rate(trades),
        'profit_factor': calculate_profit_factor(trades)
    }
    
    if market_returns is not None:
        metrics.update({
            'treynor_ratio': calculate_treynor_ratio(returns, market_returns),
            'information_ratio': calculate_information_ratio(returns, market_returns),
            'beta': np.cov(returns, market_returns)[0, 1] / np.var(market_returns),
            'alpha': np.mean(returns) - metrics['beta'] * np.mean(market_returns)
        })
    
    return metrics 