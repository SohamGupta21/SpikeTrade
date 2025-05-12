"""
Portfolio management and position tracking functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

class Portfolio:
    """Portfolio management class."""
    
    def __init__(self, initial_capital: float = 1000000.0):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Initial capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.performance_history = []
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """
        Update position for a symbol.
        
        Args:
            symbol: Stock symbol
            quantity: Position quantity
            price: Current price
        """
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        # Record trade
        self.trades.append({
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now()
        })
        
        # Update position
        self.positions[symbol] += quantity
        
        # Update cash
        self.cash -= quantity * price
    
    def get_position_value(self, symbol: str, price: float) -> float:
        """
        Get current value of position.
        
        Args:
            symbol: Stock symbol
            price: Current price
            
        Returns:
            Position value
        """
        if symbol not in self.positions:
            return 0.0
        return self.positions[symbol] * price
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Get total portfolio value.
        
        Args:
            current_prices: Dictionary of current prices
            
        Returns:
            Total portfolio value
        """
        position_values = sum(
            self.get_position_value(symbol, price)
            for symbol, price in current_prices.items()
        )
        return self.cash + position_values
    
    def get_returns(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate portfolio returns.
        
        Args:
            current_prices: Dictionary of current prices
            
        Returns:
            Portfolio returns
        """
        total_value = self.get_total_value(current_prices)
        return (total_value - self.initial_capital) / self.initial_capital
    
    def get_position_weights(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate position weights.
        
        Args:
            current_prices: Dictionary of current prices
            
        Returns:
            Dictionary of position weights
        """
        total_value = self.get_total_value(current_prices)
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, price in current_prices.items():
            position_value = self.get_position_value(symbol, price)
            weights[symbol] = position_value / total_value
        
        return weights
    
    def get_risk_metrics(self, current_prices: Dict[str, float],
                        historical_returns: Dict[str, pd.Series]) -> Dict:
        """
        Calculate portfolio risk metrics.
        
        Args:
            current_prices: Dictionary of current prices
            historical_returns: Dictionary of historical returns
            
        Returns:
            Dictionary of risk metrics
        """
        weights = self.get_position_weights(current_prices)
        
        # Calculate portfolio volatility
        portfolio_returns = pd.Series(0, index=next(iter(historical_returns.values())).index)
        for symbol, returns in historical_returns.items():
            if symbol in weights:
                portfolio_returns += returns * weights[symbol]
        
        volatility = portfolio_returns.std()
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'max_drawdown': max_drawdown
        }
    
    def rebalance(self, target_weights: Dict[str, float],
                 current_prices: Dict[str, float]):
        """
        Rebalance portfolio to target weights.
        
        Args:
            target_weights: Dictionary of target weights
            current_prices: Dictionary of current prices
        """
        total_value = self.get_total_value(current_prices)
        
        for symbol, target_weight in target_weights.items():
            if symbol in current_prices:
                target_value = total_value * target_weight
                current_value = self.get_position_value(symbol, current_prices[symbol])
                value_diff = target_value - current_value
                
                if abs(value_diff) > 0.01 * total_value:  # 1% threshold
                    quantity = value_diff / current_prices[symbol]
                    self.update_position(symbol, quantity, current_prices[symbol])
    
    def record_performance(self, current_prices: Dict[str, float]):
        """
        Record portfolio performance.
        
        Args:
            current_prices: Dictionary of current prices
        """
        performance = {
            'timestamp': datetime.now(),
            'total_value': self.get_total_value(current_prices),
            'returns': self.get_returns(current_prices),
            'cash': self.cash,
            'positions': self.positions.copy()
        }
        self.performance_history.append(performance)
    
    def get_performance_summary(self) -> Dict:
        """
        Get portfolio performance summary.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.performance_history:
            return {}
        
        returns = [p['returns'] for p in self.performance_history]
        total_values = [p['total_value'] for p in self.performance_history]
        
        return {
            'total_return': returns[-1],
            'annualized_return': np.mean(returns) * 252,
            'volatility': np.std(returns) * np.sqrt(252),
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),
            'max_drawdown': min(returns),
            'final_value': total_values[-1]
        } 