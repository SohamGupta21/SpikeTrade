"""
Data structures for the trading framework.
"""

from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class TradePosition:
    entry_price: float
    entry_date: datetime
    position_type: str  # 'long' or 'short'
    size: int
    exit_price: float = None
    exit_date: datetime = None
    pnl: float = None

class PortfolioMetrics:
    """
    Class to track portfolio performance metrics
    """
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.current_position = None
        self.trades = []
        self.daily_capital = []
        self.daily_returns = []
        self.trade_count = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
    
    def open_position(self, price, date, position_type, size=1.0):
        """
        Open a new trading position
        
        Args:
            price (float): Entry price
            date (datetime): Entry date
            position_type (str): Either 'long' or 'short'
            size (float): Position size (default=1.0)
        """
        if self.current_position is not None:
            self.close_position(price, date)
        
        self.current_position = TradePosition(
            entry_price=price,
            entry_date=date,
            position_type=position_type,
            size=size
        )
    
    def close_position(self, price, date):
        """
        Close the current trading position
        
        Args:
            price (float): Exit price
            date (datetime): Exit date
        """
        if self.current_position is None:
            return
        
        # Calculate PnL
        if self.current_position.position_type == 'long':
            pnl = (price - self.current_position.entry_price) * self.current_position.size
        else:  # short
            pnl = (self.current_position.entry_price - price) * self.current_position.size
        
        # Update metrics
        self.current_capital += pnl
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0:
            self.winning_trades += 1
        
        # Record trade
        self.trades.append({
            'entry_date': self.current_position.entry_date,
            'exit_date': date,
            'entry_price': self.current_position.entry_price,
            'exit_price': price,
            'position_type': self.current_position.position_type,
            'pnl': pnl,
            'return': pnl / self.current_position.entry_price * 100
        })
        
        self.current_position = None
    
    def calculate_metrics(self):
        """
        Calculate portfolio performance metrics
        
        Returns:
            dict: Dictionary of performance metrics
        """
        if not self.trades:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_return_per_trade': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Calculate metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        win_rate = self.winning_trades / self.trade_count * 100 if self.trade_count > 0 else 0
        avg_return_per_trade = np.mean([trade['return'] for trade in self.trades])
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        if len(self.daily_returns) > 1:
            returns_std = np.std(self.daily_returns)
            sharpe_ratio = np.mean(self.daily_returns) / returns_std * np.sqrt(252) if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        if len(self.daily_capital) > 0:
            cummax = np.maximum.accumulate(self.daily_capital)
            drawdown = (cummax - self.daily_capital) / cummax
            max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return_per_trade,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        } 