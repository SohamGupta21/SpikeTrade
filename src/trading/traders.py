"""
Trader classes implementing different trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .portfolio import Portfolio

class BaseTrader:
    """Base class for all traders."""
    
    def __init__(self, name: str, initial_capital: float = 1000000.0):
        """
        Initialize base trader.
        
        Args:
            name: Trader name
            initial_capital: Initial capital
        """
        self.name = name
        self.portfolio = Portfolio(initial_capital)
        self.trades = []
        self.positions = {}
    
    def update(self, market_data: Dict, current_prices: Dict) -> Dict[str, float]:
        """
        Update trader state and generate orders.
        
        Args:
            market_data: Dictionary of market data
            current_prices: Dictionary of current prices
            
        Returns:
            Dictionary of orders (symbol -> quantity)
        """
        raise NotImplementedError
    
    def record_trade(self, symbol: str, quantity: float, price: float):
        """
        Record a trade.
        
        Args:
            symbol: Stock symbol
            quantity: Trade quantity
            price: Trade price
        """
        self.trades.append({
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': pd.Timestamp.now()
        })
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += quantity

class SNNTrader(BaseTrader):
    """Spiking Neural Network trader implementation."""
    
    def __init__(self, name: str, initial_capital: float = 1000000.0,
                 learning_rate: float = 0.01, volatility_threshold: float = 0.02):
        """
        Initialize SNN trader.
        
        Args:
            name: Trader name
            initial_capital: Initial capital
            learning_rate: Learning rate for weight updates
            volatility_threshold: Volatility threshold for trading
        """
        super().__init__(name, initial_capital)
        self.learning_rate = learning_rate
        self.volatility_threshold = volatility_threshold
        self.weights = {}
        self.spike_history = {}
    
    def update(self, market_data: Dict, current_prices: Dict) -> Dict[str, float]:
        """
        Update SNN trader state and generate orders.
        
        Args:
            market_data: Dictionary of market data
            current_prices: Dictionary of current prices
            
        Returns:
            Dictionary of orders (symbol -> quantity)
        """
        orders = {}
        
        for symbol, data in market_data.items():
            if symbol not in self.weights:
                self.weights[symbol] = np.random.normal(0, 0.1, 5)
                self.spike_history[symbol] = []
            
            # Calculate input features
            returns = data[f'Close_{symbol}'].pct_change().dropna()
            volatility = returns.std()
            momentum = (data[f'Close_{symbol}'].iloc[-1] / 
                       data[f'Close_{symbol}'].iloc[-20] - 1)
            volume_ratio = (data[f'Volume_{symbol}'].iloc[-1] /
                           data[f'Volume_{symbol}'].rolling(20).mean().iloc[-1])
            price_trend = np.polyfit(range(20), data[f'Close_{symbol}'].iloc[-20:], 1)[0]
            
            # Generate spike
            features = np.array([volatility, momentum, volume_ratio, price_trend, 1])
            spike = np.dot(self.weights[symbol], features)
            self.spike_history[symbol].append(spike)
            
            # Update weights based on performance
            if len(self.spike_history[symbol]) > 1:
                prev_spike = self.spike_history[symbol][-2]
                current_return = returns.iloc[-1]
                weight_update = self.learning_rate * current_return * prev_spike
                self.weights[symbol] += weight_update
            
            # Generate order if spike exceeds threshold
            if abs(spike) > self.volatility_threshold:
                position_size = self.portfolio.cash * 0.1 * np.sign(spike)
                orders[symbol] = position_size / current_prices[symbol]
        
        return orders

class SignalTrader(BaseTrader):
    """Signal-based trader implementation."""
    
    def __init__(self, name: str, initial_capital: float = 1000000.0,
                 signal_threshold: float = 0.5, position_size: float = 0.1):
        """
        Initialize signal trader.
        
        Args:
            name: Trader name
            initial_capital: Initial capital
            signal_threshold: Threshold for signal generation
            position_size: Position size as fraction of capital
        """
        super().__init__(name, initial_capital)
        self.signal_threshold = signal_threshold
        self.position_size = position_size
        self.signals = {}
    
    def update(self, market_data: Dict, current_prices: Dict) -> Dict[str, float]:
        """
        Update signal trader state and generate orders.
        
        Args:
            market_data: Dictionary of market data
            current_prices: Dictionary of current prices
            
        Returns:
            Dictionary of orders (symbol -> quantity)
        """
        orders = {}
        
        for symbol, data in market_data.items():
            if symbol not in self.signals:
                self.signals[symbol] = 0
            
            # Calculate technical indicators
            returns = data[f'Close_{symbol}'].pct_change().dropna()
            volatility = returns.std()
            momentum = (data[f'Close_{symbol}'].iloc[-1] /
                       data[f'Close_{symbol}'].iloc[-20] - 1)
            volume_ratio = (data[f'Volume_{symbol}'].iloc[-1] /
                           data[f'Volume_{symbol}'].rolling(20).mean().iloc[-1])
            
            # Generate signal
            signal = (0.4 * momentum +
                     0.3 * volume_ratio +
                     0.3 * (1 if volatility > self.signal_threshold else 0))
            
            # Update signal with momentum
            self.signals[symbol] = 0.7 * self.signals[symbol] + 0.3 * signal
            
            # Generate order if signal exceeds threshold
            if abs(self.signals[symbol]) > self.signal_threshold:
                position_size = self.portfolio.cash * self.position_size * np.sign(self.signals[symbol])
                orders[symbol] = position_size / current_prices[symbol]
        
        return orders

class RandomTrader(BaseTrader):
    """Random trading strategy implementation."""
    
    def __init__(self, name: str, initial_capital: float = 1000000.0,
                 trade_probability: float = 0.1, position_size: float = 0.1):
        """
        Initialize random trader.
        
        Args:
            name: Trader name
            initial_capital: Initial capital
            trade_probability: Probability of trading
            position_size: Position size as fraction of capital
        """
        super().__init__(name, initial_capital)
        self.trade_probability = trade_probability
        self.position_size = position_size
    
    def update(self, market_data: Dict, current_prices: Dict) -> Dict[str, float]:
        """
        Update random trader state and generate orders.
        
        Args:
            market_data: Dictionary of market data
            current_prices: Dictionary of current prices
            
        Returns:
            Dictionary of orders (symbol -> quantity)
        """
        orders = {}
        
        for symbol in market_data.keys():
            # Randomly decide to trade
            if np.random.random() < self.trade_probability:
                # Randomly decide direction
                direction = np.random.choice([-1, 1])
                position_size = self.portfolio.cash * self.position_size * direction
                orders[symbol] = position_size / current_prices[symbol]
        
        return orders 