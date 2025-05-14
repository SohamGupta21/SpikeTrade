"""
Trading strategy implementations.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from structures import TradePosition, PortfolioMetrics
from neural_network import TradingSNN
from financial_data import FinancialDataLoader

class TradingStrategy:
    """
    Base trading strategy class
    """
    def __init__(self, name: str, initial_capital: float = 100000.0):
        self.name = name
        self.portfolio = PortfolioMetrics(initial_capital)
        self.current_position = None
    
    def generate_signal(self, data: pd.DataFrame) -> str:
        """
        Generate trading signal
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            str: Trading signal ('buy', 'sell', or 'hold')
        """
        raise NotImplementedError
    
    def execute_trade(self, signal: str, price: float, date: datetime):
        """
        Execute trade based on signal
        
        Args:
            signal (str): Trading signal
            price (float): Current price
            date (datetime): Current date
        """
        if signal == 'buy' and self.current_position is None:
            self.portfolio.open_position(price, date, 'long')
            self.current_position = 'long'
        elif signal == 'sell' and self.current_position is None:
            self.portfolio.open_position(price, date, 'short')
            self.current_position = 'short'
        elif signal == 'sell' and self.current_position == 'long':
            self.portfolio.close_position(price, date)
            self.current_position = None
        elif signal == 'buy' and self.current_position == 'short':
            self.portfolio.close_position(price, date)
            self.current_position = None

class SNNTrader(TradingStrategy):
    """
    Trader that uses a Spiking Neural Network to make trading decisions
    """
    def __init__(self, name: str, initial_capital: float = 100000.0):
        super().__init__(name, initial_capital)
        self.model = TradingSNN(
            n_market_inputs=10,
            n_news_inputs=1,
            n_earnings_inputs=1
        )
        self.data_loader = FinancialDataLoader()
        
        # Market data features
        self.market_features = [
            'Returns', 'SMA_20', 'SMA_50', 'RSI', 'MACD',
            'Signal_Line', 'BB_Upper', 'BB_Lower', 'Volume', 'ATR'
        ]
    
    def generate_signal(self, data: pd.DataFrame) -> str:
        """
        Generate trading signal using SNN
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            str: Trading signal
        """
        # Prepare market features
        market_features = data[self.market_features].values
        market_features = (market_features - market_features.mean()) / market_features.std()  # Normalize
        market_features = torch.FloatTensor(market_features)
        
        # Prepare news sentiment
        news_sentiment = torch.FloatTensor(data['news_sentiment'].values.reshape(-1, 1))
        
        # Prepare earnings indicator
        earnings_indicator = torch.FloatTensor(data['earnings_indicator'].values.reshape(-1, 1))
        
        # Get prediction
        spikes, _ = self.model.forward(
            market_data=market_features,
            news_data=news_sentiment,
            earnings_data=earnings_indicator
        )
        signal = self.model.get_prediction(spikes)
        
        return signal
    
    def update_model(self, reward: float):
        """
        Update model weights based on reward
        
        Args:
            reward (float): Reward signal
        """
        self.model.update_weights(reward)
    
    def load_additional_data(self, symbol: str, start_date: str, end_date: str):
        """
        Load news and earnings data
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date
            end_date (str): End date
        """
        # Load news data
        news_data = self.data_loader.load_news_data(symbol, start_date, end_date)
        
        # Load earnings data
        earnings_data = self.data_loader.get_earnings_data(symbol)
        
        return news_data, earnings_data

class SignalTrader(TradingStrategy):
    """
    Trader that uses technical signals to make trading decisions
    """
    def __init__(self, name: str, initial_capital: float = 100000.0):
        super().__init__(name, initial_capital)
    
    def generate_signal(self, data: pd.DataFrame) -> str:
        """
        Generate trading signal using technical indicators
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            str: Trading signal
        """
        # Get latest data
        latest = data.iloc[-1]
        
        # Check RSI
        if latest['RSI'] < 30:
            return 'buy'
        elif latest['RSI'] > 70:
            return 'sell'
        
        # Check MACD
        if latest['MACD'] > latest['Signal_Line']:
            return 'buy'
        elif latest['MACD'] < latest['Signal_Line']:
            return 'sell'
        
        # Check Bollinger Bands
        if latest['Close'] < latest['BB_Lower']:
            return 'buy'
        elif latest['Close'] > latest['BB_Upper']:
            return 'sell'
        
        return 'hold'

class RandomTrader(TradingStrategy):
    """
    Trader that makes random trading decisions
    """
    def __init__(self, name: str, initial_capital: float = 100000.0):
        super().__init__(name, initial_capital)
    
    def generate_signal(self, data: pd.DataFrame) -> str:
        """
        Generate random trading signal
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            str: Trading signal
        """
        return np.random.choice(['buy', 'sell', 'hold'], p=[0.3, 0.3, 0.4]) 