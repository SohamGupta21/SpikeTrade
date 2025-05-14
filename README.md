# SpikeTrade
Spiking Neural Networks approach to emulate human retail investors and generate a large scale stock market simulation. Final Project for Neuro240.

## Project Structure

```
src/
├── main.py                 # Main script to run the simulation
├── structures.py           # Data structures and portfolio metrics
├── data_loader.py          # Market data fetching and preprocessing
├── market_analysis.py      # Technical indicators and market analysis
├── neural_network.py       # Spiking Neural Network implementation
├── trading_strategies.py   # Trading strategy implementations
└── market_simulation.py    # Market simulation and visualization
```

## Components

### Data Structures (`structures.py`)
- `TradePosition`: Represents a single trading position with entry/exit details
- `PortfolioMetrics`: Tracks portfolio performance, trades, and calculates metrics

### Data Loading (`data_loader.py`)
- Fetches historical stock data from Yahoo Finance
- Preprocesses market data
- Generates technical indicators and sentiment data

### Market Analysis (`market_analysis.py`)
- Calculates technical indicators (RSI, MACD, Bollinger Bands)
- Analyzes market regimes
- Computes volatility metrics

### Neural Network (`neural_network.py`)
- Implements a Spiking Neural Network using BindsNET
- Uses LIF (Leaky Integrate-and-Fire) neurons
- Implements STDP (Spike-Timing-Dependent Plasticity) learning

### Trading Strategies (`trading_strategies.py`)
- Base `TradingStrategy` class
- `SNNTrader`: Uses Spiking Neural Network for predictions
- `SignalTrader`: Uses technical indicators for trading signals
- `RandomTrader`: Generates random trading signals (baseline)

### Market Simulation (`market_simulation.py`)
- Simulates market behavior for multiple stocks
- Handles cross-asset interactions
- Generates visualizations of results

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simulation:
```bash
cd src
python main.py
```

## Dependencies
- numpy: Numerical computations
- pandas: Data manipulation
- yfinance: Market data fetching
- matplotlib: Visualization
- seaborn: Statistical visualization
- torch: Deep learning framework
- bindsnet: Spiking Neural Network implementation


## Note
The simulation uses Yahoo Finance API for market data, which has rate limits. The code includes retry mechanisms and delays to handle these limitations.
