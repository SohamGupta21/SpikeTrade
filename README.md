# Neuro240: Market Simulation and Trading Framework

A comprehensive framework for market simulation and algorithmic trading, featuring multiple trading strategies including a Spiking Neural Network (SNN) implementation.

## Features

- Market simulation with cross-asset interactions
- Multiple trading strategies:
  - Spiking Neural Network (SNN) trader
  - Signal-based trader
  - Random trader
- Advanced portfolio management
- Comprehensive performance metrics
- Enhanced visualization tools
- Data loading and preprocessing utilities

## Project Structure

```
src/
├── market_simulation/
│   ├── market_model.py        # Market simulation core logic
│   ├── liquidity.py          # Liquidity metrics (Kyle's lambda, Amihud)
│   ├── volatility.py         # Volatility calculations and grouping
│   └── visualization.py      # Market visualization functions
├── trading/
│   ├── traders.py           # Trader classes (SNN, Signal, Random)
│   ├── portfolio.py         # Portfolio and position management
│   ├── metrics.py           # Performance metrics calculation
│   └── visualization.py     # Trading visualization functions
└── utils/
    ├── data_loader.py       # Data loading and preprocessing
    └── visualization.py     # Common visualization utilities
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Neuro240.git
cd Neuro240
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Market Simulation

```python
from src.market_simulation.market_model import simulate_market_multiple
from src.utils.data_loader import load_market_data, preprocess_market_data

# Load and preprocess market data
symbols = ['AAPL', 'MSFT', 'GOOGL']
market_data = load_market_data(symbols, '2020-01-01')
processed_data = preprocess_market_data(market_data)

# Group stocks by volatility
from src.market_simulation.volatility import group_stocks_by_volatility
stock_groups = group_stocks_by_volatility(processed_data)

# Simulate market
simulated_prices, simulated_volumes, group_correlations = simulate_market_multiple(
    traders=[],  # Add your traders here
    market_data=processed_data,
    stock_groups=stock_groups,
    days=100
)
```

### Trading Strategies

```python
from src.trading.traders import SNNTrader, SignalTrader, RandomTrader

# Initialize traders
snn_trader = SNNTrader('SNN_Trader', initial_capital=1000000.0)
signal_trader = SignalTrader('Signal_Trader', initial_capital=1000000.0)
random_trader = RandomTrader('Random_Trader', initial_capital=1000000.0)

# Generate orders
orders = snn_trader.update(market_data, current_prices)
```

### Performance Analysis

```python
from src.trading.metrics import calculate_metrics_summary
from src.trading.visualization import plot_equity_curve, plot_drawdown

# Calculate performance metrics
metrics = calculate_metrics_summary(returns, trades, market_returns)

# Visualize performance
equity_curve = plot_equity_curve(returns, market_returns)
drawdown_plot = plot_drawdown(returns)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Spiking Neural Network implementation is based on research in computational neuroscience and market microstructure
- Market simulation incorporates insights from high-frequency trading and market making literature
- Performance metrics follow industry standards for quantitative trading evaluation
