"""
Main script to run market simulation and trading strategies.
"""

import numpy as np
import pandas as pd
import time
from market_simulation.market_model import simulate_market_multiple
from market_simulation.volatility import group_stocks_by_volatility
from trading.traders import SNNTrader, SignalTrader, RandomTrader
from trading.metrics import calculate_metrics_summary
from trading.visualization import plot_equity_curve, plot_drawdown
from utils.data_loader import load_market_data, preprocess_market_data

def main():
    # Define symbols to analyze (reduced set)
    symbols = ['AAPL', 'MSFT']
    
    print("Loading market data...")
    # Load and preprocess market data with delay between requests
    market_data = {}
    for symbol in symbols:
        try:
            print(f"Loading data for {symbol}...")
            data = load_market_data([symbol], '2023-01-01')
            if symbol in data:
                market_data[symbol] = data[symbol]
            time.sleep(2)  # Add delay between requests
        except Exception as e:
            print(f"Error loading {symbol}: {str(e)}")
    
    if not market_data:
        print("No market data loaded. Exiting...")
        return
    
    print("Preprocessing market data...")
    processed_data = preprocess_market_data(market_data)
    
    if not processed_data:
        print("No processed data available. Exiting...")
        return
    
    print("Grouping stocks by volatility...")
    stock_groups = group_stocks_by_volatility(processed_data)
    
    if not stock_groups:
        print("No stock groups created. Exiting...")
        return
    
    print("Initializing traders...")
    traders = [
        SNNTrader('SNN_Trader', initial_capital=1000000.0),
        SignalTrader('Signal_Trader', initial_capital=1000000.0),
        RandomTrader('Random_Trader', initial_capital=1000000.0)
    ]
    
    print("Running market simulation...")
    try:
        simulated_prices, simulated_volumes, group_correlations = simulate_market_multiple(
            traders=traders,
            market_data=processed_data,
            stock_groups=stock_groups,
            days=50  # Reduced simulation period
        )
        
        print("\nSimulation Results:")
        print("------------------")
        
        # Calculate and display performance metrics for each trader
        for trader in traders:
            print(f"\n{trader.name} Performance:")
            print("-" * 20)
            
            # Calculate returns
            returns = np.array([trade['price'] * trade['quantity'] for trade in trader.trades])
            if len(returns) > 0:
                # Calculate market returns for comparison
                market_returns = np.array([
                    processed_data[symbol]['Returns_Close_' + symbol].iloc[-50:].mean()
                    for symbol in symbols
                ]).mean()
                
                # Calculate metrics
                metrics = calculate_metrics_summary(returns, trader.trades, market_returns)
                
                # Display key metrics
                print(f"Total Return: {metrics['total_return']:.2%}")
                print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
                print(f"Win Rate: {metrics['win_rate']:.2%}")
                
                # Generate performance plots
                equity_curve = plot_equity_curve(returns, market_returns)
                drawdown_plot = plot_drawdown(returns)
                
                # Save plots
                equity_curve.write_html(f"visualizations/{trader.name}_equity_curve.html")
                drawdown_plot.write_html(f"visualizations/{trader.name}_drawdown.html")
            else:
                print("No trades executed")
    except Exception as e:
        print(f"Error during simulation: {str(e)}")

if __name__ == "__main__":
    main() 