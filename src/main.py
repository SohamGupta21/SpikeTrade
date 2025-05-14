"""
Main script to run market simulation and trading strategies.
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
from market_simulation import simulate_market_multiple, plot_group_results, analyze_group_performance
from trading_strategies import SNNTrader, SignalTrader, RandomTrader
from data_loader import DataLoader, preprocess_market_data
from market_analysis import calculate_volatility_metrics, analyze_market_regime
from financial_data import FinancialDataLoader

def main():
    print("\n" + "="*80)
    print("NEURO240 MARKET SIMULATION AND TRADING FRAMEWORK")
    print("="*80 + "\n")

    # =========================================================================
    # DATA LOADING AND PREPROCESSING
    # =========================================================================
    print("\n" + "-"*80)
    print("SECTION 1: DATA LOADING AND PREPROCESSING")
    print("-"*80)
    
    # Define symbols to analyze (reduced set for testing)
    symbols = ['AAPL']  # Just one symbol for testing
    
    print("\nLoading market data...")
    # Initialize data loaders
    market_loader = DataLoader()
    financial_loader = FinancialDataLoader()
    
    # Load and preprocess market data with delay between requests
    market_data = {}
    
    for symbol in symbols:
        try:
            print(f"Loading data for {symbol}...")
            # Get market data
            data = market_loader.get_stock_data(symbol, '2024-01-01', '2024-02-01')
            if data is not None:
                # Get news and earnings data
                news_data, earnings_data = financial_loader.load_additional_data(
                    symbol, '2024-01-01', '2024-02-01'
                )
                
                # Combine all data
                combined_data = financial_loader.combine_market_data(
                    data, news_data, earnings_data
                )
                
                market_data[symbol] = combined_data
            time.sleep(30)  # Increased delay between requests
        except Exception as e:
            print(f"Error loading {symbol}: {str(e)}")
    
    if not market_data:
        print("No market data loaded. Exiting...")
        return
    
    print("\nPreprocessing market data...")
    processed_data = preprocess_market_data(market_data)
    
    if not processed_data:
        print("No processed data available. Exiting...")
        return

    # =========================================================================
    # MARKET ANALYSIS AND GROUPING
    # =========================================================================
    print("\n" + "-"*80)
    print("SECTION 2: MARKET ANALYSIS AND GROUPING")
    print("-"*80)
    
    print("\nAnalyzing market data...")
    # Calculate volatility metrics for each stock
    volatility_metrics = {}
    for symbol, data in processed_data.items():
        metrics = calculate_volatility_metrics(data)
        volatility_metrics[symbol] = metrics
    
    # Group stocks by volatility characteristics
    stock_groups = {
        'high_volatility': [],
        'medium_volatility': [],
        'low_volatility': []
    }
    
    for symbol, metrics in volatility_metrics.items():
        if metrics['annualized_volatility'] > 0.4:  # 40% annualized volatility
            stock_groups['high_volatility'].append(symbol)
        elif metrics['annualized_volatility'] > 0.2:  # 20% annualized volatility
            stock_groups['medium_volatility'].append(symbol)
        else:
            stock_groups['low_volatility'].append(symbol)
    
    print("\nStock groups:")
    for group, stocks in stock_groups.items():
        print(f"{group}: {', '.join(stocks)}")

    # =========================================================================
    # TRADER INITIALIZATION AND CONFIGURATION
    # =========================================================================
    print("\n" + "-"*80)
    print("SECTION 3: TRADER INITIALIZATION AND CONFIGURATION")
    print("-"*80)
    
    print("\nInitializing traders...")
    traders = [
        SNNTrader('SNN_Trader', initial_capital=1000000.0),
        SignalTrader('Signal_Trader', initial_capital=1000000.0),
        RandomTrader('Random_Trader', initial_capital=1000000.0)
    ]

    # =========================================================================
    # MARKET SIMULATION
    # =========================================================================
    print("\n" + "-"*80)
    print("SECTION 4: MARKET SIMULATION")
    print("-"*80)
    
    print("\nRunning market simulation...")
    try:
        simulated_prices, simulated_volumes, group_correlations = simulate_market_multiple(
            traders=traders,
            market_data=processed_data,
            stock_groups=stock_groups,
            days=50  # Reduced simulation period
        )

    # =========================================================================
    # PERFORMANCE ANALYSIS AND VISUALIZATION
    # =========================================================================
        print("\n" + "-"*80)
        print("SECTION 5: PERFORMANCE ANALYSIS AND VISUALIZATION")
        print("-"*80)
        
        print("\nAnalyzing simulation results...")
        # Analyze each group
        for group_name, symbols in stock_groups.items():
            if not symbols:
                continue
                
            print(f"\nAnalyzing {group_name} group...")
            
            # Plot results
            plot_group_results(
                actual_data=processed_data,
                simulated_prices=simulated_prices,
                simulated_volumes=simulated_volumes,
                group_name=group_name,
                dates=simulated_prices[list(simulated_prices.keys())[0]].index
            )
            
            # Calculate performance metrics
            metrics = analyze_group_performance(
                actual_data=processed_data,
                simulated_prices=simulated_prices,
                simulated_volumes=simulated_volumes,
                group_name=group_name
            )
            
            # Print metrics
            print("\nPerformance metrics:")
            for symbol, symbol_metrics in metrics.items():
                print(f"\n{symbol}:")
                for metric_name, value in symbol_metrics.items():
                    print(f"  {metric_name}: {value:.4f}")

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    except Exception as e:
        print("\n" + "-"*80)
        print("ERROR HANDLING")
        print("-"*80)
        print(f"Error during simulation: {str(e)}")

    # =========================================================================
    # COMPLETION
    # =========================================================================
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main() 