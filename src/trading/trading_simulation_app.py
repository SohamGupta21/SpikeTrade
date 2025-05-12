import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from neuro240_project import TradingSNN, SNNTrader, SignalTrader, RandomTrader, simulate_market
import os
from src.market_simulation.neuro240_project import get_visualization_path

def get_stock_data(ticker, start_date, end_date):
    """Get historical stock data from Yahoo Finance"""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def generate_sentiment_data(stock_data):
    """Generate simulated sentiment data"""
    dates = stock_data.index
    n = len(dates)
    ar_params = [0.7]
    ma_params = [0.2]
    ar = np.random.normal(0, 1, n)
    for i in range(1, n):
        ar[i] += ar_params[0] * ar[i-1]
    ma = np.random.normal(0, 0.5, n)
    for i in range(1, n):
        ar[i] += ma_params[0] * ma[i-1]
    sentiment_scores = (ar - np.mean(ar)) / np.std(ar)
    sentiment_data = pd.DataFrame({
        'Date': dates,
        'sentiment_score': sentiment_scores
    })
    return sentiment_data

def run_simulation(market_data, n_snn_traders, n_signal_traders, n_random_traders, days_to_simulate):
    """Run the trading simulation with specified parameters"""
    # Initialize traders
    numeric_cols_count = len(market_data.select_dtypes(include=[np.number]).columns)
    snn_model = TradingSNN(input_size=(20, numeric_cols_count))
    
    traders = (
        [SNNTrader(snn_model) for _ in range(n_snn_traders)] +
        [SignalTrader() for _ in range(n_signal_traders)] +
        [RandomTrader() for _ in range(n_random_traders)]
    )
    
    # Run simulation
    simulated_prices = simulate_market(traders, market_data, days=days_to_simulate)
    return simulated_prices

def plot_trading_analysis(trading_results):
    # ... existing plotting code ...
    
    # Save the plot
    plt.savefig(get_visualization_path('trading', 'trade_analysis.png'))
    plt.close()

def plot_real_time_predictions(predictions):
    # ... existing plotting code ...
    
    # Save the plot
    plt.savefig(get_visualization_path('trading', 'real_time_predictions.png'))
    plt.close()

def main():
    st.title("Interactive Trading Simulation")
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    
    # Date range selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Default to 1 year of data
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(start_date, end_date),
        min_value=end_date - timedelta(days=365*5),
        max_value=end_date
    )
    
    # Trader distribution
    st.sidebar.subheader("Trader Distribution")
    total_traders = st.sidebar.slider("Total Number of Traders", 10, 1000, 300)
    n_snn_traders = st.sidebar.slider("Number of SNN Traders", 0, total_traders, 100)
    remaining_traders = total_traders - n_snn_traders
    n_signal_traders = st.sidebar.slider("Number of Signal Traders", 0, remaining_traders, 100)
    n_random_traders = remaining_traders - n_signal_traders
    
    # Simulation parameters
    st.sidebar.subheader("Simulation Parameters")
    days_to_simulate = st.sidebar.slider("Days to Simulate", 10, 200, 100)
    
    # Display trader distribution
    st.sidebar.write("Trader Distribution:")
    st.sidebar.write(f"- SNN Traders: {n_snn_traders}")
    st.sidebar.write(f"- Signal Traders: {n_signal_traders}")
    st.sidebar.write(f"- Random Traders: {n_random_traders}")
    
    # Get stock data
    ticker = "AAPL"  # Default to Apple stock
    stock_data = get_stock_data(ticker, date_range[0], date_range[1])
    sentiment_data = generate_sentiment_data(stock_data)
    
    # Merge data
    market_data = pd.merge(stock_data, sentiment_data, on='Date', how='inner').dropna()
    
    # Run simulation
    if len(market_data) > 100:
        # Split data for historical comparison
        data_for_simulation_setup = market_data.iloc[:-days_to_simulate]
        actual_last_days_data = market_data.iloc[-days_to_simulate:]
        actual_prices = actual_last_days_data['Close'].values
        actual_dates = actual_last_days_data.index
        
        # Run simulation
        simulated_prices = run_simulation(
            data_for_simulation_setup,
            n_snn_traders,
            n_signal_traders,
            n_random_traders,
            days_to_simulate
        )
        
        # Create interactive plot using Plotly
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_prices,
            name='Actual Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add simulated prices
        if len(simulated_prices) == days_to_simulate + 1:
            fig.add_trace(go.Scatter(
                x=actual_dates,
                y=simulated_prices[1:],
                name='Simulated Price',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title='Historical Simulation vs. Actual Price',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=True,
            height=600
        )
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Actual Price Metrics")
            st.write(f"Starting Price: ${actual_prices[0]:.2f}")
            st.write(f"Ending Price: ${actual_prices[-1]:.2f}")
            st.write(f"Total Change: {((actual_prices[-1] - actual_prices[0]) / actual_prices[0] * 100):.2f}%")
        
        with col2:
            st.subheader("Simulated Price Metrics")
            st.write(f"Starting Price: ${simulated_prices[0]:.2f}")
            st.write(f"Ending Price: ${simulated_prices[-1]:.2f}")
            st.write(f"Total Change: {((simulated_prices[-1] - simulated_prices[0]) / simulated_prices[0] * 100):.2f}%")
        
        # Calculate and display error metrics
        if len(simulated_prices) == days_to_simulate + 1:
            mse = np.mean((actual_prices - simulated_prices[1:]) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual_prices - simulated_prices[1:]))
            
            st.subheader("Error Metrics")
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"Root Mean Squared Error: {rmse:.2f}")
            st.write(f"Mean Absolute Error: {mae:.2f}")
    else:
        st.error("Not enough data to perform simulation comparison")

if __name__ == "__main__":
    main() 