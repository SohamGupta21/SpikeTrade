"""
Market visualization functions for analyzing simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_price_evolution(simulated_prices: Dict, market_data: Dict, stock_groups: Dict):
    """
    Plot price evolution for each stock group with enhanced visualization.
    
    Args:
        simulated_prices: Dictionary of simulated prices by group and stock
        market_data: Dictionary of actual market data
        stock_groups: Dictionary of stock groups
    """
    # Create subplots for each group
    n_groups = len(stock_groups)
    fig = make_subplots(
        rows=n_groups,
        cols=1,
        subplot_titles=[f'Group {i+1}' for i in range(n_groups)],
        vertical_spacing=0.1
    )
    
    # Plot each group
    for i, (group_name, stocks) in enumerate(stock_groups.items(), 1):
        for stock in stocks:
            if stock in simulated_prices[group_name]:
                # Plot simulated prices
                sim_prices = simulated_prices[group_name][stock]
                fig.add_trace(
                    go.Scatter(
                        y=sim_prices,
                        name=f'{stock} (Simulated)',
                        line=dict(width=1),
                        opacity=0.7
                    ),
                    row=i,
                    col=1
                )
                
                # Plot actual prices if available
                if stock in market_data:
                    actual_prices = market_data[stock][f'Close_{stock}'].values
                    fig.add_trace(
                        go.Scatter(
                            y=actual_prices,
                            name=f'{stock} (Actual)',
                            line=dict(width=1, dash='dash'),
                            opacity=0.7
                        ),
                        row=i,
                        col=1
                    )
    
    # Update layout
    fig.update_layout(
        height=300 * n_groups,
        showlegend=True,
        title_text='Price Evolution by Volatility Group',
        xaxis_title='Time',
        yaxis_title='Price'
    )
    
    return fig

def plot_volume_analysis(simulated_volumes: Dict, market_data: Dict, stock_groups: Dict):
    """
    Plot volume analysis with enhanced visualization.
    
    Args:
        simulated_volumes: Dictionary of simulated volumes by group and stock
        market_data: Dictionary of actual market data
        stock_groups: Dictionary of stock groups
    """
    # Create subplots for each group
    n_groups = len(stock_groups)
    fig = make_subplots(
        rows=n_groups,
        cols=1,
        subplot_titles=[f'Group {i+1}' for i in range(n_groups)],
        vertical_spacing=0.1
    )
    
    # Plot each group
    for i, (group_name, stocks) in enumerate(stock_groups.items(), 1):
        for stock in stocks:
            if stock in simulated_volumes[group_name]:
                # Plot simulated volumes
                sim_volumes = simulated_volumes[group_name][stock]
                fig.add_trace(
                    go.Scatter(
                        y=sim_volumes,
                        name=f'{stock} (Simulated)',
                        line=dict(width=1),
                        opacity=0.7
                    ),
                    row=i,
                    col=1
                )
                
                # Plot actual volumes if available
                if stock in market_data:
                    actual_volumes = market_data[stock][f'Volume_{stock}'].values
                    fig.add_trace(
                        go.Scatter(
                            y=actual_volumes,
                            name=f'{stock} (Actual)',
                            line=dict(width=1, dash='dash'),
                            opacity=0.7
                        ),
                        row=i,
                        col=1
                    )
    
    # Update layout
    fig.update_layout(
        height=300 * n_groups,
        showlegend=True,
        title_text='Volume Analysis by Volatility Group',
        xaxis_title='Time',
        yaxis_title='Volume'
    )
    
    return fig

def plot_correlation_heatmap(correlation_matrix: pd.DataFrame):
    """
    Plot correlation heatmap with enhanced visualization.
    
    Args:
        correlation_matrix: DataFrame containing correlation matrix
    """
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0
    ))
    
    # Update layout
    fig.update_layout(
        title='Cross-Asset Correlation Matrix',
        xaxis_title='Assets',
        yaxis_title='Assets',
        height=600,
        width=800
    )
    
    return fig

def plot_volatility_regime(returns: np.ndarray, window: int = 60):
    """
    Plot volatility regime analysis with enhanced visualization.
    
    Args:
        returns: Array of returns
        window: Rolling window size
    """
    # Calculate rolling volatility
    volatility = pd.Series(returns).rolling(
        window=window,
        min_periods=1,
        win_type='gaussian'
    ).std(std=3)
    
    # Calculate regime thresholds
    vol_percentiles = volatility.quantile([0.33, 0.66])
    
    # Create figure
    fig = go.Figure()
    
    # Plot volatility
    fig.add_trace(go.Scatter(
        y=volatility,
        name='Volatility',
        line=dict(width=2)
    ))
    
    # Add regime thresholds
    fig.add_hline(
        y=vol_percentiles[0.33],
        line_dash="dash",
        line_color="green",
        annotation_text="Low/Medium Threshold"
    )
    fig.add_hline(
        y=vol_percentiles[0.66],
        line_dash="dash",
        line_color="red",
        annotation_text="Medium/High Threshold"
    )
    
    # Update layout
    fig.update_layout(
        title='Volatility Regime Analysis',
        xaxis_title='Time',
        yaxis_title='Volatility',
        height=400,
        width=800
    )
    
    return fig 