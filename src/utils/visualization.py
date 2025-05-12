"""
Common visualization utilities for market and trading analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_time_series(data: pd.Series, title: str = 'Time Series',
                    xlabel: str = 'Time', ylabel: str = 'Value'):
    """
    Plot time series data with enhanced visualization.
    
    Args:
        data: Time series data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        name='Value',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        width=800,
        showlegend=True
    )
    
    return fig

def plot_multiple_time_series(data_dict: Dict[str, pd.Series],
                            title: str = 'Multiple Time Series',
                            xlabel: str = 'Time', ylabel: str = 'Value'):
    """
    Plot multiple time series with enhanced visualization.
    
    Args:
        data_dict: Dictionary mapping names to time series
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig = go.Figure()
    
    for name, series in data_dict.items():
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        width=800,
        showlegend=True
    )
    
    return fig

def plot_heatmap(data: pd.DataFrame, title: str = 'Heatmap',
                xlabel: str = 'X', ylabel: str = 'Y'):
    """
    Plot heatmap with enhanced visualization.
    
    Args:
        data: DataFrame to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=600,
        width=800
    )
    
    return fig

def plot_distribution(data: np.ndarray, title: str = 'Distribution',
                     xlabel: str = 'Value', ylabel: str = 'Frequency'):
    """
    Plot distribution with enhanced visualization.
    
    Args:
        data: Array of values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data,
        name='Distribution',
        nbinsx=50
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        width=800,
        showlegend=True
    )
    
    return fig

def plot_scatter(x: np.ndarray, y: np.ndarray,
                title: str = 'Scatter Plot',
                xlabel: str = 'X', ylabel: str = 'Y'):
    """
    Plot scatter plot with enhanced visualization.
    
    Args:
        x: X-axis values
        y: Y-axis values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Data Points'
    ))
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=x,
        y=p(x),
        mode='lines',
        name='Trend Line'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        width=800,
        showlegend=True
    )
    
    return fig

def plot_box_plot(data: pd.DataFrame, title: str = 'Box Plot',
                 xlabel: str = 'Category', ylabel: str = 'Value'):
    """
    Plot box plot with enhanced visualization.
    
    Args:
        data: DataFrame to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig = go.Figure()
    
    for column in data.columns:
        fig.add_trace(go.Box(
            y=data[column],
            name=column
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        width=800,
        showlegend=True
    )
    
    return fig

def plot_correlation_matrix(correlation_matrix: pd.DataFrame,
                          title: str = 'Correlation Matrix'):
    """
    Plot correlation matrix with enhanced visualization.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        title: Plot title
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title=title,
        height=600,
        width=800
    )
    
    return fig

def plot_rolling_statistics(data: pd.Series, window: int = 20,
                          title: str = 'Rolling Statistics'):
    """
    Plot rolling statistics with enhanced visualization.
    
    Args:
        data: Time series data
        window: Rolling window size
        title: Plot title
    """
    # Calculate rolling statistics
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    # Create figure
    fig = go.Figure()
    
    # Plot original data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        name='Original',
        line=dict(width=1)
    ))
    
    # Plot rolling mean
    fig.add_trace(go.Scatter(
        x=rolling_mean.index,
        y=rolling_mean.values,
        name=f'{window}-day Moving Average',
        line=dict(width=2)
    ))
    
    # Plot rolling standard deviation
    fig.add_trace(go.Scatter(
        x=rolling_std.index,
        y=rolling_std.values,
        name=f'{window}-day Standard Deviation',
        line=dict(width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        height=400,
        width=800,
        showlegend=True
    )
    
    return fig 