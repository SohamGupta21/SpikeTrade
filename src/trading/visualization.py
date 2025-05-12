"""
Trading visualization functions for analyzing strategy performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_equity_curve(returns: np.ndarray, benchmark_returns: np.ndarray = None,
                     title: str = 'Equity Curve'):
    """
    Plot equity curve with optional benchmark comparison.
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns (optional)
        title: Plot title
    """
    # Calculate cumulative returns
    strategy_equity = (1 + returns).cumprod()
    
    # Create figure
    fig = go.Figure()
    
    # Plot strategy equity
    fig.add_trace(go.Scatter(
        y=strategy_equity,
        name='Strategy',
        line=dict(width=2)
    ))
    
    # Plot benchmark if provided
    if benchmark_returns is not None:
        benchmark_equity = (1 + benchmark_returns).cumprod()
        fig.add_trace(go.Scatter(
            y=benchmark_equity,
            name='Benchmark',
            line=dict(width=2, dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Equity',
        height=400,
        width=800,
        showlegend=True
    )
    
    return fig

def plot_drawdown(returns: np.ndarray, title: str = 'Drawdown Analysis'):
    """
    Plot drawdown analysis.
    
    Args:
        returns: Array of returns
        title: Plot title
    """
    # Calculate cumulative returns and drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = pd.Series(cumulative_returns).expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    
    # Create figure
    fig = go.Figure()
    
    # Plot drawdown
    fig.add_trace(go.Scatter(
        y=drawdowns * 100,
        name='Drawdown',
        line=dict(width=2, color='red')
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Drawdown (%)',
        height=400,
        width=800,
        showlegend=True
    )
    
    return fig

def plot_monthly_returns(returns: np.ndarray, title: str = 'Monthly Returns'):
    """
    Plot monthly returns heatmap.
    
    Args:
        returns: Array of returns
        title: Plot title
    """
    # Convert returns to DataFrame with datetime index
    returns_df = pd.Series(returns, index=pd.date_range(start='2020-01-01', periods=len(returns)))
    
    # Resample to monthly returns
    monthly_returns = returns_df.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create monthly returns matrix
    monthly_matrix = monthly_returns.values.reshape(-1, 12)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=monthly_matrix * 100,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=monthly_returns.index.year.unique(),
        colorscale='RdBu',
        zmid=0
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Month',
        yaxis_title='Year',
        height=400,
        width=800
    )
    
    return fig

def plot_trade_analysis(trades: List[Dict], title: str = 'Trade Analysis'):
    """
    Plot trade analysis with various metrics.
    
    Args:
        trades: List of trade dictionaries
        title: Plot title
    """
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculate trade metrics
    trade_returns = trades_df['quantity'] * trades_df['price']
    winning_trades = trade_returns[trade_returns > 0]
    losing_trades = trade_returns[trade_returns < 0]
    
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=('Trade Returns Distribution',
                       'Win/Loss Ratio',
                       'Cumulative P&L',
                       'Trade Size Distribution')
    )
    
    # Plot trade returns distribution
    fig.add_trace(
        go.Histogram(x=trade_returns, name='Returns'),
        row=1, col=1
    )
    
    # Plot win/loss ratio
    fig.add_trace(
        go.Bar(
            x=['Winning Trades', 'Losing Trades'],
            y=[len(winning_trades), len(losing_trades)],
            name='Trade Count'
        ),
        row=1, col=2
    )
    
    # Plot cumulative P&L
    cumulative_pnl = trade_returns.cumsum()
    fig.add_trace(
        go.Scatter(y=cumulative_pnl, name='Cumulative P&L'),
        row=2, col=1
    )
    
    # Plot trade size distribution
    fig.add_trace(
        go.Histogram(x=abs(trades_df['quantity']), name='Trade Size'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        width=1000,
        showlegend=True
    )
    
    return fig

def plot_risk_metrics(returns: np.ndarray, benchmark_returns: np.ndarray = None,
                     title: str = 'Risk Metrics'):
    """
    Plot risk metrics analysis.
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns (optional)
        title: Plot title
    """
    # Calculate rolling metrics
    window = 20
    rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
    rolling_sharpe = pd.Series(returns).rolling(window).mean() / pd.Series(returns).rolling(window).std() * np.sqrt(252)
    
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=('Rolling Volatility',
                       'Rolling Sharpe Ratio')
    )
    
    # Plot rolling volatility
    fig.add_trace(
        go.Scatter(y=rolling_vol * 100, name='Strategy Volatility'),
        row=1, col=1
    )
    
    if benchmark_returns is not None:
        benchmark_vol = pd.Series(benchmark_returns).rolling(window).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(y=benchmark_vol * 100, name='Benchmark Volatility'),
            row=1, col=1
        )
    
    # Plot rolling Sharpe ratio
    fig.add_trace(
        go.Scatter(y=rolling_sharpe, name='Strategy Sharpe'),
        row=2, col=1
    )
    
    if benchmark_returns is not None:
        benchmark_sharpe = (pd.Series(benchmark_returns).rolling(window).mean() /
                          pd.Series(benchmark_returns).rolling(window).std() * np.sqrt(252))
        fig.add_trace(
            go.Scatter(y=benchmark_sharpe, name='Benchmark Sharpe'),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        width=800,
        showlegend=True
    )
    
    return fig

def plot_correlation_analysis(returns: np.ndarray, benchmark_returns: np.ndarray,
                            title: str = 'Correlation Analysis'):
    """
    Plot correlation analysis between strategy and benchmark.
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns
        title: Plot title
    """
    # Calculate rolling correlation
    window = 20
    rolling_corr = pd.Series(returns).rolling(window).corr(pd.Series(benchmark_returns))
    
    # Create figure
    fig = go.Figure()
    
    # Plot rolling correlation
    fig.add_trace(go.Scatter(
        y=rolling_corr,
        name='Rolling Correlation',
        line=dict(width=2)
    ))
    
    # Add horizontal line at 0
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Correlation',
        height=400,
        width=800,
        showlegend=True
    )
    
    return fig 