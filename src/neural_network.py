"""
Spiking Neural Network implementation for market prediction.
"""

import torch
import torch.nn as nn
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre, MSTDP
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
import numpy as np
from typing import Dict, List, Tuple, Optional

class TradingSNN(Network):
    """
    Trading model using a Spiking Neural Network with LIF neurons and STDP learning
    """
    def __init__(
        self,
        n_market_inputs: int = 10,
        n_news_inputs: int = 1,
        n_earnings_inputs: int = 1,
        n_neurons: int = 100,
        dt: float = 1.0,
        reward_fn: Optional[callable] = None,
        **kwargs
    ):
        super().__init__(dt=dt, **kwargs)
        
        # Create input layers for different data types
        self.market_input = Input(n=n_market_inputs, traces=True)
        self.news_input = Input(n=n_news_inputs, traces=True)
        self.earnings_input = Input(n=n_earnings_inputs, traces=True)
        
        # Create hidden and output layers
        self.hidden_layer = LIFNodes(n=n_neurons, traces=True)
        self.output_layer = LIFNodes(n=2, traces=True)  # Buy/Sell signals
        
        # Add layers to network
        self.add_layer(self.market_input, name="market")
        self.add_layer(self.news_input, name="news")
        self.add_layer(self.earnings_input, name="earnings")
        self.add_layer(self.hidden_layer, name="hidden")
        self.add_layer(self.output_layer, name="output")
        
        # Create connections
        self.market_connection = Connection(
            source=self.market_input,
            target=self.hidden_layer,
            update_rule=PostPre,
            nu=1e-4,
            weight_decay=0.0
        )
        
        self.news_connection = Connection(
            source=self.news_input,
            target=self.hidden_layer,
            update_rule=PostPre,
            nu=1e-4,
            weight_decay=0.0
        )
        
        self.earnings_connection = Connection(
            source=self.earnings_input,
            target=self.hidden_layer,
            update_rule=PostPre,
            nu=1e-4,
            weight_decay=0.0
        )
        
        self.hidden_connection = Connection(
            source=self.hidden_layer,
            target=self.output_layer,
            update_rule=PostPre,
            nu=1e-4,
            weight_decay=0.0
        )
        
        # Add connections to network
        self.add_connection(self.market_connection, source="market", target="hidden")
        self.add_connection(self.news_connection, source="news", target="hidden")
        self.add_connection(self.earnings_connection, source="earnings", target="hidden")
        self.add_connection(self.hidden_connection, source="hidden", target="output")
        
        # Add monitors
        self.add_monitor(Monitor(self.market_input, ["s", "v"], time=1000), "market")
        self.add_monitor(Monitor(self.news_input, ["s", "v"], time=1000), "news")
        self.add_monitor(Monitor(self.earnings_input, ["s", "v"], time=1000), "earnings")
        self.add_monitor(Monitor(self.hidden_layer, ["s", "v"], time=1000), "hidden")
        self.add_monitor(Monitor(self.output_layer, ["s", "v"], time=1000), "output")
        
        # Store reward function
        self.reward_fn = reward_fn
    
    def forward(
        self,
        market_data: torch.Tensor,
        news_data: torch.Tensor,
        earnings_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            market_data (torch.Tensor): Market data input
            news_data (torch.Tensor): News sentiment input
            earnings_data (torch.Tensor): Earnings indicator input
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output spikes and voltages
        """
        # Set input spikes
        self.market_input.s = market_data
        self.news_input.s = news_data
        self.earnings_input.s = earnings_data
        
        # Run network
        self.run(time=1, input_monitor="market")
        
        # Get output
        spikes = self.output_layer.s
        voltages = self.output_layer.v
        
        return spikes, voltages
    
    def update_weights(self, reward: float):
        """
        Update network weights based on reward
        
        Args:
            reward (float): Reward signal
        """
        if self.reward_fn is not None:
            self.reward_fn(reward)
    
    def get_prediction(self, spikes: torch.Tensor) -> str:
        """
        Convert output spikes to trading signal
        
        Args:
            spikes (torch.Tensor): Output spikes
            
        Returns:
            str: Trading signal ('buy', 'sell', or 'hold')
        """
        buy_spikes = spikes[0].sum()
        sell_spikes = spikes[1].sum()
        
        if buy_spikes > sell_spikes and buy_spikes > 0:
            return 'buy'
        elif sell_spikes > buy_spikes and sell_spikes > 0:
            return 'sell'
        else:
            return 'hold'
    
    def plot_activity(self, save_path: Optional[str] = None):
        """
        Plot network activity
        
        Args:
            save_path (Optional[str]): Path to save plots
        """
        # Plot input spikes
        plot_spikes(self.monitors["market"].get("s"), save_path=f"{save_path}_market_spikes.png" if save_path else None)
        plot_spikes(self.monitors["news"].get("s"), save_path=f"{save_path}_news_spikes.png" if save_path else None)
        plot_spikes(self.monitors["earnings"].get("s"), save_path=f"{save_path}_earnings_spikes.png" if save_path else None)
        
        # Plot hidden layer activity
        plot_spikes(self.monitors["hidden"].get("s"), save_path=f"{save_path}_hidden_spikes.png" if save_path else None)
        plot_voltages(self.monitors["hidden"].get("v"), save_path=f"{save_path}_hidden_voltages.png" if save_path else None)
        
        # Plot output activity
        plot_spikes(self.monitors["output"].get("s"), save_path=f"{save_path}_output_spikes.png" if save_path else None)
        plot_voltages(self.monitors["output"].get("v"), save_path=f"{save_path}_output_voltages.png" if save_path else None) 