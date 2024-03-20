import torch
from torch import nn
import streamlit as st

from plotly_figures import get_fig_decoded

def reconstruct_data(data: torch.Tensor, autoencoder: nn.Module, network_trained: bool):

	if network_trained:
		fig_data_decoded = get_fig_decoded(autoencoder, data)
		st.plotly_chart(fig_data_decoded, use_container_width=True)
