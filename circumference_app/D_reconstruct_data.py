import torch
from torch import nn
import streamlit as st
from streamlit_pills import pills

from plotly_figures import get_fig_decoded, get_fig_decoded_colors, get_fig_theta_vs_encoded
from circumference_app.text import text

def reconstruct_data(data: torch.Tensor, autoencoder: nn.Module, encoder: nn.Module, network_trained: bool):

	st.markdown(text.reconstruct_data_1)
	
	if network_trained:	
		fig_data_decoded = get_fig_decoded(autoencoder, data)
		st.plotly_chart(fig_data_decoded, use_container_width=True)
		
	st.markdown(text.reconstruct_data_2)
		
	if network_trained:
		tab_circumference, tab_scatter = st.tabs(['Data reconstruction', 'Theta vs Encoded'])
		with tab_circumference:
			fig_data_decoded_colors = get_fig_decoded_colors(encoder, autoencoder, data)
			st.plotly_chart(fig_data_decoded_colors, use_container_width=True)
		with tab_scatter:
			color_column = pills(
				label='Color by',
				options=['Theta', 'Encoded', 'No color'],
				index=1,
				key='color scatter pills'
			)
			fig_theta_vs_encoded = get_fig_theta_vs_encoded(encoder, data, color_column)
			st.plotly_chart(fig_theta_vs_encoded, use_container_width=True)
