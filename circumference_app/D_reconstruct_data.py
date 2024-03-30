import torch
from torch import nn
import streamlit as st
from streamlit_pills import pills

from plotly_figures import get_fig_decoded, get_fig_decoded_colors, get_fig_theta_vs_encoded
from circumference_app.text import text

def reconstruct_data(data: torch.Tensor):

	st.markdown(text.reconstruct_data_1)
	
	if st.session_state.train_info.trained:	
		fig_data_decoded = get_fig_decoded(st.session_state.train_info.autoencoder, data)
		st.plotly_chart(fig_data_decoded, use_container_width=True)
		
	st.markdown(text.reconstruct_data_2)
		
	if st.session_state.train_info.trained:
		tab_circumference, tab_scatter = st.tabs(['Data reconstruction', 'Theta vs Encoded'])
		with tab_circumference:
			fig_data_decoded_colors = get_fig_decoded_colors(
				st.session_state.train_info.encoder, 
				st.session_state.train_info.autoencoder, 
				data
			)
			st.plotly_chart(fig_data_decoded_colors, use_container_width=True)
		with tab_scatter:
			# TODO: ¿Qué pasa aquí si hay dos neuronas en el bottleneck?
			color_column = pills(
				label='Color by',
				options=['Theta', 'Encoded', 'No color'],
				index=1,
				key='color scatter pills'
			)
			fig_theta_vs_encoded = get_fig_theta_vs_encoded(
				st.session_state.train_info.encoder, 
				data, 
				color_column
			)
			st.plotly_chart(fig_theta_vs_encoded, use_container_width=True)
