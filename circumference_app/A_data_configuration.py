import torch
import streamlit as st

from data import generate_data
from plotly_figures import get_fig_data

def data_configuration(seed: int) -> torch.Tensor:

	column_n_points, column_mean, column_std = st.columns(3)

	with column_n_points:
		n_points = st.number_input(
			label='Number of points', 
			key='n_points',
			min_value=100, 
			max_value=10_000, 
			value=1000,
		)

	with column_mean:
		mean = st.slider(
			label='Radious',
			key='mean',
			min_value=1.,
			max_value=20.,
			value=10.,
			step=0.1,
		)

	with column_std:
		std = st.slider(
			label='Deviation',
			key='std',
			min_value=0.,
			max_value=1.,
			value=0.5,
			step=0.01,
		)

	data = generate_data(n_points, mean, std, seed)

	fig_data = get_fig_data(data)
	st.plotly_chart(fig_data, use_container_width=True)

	return data
