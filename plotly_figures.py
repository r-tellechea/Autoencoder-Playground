import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

def get_fig_data(data: torch.Tensor) -> go.Figure:
	fig_data = px.scatter(
		data_frame=pd.DataFrame(
			data=data.cpu().numpy(),
			columns=['x', 'y']
		),
		x='x',
		y='y',
	)
	
	fig_data.update_layout(
		margin=dict(l=0, r=0, t=0, b=0),
		height=700,
		width=700,
	)

	return fig_data

@st.cache_resource
def get_fig_nn(architecture: tuple[int]) -> go.Figure:
	from neural_network_plot import NeuralNetworkPlot
	return NeuralNetworkPlot(architecture).fig()

def get_fig_loss(loss_values: list[float], epochs: int) -> go.Figure:
	return px.line(
		data_frame=(
			pd.DataFrame(
				data=loss_values, 
				columns=['loss']
			)
			.assign(epoch=range(1, epochs+1))
		),
		x='epoch',
		y='loss',
		title='Loss value over epochs',
	)

def get_fig_decoded(autoencoder: torch.nn.Module, data: torch.Tensor) -> go.Figure:
	fig_data = get_fig_data(data)

	fig_decoded = px.scatter(
		data_frame=pd.DataFrame(
			data=autoencoder(data).detach().cpu().numpy(),
			columns=['x', 'y']
		),
		x='x',
		y='y',
		color_discrete_sequence=['salmon']
	)

	fig_data_decoded = go.Figure(data=(fig_data.data + fig_decoded.data))

	fig_data_decoded.update_layout(
		margin=dict(l=0, r=0, t=0, b=0),
		height=700,
		width=700,
	)

	return fig_data_decoded

