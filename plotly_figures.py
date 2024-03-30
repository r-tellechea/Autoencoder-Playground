import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import numpy as np

from neural_network_plot import NeuralNetworkPlot, update_nn_fig_hoverlabel

color_theme_plot_1 = '#2196f3'
color_theme_plot_2 = '#ff8484'
color_theme = '#5e60ce'

def angle_from_x_y(x: np.array, y: np.array) -> np.array:
	return np.pi + 2 * np.arctan(-y / (-x + (x**2 + y**2)**0.5))

def get_fig_data(data: torch.Tensor) -> go.Figure:
	fig_data = px.scatter(
		data_frame=pd.DataFrame(
			data=data.cpu().numpy(),
			columns=['x', 'y']
		),
		x='x',
		y='y',
		color_discrete_sequence=[color_theme_plot_1],
	)
	
	fig_data.update_layout(
		margin=dict(l=0, r=0, t=0, b=0),
		height=700,
		width=700,
	)

	return fig_data

@st.cache_resource
def get_fig_nn(encoder_arch: tuple[int], decoder_arch: tuple[int], n_bottleneck_neurons: int) -> go.Figure:
	architecture = (2,) + encoder_arch + (n_bottleneck_neurons,) + decoder_arch + (2,)
	fig = NeuralNetworkPlot(architecture).fig()
	fig = update_nn_fig_hoverlabel(fig, encoder_arch, decoder_arch, n_bottleneck_neurons)
	return fig


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
		color_discrete_sequence=[color_theme],
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
		color_discrete_sequence=[color_theme_plot_2],
	)

	fig_data_decoded = go.Figure(data=(fig_data.data + fig_decoded.data))

	fig_data_decoded.update_layout(
		margin=dict(l=0, r=0, t=0, b=0),
		height=700,
		width=700,
	)

	return fig_data_decoded

def get_fig_decoded_colors(encoder: torch.nn.Module, autoencoder: torch.nn.Module, data: torch.Tensor) -> go.Figure:
	fig_data = px.scatter(
		data_frame=(
			pd.DataFrame(
				data=data.cpu().numpy(),
				columns=['x', 'y']
			)
			.assign(theta = lambda df : angle_from_x_y(df.x, df.y))
		),
		x='x',
		y='y',
		color='theta',
		color_continuous_scale='viridis'
	)

	fig_decoded = px.scatter(
		data_frame=(
			pd.DataFrame(
				data=autoencoder(data).detach().cpu().numpy(),
				columns=['x', 'y']
			)
			.assign(encoded = encoder(data).cpu().detach().numpy())
		),
		x='x',
		y='y',
		color='encoded',
		color_continuous_scale='electric'
	)

	fig_decoded_colors = go.Figure(data=(fig_data.data + fig_decoded.data))

	fig_decoded_colors.layout.coloraxis = fig_data.layout.coloraxis
	fig_decoded_colors.layout.coloraxis2 = fig_decoded.layout.coloraxis
	fig_decoded_colors.data[1].marker = {
		'color' : (
			pd.DataFrame(
				data=autoencoder(data).detach().cpu().numpy(),
				columns=['x', 'y']
			)
			.assign(encoded = encoder(data).cpu().detach().numpy())
			.encoded
		),
		'coloraxis' : 'coloraxis2'
	}
	fig_decoded_colors.layout.coloraxis2.colorbar.x = 1.1

	fig_decoded_colors.update_layout(
		margin=dict(l=0, r=0, t=0, b=0),
		height=550,
		width=700,
	)

	return fig_decoded_colors


def get_fig_theta_vs_encoded(encoder: torch.nn.Module, data: torch.Tensor, color_column: str='Theta') -> go.Figure:
	if color_column == 'Theta':
		color_scale = 'viridis'
	elif color_column == 'Encoded':
		color_scale = 'electric'
	else:
		color_column = 'No color'
		color_scale = None

	df = (
		pd.DataFrame(
			data=data.cpu().numpy(),
			columns=['x', 'y']
		)
		.assign(Theta = lambda df : angle_from_x_y(df.x, df.y))
		.assign(Encoded = encoder(data).cpu().detach().numpy())
		.assign(**{'No color' : 1})
	)
	
	fig_theta_vs_encoded = px.scatter(
		data_frame=df,
		x='Theta',
		y='Encoded',
		color=color_column,
		color_continuous_scale=color_scale,
		color_discrete_sequence=([color_theme_plot_1] if color_column == 'No color' else None)
	)
	
	fig_theta_vs_encoded.update_layout(
		margin=dict(l=0, r=0, t=0, b=0),
		height=550,
		width=700,
	)
	
	if color_column == 'No color':
		fig_theta_vs_encoded.update(layout_coloraxis_showscale=False)
	
	return fig_theta_vs_encoded
