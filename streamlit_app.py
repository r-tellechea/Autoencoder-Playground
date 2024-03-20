import torch
from torch import nn
import streamlit as st

st.set_page_config(
	page_title='Autoencoder', 
	page_icon='⚗️', 
	layout='centered', 
	initial_sidebar_state='auto'
)

# Title
st.title('Autoencoder')

# Seed
seed = st.number_input(
	label='Set the manual seed:', 
	key='seed', 
	value=31415, 
	min_value=0, 
	max_value=1_000_000
)

st.subheader('Create the data')

########################################
# Data configuration
########################################

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

# Generate the data
from data import generate_data
data = generate_data(n_points, mean, std, seed)

from plotly_figures import get_fig_data
fig_data = get_fig_data(data)
st.plotly_chart(fig_data, use_container_width=True)

########################################
# Neural Networks Configuration
########################################

st.subheader('Autoencoder architecture')
col_encoder, col_decoder, col_bottleneck = st.columns((2,2,1))

with col_encoder:
	n_encoder_hidden_layers = st.slider(
			label="Number of encoder's hidden layers",
			key='n_encoder_hidden_layers',
			min_value=0,
			max_value=4,
			value=3,
			step=1,
		)

with col_decoder:	
	n_decoder_hidden_layers = st.slider(
			label="Number of decoder's hidden layers",
			key='n_decoder_hidden_layers',
			min_value=0,
			max_value=4,
			value=3,
			step=1,
		)

with col_bottleneck:
	n_bottleneck_neurons = st.radio(
		label='Bottleneck neurons',
		key='bottleneck_neurons',
		options=[1,2],
	)

list_encoder_layer_size = [1] * n_encoder_hidden_layers
list_decoder_layer_size = [1] * n_decoder_hidden_layers

with col_encoder:
	for layer in range(n_encoder_hidden_layers):
		list_encoder_layer_size[layer] = st.number_input(
			label=f"Neurons in encoder's layer {layer+1}",
			key=f'encoder_{layer}_size',
			min_value=1,
			max_value=8,
			value=8,
		)

with col_decoder:
	for layer in range(n_decoder_hidden_layers):
		list_decoder_layer_size[layer] = st.number_input(
			label=f"Neurons in decoder's layer {layer+1}",
			key=f'decoder_{layer}_size',
			min_value=1,
			max_value=8,
			value=8,
		)

encoder_arch = tuple(list_encoder_layer_size)
decoder_arch = tuple(list_decoder_layer_size)

# TODO: Temporal.
st.text(f'Autoencoder architecture: {(2,) + encoder_arch + (n_bottleneck_neurons,) + decoder_arch + (2,)}')

########################################
# Train the network
########################################

st.subheader('Train the networks')

from autoencoder import get_autoencoder
from plotly_figures import get_fig_loss

# Training configuration.
col_epochs, col_lr, col_train_button = st.columns((4, 1, 1))

with col_epochs:
	epochs = st.slider(
		label="Epochs",
		key='epochs',
		min_value=1,
		max_value=1000,
		value=500,
		step=1,
	)

with col_lr:
	lr = st.radio(
		label='Learning Rate',
		key='learning_rate',
		options=[0.001, 0.01, 0.1, 1],
		index=1
	)
with col_train_button:
	train_button = st.button(label='TRAIN!', key='train_button')

network_trained = False

# Create and train the neural networks.
loss_plot_placeholder = st.empty()

if train_button:
	with st.spinner('Training autoencoder...'):
		encoder, decoder, autoencoder, loss_values = get_autoencoder(
			encoder_arch, 
			decoder_arch, 
			n_bottleneck_neurons,
			data,
			epochs,
			lr,
		)
	st.toast('Autoencoder trained!')
	fig_loss = get_fig_loss(loss_values, epochs)
	loss_plot_placeholder.plotly_chart(fig_loss, use_container_width=True)
	network_trained = True

########################################
# Autoencoded data
########################################

st.subheader('Reconstruct the data with the autoencoder')

from plotly_figures import get_fig_decoded
if network_trained:
	fig_data_decoded = get_fig_decoded(autoencoder, data)
	st.plotly_chart(fig_data_decoded, use_container_width=True)
