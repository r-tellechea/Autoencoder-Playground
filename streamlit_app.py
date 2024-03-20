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

########################################
# Data configuration
########################################

st.subheader('Create the data')

from circumference_app.A_data_configuration import data_configuration
data = data_configuration(seed)

########################################
# Neural Networks Configuration
########################################

st.subheader('Autoencoder architecture')

from circumference_app.B_autoencoder_configuration import autoencoder_configuration
encoder_arch, decoder_arch, n_bottleneck_neurons = autoencoder_configuration(seed, data)

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
