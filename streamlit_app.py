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

from circumference_app.C_autoencoder_build_and_train import autoencoder_build_and_train
network_trained, encoder, decoder, autoencoder, loss_values = autoencoder_build_and_train(seed, data, encoder_arch, decoder_arch, n_bottleneck_neurons)

########################################
# Autoencoded data
########################################

st.subheader('Reconstruct the data with the autoencoder')

from plotly_figures import get_fig_decoded
if network_trained:
	fig_data_decoded = get_fig_decoded(autoencoder, data)
	st.plotly_chart(fig_data_decoded, use_container_width=True)
