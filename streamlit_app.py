import streamlit as st
st.set_page_config(
	page_title='Autoencoder', 
	page_icon='⚗️', 
	layout='centered', 
	initial_sidebar_state='auto'
)

# Text
from circumference_app.text import text

########################################
# Intro
########################################
st.title('Autoencoder')

st.markdown(text.intro)

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

st.subheader('Crear los datos')

st.markdown(text.circumference)

from circumference_app.A_data_configuration import data_configuration
data = data_configuration(seed)

########################################
# Neural Networks Configuration
########################################

st.subheader('Arquitectura del autoencoder')

from circumference_app.B_autoencoder_configuration import autoencoder_configuration
from get_arch_code import get_code_autoencoder

st.markdown(text.autoencoder_architecture)
encoder_arch, decoder_arch, n_bottleneck_neurons = autoencoder_configuration(seed, data)
st.markdown(text.autoencoder_architecture_code)
st.code(
	body=get_code_autoencoder(
		encoder_arch=( (2, ) + encoder_arch + (n_bottleneck_neurons,) ), 
		decoder_arch=( (n_bottleneck_neurons, ) + decoder_arch + (2,) )),
	language='python'
)

########################################
# Train the network
########################################

st.subheader('Entrenar el autoencoder')

st.markdown(text.training)

st.code(text.train_code, language='python')

from circumference_app.C_autoencoder_build_and_train import autoencoder_build_and_train
network_trained, encoder, decoder, autoencoder, loss_values = autoencoder_build_and_train(seed, data, encoder_arch, decoder_arch, n_bottleneck_neurons)

########################################
# Autoencoded data
########################################

st.subheader('Reconstruir los datos con el autoencoder')

from circumference_app.D_reconstruct_data import reconstruct_data
reconstruct_data(data, autoencoder, encoder, network_trained)
