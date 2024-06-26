import torch
import streamlit as st
from streamlit_pills import pills 

from plotly_figures import get_fig_nn

def autoencoder_configuration(seed: int, data: torch.Tensor) -> tuple[tuple[int] | int]:

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
		bottleneck_neurons = pills(
			label='Bottleneck neurons',
			key='bottleneck_neurons',
			options=['1 neuron', '2 neurons'],
			icons=['💧', '🌊']
		)
		n_bottleneck_neurons = 1 if bottleneck_neurons == '1 neuron' else 2

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

	autoencoder_arch = (2,) + encoder_arch + (n_bottleneck_neurons,) + decoder_arch + (2,)

	st.plotly_chart(get_fig_nn(encoder_arch, decoder_arch, n_bottleneck_neurons))

	return encoder_arch, decoder_arch, n_bottleneck_neurons
