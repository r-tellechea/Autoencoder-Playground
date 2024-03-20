import torch
import streamlit as st

def autoencoder_configuration(seed: int, data: torch.Tensor) -> tuple[tuple[float]]:

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

	return encoder_arch, decoder_arch, n_bottleneck_neurons
