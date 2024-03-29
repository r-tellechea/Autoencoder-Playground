import torch
import streamlit as st

from autoencoder import get_autoencoder
from plotly_figures import get_fig_loss

def autoencoder_build_and_train(seed: int, data: torch.Tensor, encoder_arch: tuple[float], decoder_arch: tuple[float], n_bottleneck_neurons: int):

	# Training configuration.
	col_epochs, col_lr = st.columns((4, 1))

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


	col_status_info, col_train_button = st.columns((4, 1))

	with col_status_info:
		status_info_placeholder = st.empty()
		status_info_placeholder.info('The autoencoder is ready to train.', icon='🚀')

	with col_train_button:
		train_button = st.button(label='TRAIN!', key='train_button')

	network_trained = False

	# Create and train the neural networks.
	loss_plot_placeholder = st.empty()

	if train_button:
		with status_info_placeholder:
			progress_bar = st.progress(value=0., text='Training autoencoder...')
			with st.spinner('Training autoencoder...'):
				encoder, decoder, autoencoder, loss_values = get_autoencoder(
					encoder_arch, 
					decoder_arch, 
					n_bottleneck_neurons,
					data,
					epochs,
					lr,
					progress_bar,
					seed
				)
		st.toast('Autoencoder trained!')
		status_info_placeholder.success('Autoencoder trained!', icon='✅')
		fig_loss = get_fig_loss(loss_values, epochs)
		loss_plot_placeholder.plotly_chart(fig_loss, use_container_width=True)
		network_trained = True

	if network_trained:
		return network_trained, encoder, decoder, autoencoder, loss_values
	else:
		return network_trained, None, None, None, None
