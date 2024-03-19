import torch
from torch import nn
from functools import reduce

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def autoencoder_train(autoencoder: nn.Module, data: torch.Tensor, epochs: int=500, lr: float=0.01):

	loss_fn = nn.MSELoss(reduction='mean')
	optimizer = torch.optim.AdamW(
		params=autoencoder.parameters(),
		lr=lr,
	)

	loss_values = []
	
	for epoch in range(epochs):
		autoencoder.train()
		data_pred = autoencoder(data)
		loss = loss_fn(data, data_pred)
		optimizer.zero_grad()
		loss.backward()
		loss_values.append(loss.item())
		optimizer.step()

	return loss_values

def get_autoencoder(
	encoder_arch: tuple[int], 
	decoder_arch: tuple[int], 
	n_bottleneck_neurons: int, 
	data: torch.Tensor, 
	epochs: int=500, 
	lr: float=0.01 ):

	encoder_in_out_pairs = zip((2,) + encoder_arch, encoder_arch + (n_bottleneck_neurons,))
	decoder_in_out_pairs = zip((n_bottleneck_neurons,) + decoder_arch, decoder_arch + (2,))

	encoder_args = reduce(
		lambda tuple_x, tuple_y : tuple_x + (nn.ReLU(),) + tuple_y,
		[(nn.Linear(input_features, output_features, device=device),)
			for input_features, output_features in encoder_in_out_pairs]
	)
	decoder_args = reduce(
		lambda tuple_x, tuple_y : tuple_x + (nn.ReLU(),) + tuple_y,
		[(nn.Linear(input_features, output_features, device=device),)
			for input_features, output_features in decoder_in_out_pairs]
	)
	
	encoder = nn.Sequential(*encoder_args)
	decoder = nn.Sequential(*decoder_args)
	autoencoder = nn.Sequential(encoder, decoder)

	loss_values = autoencoder_train(autoencoder, data, epochs, lr)

	return encoder, decoder, autoencoder, loss_values
