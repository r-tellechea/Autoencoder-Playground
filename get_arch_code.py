#%%
from functools import reduce

str_arch = '{} = nn.Sequential(\n{}\n)\n'
str_linear_layer = '    nn.Linear({}, {}, bias=True),'
str_relu_activation = ' nn.ReLU(),\n'

def get_arch_code(arch: tuple[int], variable_name: str) -> str:
	
	list_layer_strings = [
		str_linear_layer.format(n_in, n_out)
			for n_in, n_out in zip(arch[:-1], arch[1:])
	]
	
	str_layers = reduce(
		lambda str_layer_a, str_layer_b : (
			str_layer_a + 
			str_relu_activation + 
			str_layer_b
		),
		list_layer_strings
	)

	str_arch_code = str_arch.format(
		variable_name,
		str_layers
	)

	return str_arch_code

def get_code_autoencoder(encoder_arch: tuple[int], decoder_arch: tuple[int]) -> str:
	return (
		'from torch import nn\n\n' +
		get_arch_code(encoder_arch, 'encoder') + '\n' +
		get_arch_code(decoder_arch, 'decoder') + '\n' +
		'autoencoder = nn.Sequential(encoder, decoder)'
	)
