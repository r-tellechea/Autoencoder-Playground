import torch
from torch import nn

encoder = nn.Sequential(
	nn.Linear(2, 8), nn.ReLU(),
	nn.Linear(8, 8), nn.ReLU(),
	nn.Linear(8, 8), nn.ReLU(),
	nn.Linear(8, 1),
)
decoder = nn.Sequential(
	nn.Linear(1, 8), nn.ReLU(),
	nn.Linear(8, 8), nn.ReLU(),
	nn.Linear(8, 8), nn.ReLU(),
	nn.Linear(8, 2),
)
autoencoder = nn.Sequential(encoder, decoder)
