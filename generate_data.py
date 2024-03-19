import torch
import streamlit as st

from get_generator_from_seed import get_generator_from_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def generate_data(n_points: int=1000, mean: float=10., std: float=0.5, seed: int=31415) -> torch.Tensor:

	radious = torch.normal(
		mean=torch.full(
			size=(n_points,), 
			fill_value=mean, 
			device=device
		), 
		std=torch.full(
			size=(n_points,), 
			fill_value=std, 
			device=device
		),
		generator=get_generator_from_seed(seed),
	)

	angle = torch.rand_like(radious, device=device, requires_grad=False) * 2 * torch.pi

	data = torch.stack(
		[radious * torch.cos(angle), 
		 radious * torch.sin(angle)],
		dim=1
	)

	return data


