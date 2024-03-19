import torch
import streamlit as st

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def get_generator_from_seed(seed: int) -> torch.Generator:
	gen = torch.Generator(device=device)
	gen.manual_seed(seed)
	return gen
