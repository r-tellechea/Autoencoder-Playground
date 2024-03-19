#%%
import torch
from torch import nn

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

#%%
gen = torch.Generator(device=device)
gen.manual_seed(1234)

#%%
n_sample = 1_000

radious = torch.normal(
	mean=torch.full(
		size=(n_sample,), 
		fill_value=10., 
		device=device
	), 
	std=torch.full(
		size=(n_sample,), 
		fill_value=0.5, 
		device=device
	),
	generator=gen,
)
angle = torch.rand_like(radious, device=device, requires_grad=False) * 2 * torch.pi

#%%
circumference = torch.stack(
	[radious * torch.cos(angle), radious * torch.sin(angle)],
	dim=1
)
outliers = torch.tensor([[0,0],[2,2],[13, 13]], device=device)
data = torch.concat([circumference, outliers])

#%%
data_fig = px.scatter(
	data_frame=pd.DataFrame(
		data=data.cpu().numpy(),
		columns=['x', 'y']
	),
	x='x',
	y='y',
)

#%%
encoder = nn.Sequential(
	nn.Linear(2, 8, device=device),
	nn.ReLU(),
	nn.Linear(8, 8, device=device),
	nn.ReLU(),
	nn.Linear(8, 8, device=device),
	nn.ReLU(),
	nn.Linear(8, 1, device=device),
)
decoder = nn.Sequential(
	nn.Linear(1, 8, device=device),
	nn.ReLU(),
	nn.Linear(8, 8, device=device),
	nn.ReLU(),
	nn.Linear(8, 8, device=device),
	nn.ReLU(),
	nn.Linear(8, 2, device=device),
)
autoencoder = nn.Sequential(encoder, decoder)

#%%
from torchinfo import summary
summary(autoencoder, input_size=data.shape)

#%%
loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.AdamW(
	params=autoencoder.parameters(),
	lr=0.01
)

#%%
epochs = 1_000
loss_values = []
for epoch in range(epochs):
	autoencoder.train()
	data_pred = autoencoder(data)
	loss = loss_fn(data, data_pred)
	optimizer.zero_grad()
	loss.backward()
	loss_values.append(loss.item())
	optimizer.step()
	if epoch % 100 == 0 or epoch == epochs-1:
		print(f'epoch: {epoch}, loss: {loss_values[-1]}')

#%%
px.line(
	data_frame=(
		pd.DataFrame(
			data=loss_values, 
			columns=['loss']
		)
		.assign(epoch=range(epochs))
	),
	x='epoch',
	y='loss',
	title='Loss value over epochs',
)

#%%
prediction_fig = px.scatter(
	data_frame=pd.DataFrame(
		data=autoencoder(data).detach().cpu().numpy(),
		columns=['x', 'y']
	),
	x='x',
	y='y',
	color_discrete_sequence=['salmon']
)

#%%
fig = go.Figure(data=data_fig.data + prediction_fig.data)
fig
