loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.AdamW(
    params=autoencoder.parameters(),
    lr=lr,
)	

for epoch in range(epochs):
    autoencoder.train()
    data_pred = autoencoder(data)
    loss = loss_fn(data, data_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()