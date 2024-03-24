Una vez tenemos la muestra de datos y hemos configurado la arquitectura de red neuronal, sólo nos queda el proceso de entrenamiento. Como función de error usaremos el [error cuadrático medio](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html), y como método de descenso del gradiente el método [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) (adaptative moment estimation with decoupled weight decay regularization). 

Los hyperparámetros en este caso son el número de etapas que recorremos optimizando el autoencoder y el _learning rate_.
