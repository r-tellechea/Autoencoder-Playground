Como decíamos en la introducción, `autoencoder` va a ser una función que a su vez es composición de otras dos funciones:
$$
\begin{cases}
\textnormal{encoder} : \mathbb{R}^2 \rightarrow \mathbb{R}^1 : x, y \mapsto t \\
\textnormal{decoder} : \mathbb{R}^1 \rightarrow \mathbb{R}^2 : t \mapsto \hat{x}, \hat{y}
\end{cases}
$$
Nuestra intención es que $\textnormal{decoder} \circ \textnormal{encoder} \approx \textnormal{Id}$, es decir, que $(x, y) \approx (\hat{x}, \hat{y})$.

Vamos a construir estas funciones como redes neuronales en las que las capas de entrada y salida van a depender de las dimensiones de partida y de llegada.

