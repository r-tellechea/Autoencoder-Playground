Como decíamos en la introducción, `autoencoder` va a ser una función que a su vez es composición de otras dos funciones:
$$
\begin{cases}
\textnormal{encoder} : \mathbb{R}^2 \rightarrow \mathbb{R}^1 : x, y \mapsto t \\
\textnormal{decoder} : \mathbb{R}^1 \rightarrow \mathbb{R}^2 : t \mapsto \hat{x}, \hat{y}
\end{cases}
$$
Nuestra intención es que $\textnormal{autoencoder} = \textnormal{decoder} \circ \textnormal{encoder} \approx \textnormal{Id}$, es decir, que $(x, y) \approx (\hat{x}, \hat{y})$. Sabemos que, en principio, esta reducción de dimensión y su reconstrucción debería ser posible porque, quitando la aleatoriedad de $r$, cada punto en $\mathcal{D}$ se corresponde únivocamente con un valor $\theta \in [0, 2\pi)$. Es decir, que es posible reconstruir los puntos en $\mathcal{D}$ sin más error que el que esté introducciendo $r$.

Vamos a construir estas funciones como redes neuronales cuyas características van a ser las siguientes:
- Como partimos de datos en dos dimensiones, y queremos obtener como resultado esos mismos datos, tanto la entrada como la salida de `autoencoder` serán capas con dos neuronas.
- Entre medias buscamos forzar una reducción de la dimensión obligando a todo el flujo de la red a pasar por una capa con una sola neurona, es decir, a pasar por una dimensión. A esta capa la llamaremos el _cuello de botella_ de la red neuronal.
- Las capas que van del _input_ al cuello de botella formarán el `encoder`, y las que van del cuello de botella al _output_ formarán el `decoder`.

Tanto el `encoder` como el `decoder` tendrán de cero a cuatro capas internas, cada una de ellas con entre una y ocho neuronas. También damos la posibilidad de probar qué pasa en caso de que introduzcamos no una sino dos neuronas en la capa del cuello de botella, en cuyo caso la red es capaz de reproducir idénticamente los puntos de partida.

Todo eso se puede configurar con los siguientes controles: