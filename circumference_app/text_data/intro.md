En esta aplicación queremos explicar de un modo interactivo la arquitectura del autoencoder y su uso: la reducción de dimensión.

Tomaremos como punto de partida una nube de puntos que se adecúan más o menos a una circunferencia en un espacio de dos dimensiones. Trataremos de proyectar estos datos sobre otro espacio de una dimensión de manera que podamos reconstruir la muestra original desde su proyección en este espacio reducido. La operación de reducción la llamaremos `encoder`, a la de reconstrucción `decoder`, y a la composición de ambas `autoencoder`.

$$
\mathbb{R}^2 
\overset{\textnormal{encoder}}{\longrightarrow} 
\mathbb{R}^1 
\overset{\textnormal{decoder}}{\longrightarrow} 
\mathbb{R}^2
$$

Las opciones seleccionadas por defecto son suficientes para entender el autoencoder, pero aconsejamos jugar con los parámetros para ver qué efecto tienen. Hemos dejado como modificables aquellas configuraciones que creemos que muestran algún aspecto del proceso.

Para que los resultados se puedan reproducir a partir de una configuración, todos los valores aleatorios dependen de una semilla inicial que se puede modificar aquí: