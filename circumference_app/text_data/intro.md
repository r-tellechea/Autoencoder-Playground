En esta aplicación queremos explicar de un modo interactivo la arquitectura del autoencoder y su uso: la reducción de dimensión.

Tomaremos como punto de partida una nube de puntos que se adecúan más o menos a una circunferencia en un espacio de dos dimensiones. Trataremos de proyectar estos datos sobre otro espacio de una dimensión de manera que podamos reconstruir aproximadamente la muestra original desde su proyección. La operación de reducción la llamaremos `encoder`, a la de reconstrucción `decoder`, y a la composición de ambas `autoencoder`.

$$
\mathbb{R}^2 
\overset{\textnormal{encoder}}{\longrightarrow} 
\mathbb{R}^1 
\overset{\textnormal{decoder}}{\longrightarrow} 
\mathbb{R}^2
$$

Las opciones seleccionadas por defecto son suficientes para entender el autoencoder, pero aconsejamos jugar con los parámetros para ver qué efecto tienen. Hemos dejado como modificables aquellas configuraciones que creemos que muestran algún aspecto del proceso. El código completo puede verse en el [repositorio](https://github.com/r-tellechea/Autoencoder-Playground) de esta aplicación. Limitamos algunos parámetros (el número de capas de la red neuronal, de neuronas por capa, de etapas de entrenamiento...) para limitar la cantidad de computación, pero a quien quiera probar más allá de los límites le invitamos a descargar el código, modificarlo libremente y lanzar la aplicación en local. En el repositorio se pueden encontrar también las instrucciones.

Para que los resultados se puedan reproducir a partir de una configuración, todos los valores aleatorios en esta aplicación dependen de una semilla inicial que se puede modificar aquí: