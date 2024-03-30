Los puntos que vamos a tomar en $\mathbb{R}^2$ tienen como referencia una circunferencia de centro $(0,0)$ y radio $r$. Cada punto de la circunferencia está determinado por un ángulo $\theta \in [0, 2\pi)$ siguiendo las fórmulas:

$$
\begin{cases}
x = r \cdot \textnormal{cos}(\theta) \\
y = r \cdot \textnormal{sin}(\theta) \\
\end{cases}
$$

Queremos elaborar una muestra de $N$ puntos. Siguiendo estas ecuaciones, bastaría con generar aleatoriamente $N$ valores de $\theta$ de manera uniforme para tener $N$ puntos de esta circunferencia. Sin embargo, para añadir un poco de incertidumbre al problema vamos a permitir que el radio $r$ de cada punto en la circunferencia sea otra variable aleatoria con un pequeño margen de desviación.

Así pues, nuestras variables aleatorias de partida siguen las distribuciones

$$
\begin{cases}
r \sim \mathcal{N}(\mu, \sigma) \\
\theta \sim \mathcal{U}(0, 2\pi)\\
\end{cases}
$$

y a partir de éstas obtenemos los puntos $\mathcal{D} = \{(x_i, y_i) : i \in \{1 , ... , N \}\}$. Si queremos que todos los puntos estén estrictamente en la circunferencia (que cumplan $x_i^2 + y_i^2 = r^2$), nos basta con establecer $\sigma = 0$.

Nuestro conjunto de datos $\mathcal{D}$ va a estar determinado, entonces, por el número de puntos $N$ y los parámetros de la distribución del radio, $(\mu, \sigma)$. Aquí podemos configurar estos valores:



