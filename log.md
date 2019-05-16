Cosas que intentamos:

* Policy Gradient a la Kaparthy directo
    * Archivo de Kaparthy cambiando Pong por Skiing.
    * Todo el juego es un episodio.
    * Red neuronal densa.
    * Se preprocesa la imagen dejando en blanco (1) el jugador y todos los elementos con los que interactúa, y en negro todo lo demás.
    * La exploración se implementa en un proceso similar a Kaparthy haciendo una ruleta de con las probabilidades calculadas por la red para elegir la acción.
    * Aprende a atorarse. Sospechamos que tiene que ver que solo estamos dando rewards negativas y lo que se da en cada episodio no es muy significativo, el reward de las banderas se da al final.
* Modificaciones a la función discounted_rewards
    * A lo largo de los experimentos hicimos varias modificaciones a la función original de Kaparthy, principalmente activar/desactivar la normalización. La hipótesis es que la normalización puede cambiar el signo de valores y meter ruido. En general, tuvimos mejores resultados dejando la función en su estado original con normalización.
* Feature engineering
    * Todo el juego es un episodio.
    * Tomando en cuenta las observaciones anteriores, decidimos abstraer la recompensa del ambiente del que usamos para entrenar.
    * Se extrae la cantidad de banderas por las que pasó el jugador, dando recompensa positiva por cada una al final del juego.
    * Para considerar el tiempo establecimos como métrica el número de cuadros. Observamos que el peor juego siempre tiene 4507 cuadros y creamos una función (reward_tiempo(cuadros)=4507/cuadros-1) que da valores 0<=x<10 de acuerdo al tiempo que haya tardado el jugador en terminar el episodio. Entre menor el tiempo mayor la recompensa.
    * Se elimina la recompensa continua de los cuadros intermedios, se reemplazan con ceros. La idea es que la recompensa final de los valores necesarios a todos los pasos con la función discouted_rewards.
    * No hubo ningún resultado apantallante. Después nos dimos cuenta que la normalización podría interferir con este plan al hacer que las acciones al principio del episodio siempre tengan un peso negativo. Probamos desactivar la normalización.
    * En varios intentos llegamos a una situación donde el agente alcanza un overfitting y sigue una ruta muy marcada, aunque esta no sea la”mejor”.
* Simulated Annealing
    * En vista de la falta de exploración, decidimos implementar simulated annealing con un epsilon similar a como vimos en tareas/ejercicios anteriores. Esto ayudó a diversificar los ejemplos de entrenamiento, pero no necesariamente le hacía aprender lo “bueno” o “ideal”.
    * Pese a haberlo implementado muy al inicio, la mayoría de las iteraciones posteriores las intentamos con y sin este tipo de exploración.
* Redes convolucionales
    * Consideramos que la complejidad visual del juego contra Pong ameritaba una CNN. Se cambió el procesamiento de la imagen a dejarla toda y a color, recortando solo los márgenes.
    * Estructura de VGG, dos capas.
    * Notamos una sutil mejora en general del juego para no atorarse y responder mejor a ciertas situaciones, pero no lo suficiente para que se vea como un agente inteligente.
* Feature engineering 1.5:
    * En vista de que nada era suficiente cambiamos todo el sistema de recompensas. Con la idea de volver el juego más de “jugadas” como Pong, y acortar el tamaño del set que se entrena con una recompensa decidimos partir el juego en sub-episodios que consisten del lapso vertical entre cada par de banderas.
    * Para esto solo vemos el marcador del juego e insertamos la recompensa en el momento que cambia. Sin embargo, solo cambia cuando el jugador pasa entre las banderas, por lo que no estábamos detectando cuando NO pasa para dar una recompensa negativa.
    * Lo único positivo es que siempre aprendió que bajar es bueno.
* Feature engineering 2: Electric Bugaloo
    * Dadas las deficiencias del intento anterior, decidimos hacer la división completa y total.
    * Esto implicó hacer feature extraction directamente del ambiente con la imágen para detectar el momento exacto en el que el juego determina si el jugador pasó o no entre las banderas, solo que ahora, en lugar de hacerlo con el marcador, lo hicimos extrayendo la información de la pantalla y calibrando para dar la recompensa en exactamente el mismo cuadro que el ambiente la daría.
    * Medimos la cantidad de cuadros promedio que toma llegar de unas banderas a las próximas para hacer un índice de velocidad para dar recompensa por el tiempo.
    * Al final, damos una recompensa en el momento que el jugador pasa por las banderas, positiva si entre, negativa si afuera y se le suma una recompensa (balanceada para ser menor) basada en el tiempo.
    * El jugador aprendió rápidamente que si no se mueve no hay riesgo de recompensas negativas, por ende aprendió a girar para detenerse arriba.
* Feature engineering 2.1
    * Para solucionar el problema anterior decidimos sumale un “bias” a las recompensas y volverlas todas positivas. De esta manera ya no se atora el jugador, pero tiene problemas para aprender lo bueno que es pasar por las banderas.
    * Con este método llegamos a tener juegos muy buenos, algunos llegando hasta 16/20 banderas cruzadas, sin embargo no lo hace de manera consistente, por lo que no nos dejó satisfechos. 
    * Por lo menos su recompensa total promedio es ~25% más alta que la de un agente aleatorio. 
        * Agente aleatorio: ~-17k
        * Promedio: ~-13k
        * Mejor: -6533
        * Un buen juego: ~-4000
* Contexto historico
    * Inspirados en DQN intentamos alimentar la red neuronal con los últimos dos cuadros con la esperanza de que aprendiera más de velocidades y trayectorias.
    * Creemos haber notado suficiente mejora como para justificar dejarlo así, pero podría ser solo una impresión empírica. 
    * Épocas a base de éxito
    * Se produce un índice de qué tan bueno fue un episodio con base en el puntaje. Se utiliza este índice para darle más épocas de entrenamiento a buenos episodios.
    * No parece haber hecho diferencia.
* DQN
    * Buscando resultados en internet vimos algo de éxito anterior utilizando DQN para Skiing. Nuestra primera aproximación fue utilizar el algoritmo como se plantea en la tarea de Pong.
    * Intentamos con las recompensas originales y las propias.
    * Aprendizaje muy muy lento, nos rendimos después de ver la lentitud y el poco progreso.
* Otra estructura convolucional
    * Estructura ‘clásica’ convolucional: tres capas seguidas.
    * Menor peso.
    * Resultados promedio muy similares, pero ligeramente peores.
* Exploración por imitación 1
    * Con la idea de darle al agente una red pre-entrenada con algo “bien”, intentamos nosotros hacer varios juegos e intentar que la red aprendiera de estos.
    * Notamos que la cantidad de juegos que tendríamos que jugar nosotros sería demasiado grande para que aprenda a extrapolar y reaccionar a diferentes situaciones.
    * El agente tuvo muchos problemas para salir de problemas para los que no había entrenamiento previo, por ejemplo atorarse con las orillas, con árboles o solito.
* Exploración por imitación 2
    * Encontramos un ejemplo que hace uso de puro feature extraction para hacer un “maestro” para el agente que le dice qué hacer en todo momento. El agente aprende excelente, pero los movimientos son muy mecánicos y parecen el producto de una lógica sencilla basada en ifs. Solo aprende a qué hacer en cada cuadro y es puro behaviour cloning. Pese a que se basa en un paper interesante al respecto, hacer puro behaviour cloning no es lo ideal.
    * Tomando esta aproximación como base, decidimos implementar al “maestro” combinado con PG. En este caso, el simulated annealing, en lugar de proponer una acción aleatoria, propone la que el maestro indica, así el agente tiene la oportunidad de aprender qué haría un agente casi ideal en sus zapatos.
    * Los resultados se ven favorables y parecen ser lo mejor hasta ahora.




