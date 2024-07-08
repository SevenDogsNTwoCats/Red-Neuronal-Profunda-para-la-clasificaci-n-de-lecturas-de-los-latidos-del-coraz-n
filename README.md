# Red Neuronal Profunda para la clasificación de lecturas de los latidos del corazón
## Proyecto de clase: Inteligencia Artificial UNAH
Implementación de un Red Neuronal Profunda para la clasificación de lecturas de los latidos del corazón humano. Este es un problema de clasificación multiclase ya que se cuenta con 3 posibles tipos de latidos (Normal (N), Anormal (A), Inclasificable (U)).


Para realizar el entrenamiento se codifica las etiquetas utilizando la técnica conocida como One Hot Encoding de clasificación multiclase, el modelo guarda los parametros e hiperparámetros en un archivo en formato JSON nombrado params.json.
El archivo contiene un arreglo nombrado “dnn_layers” con los datos de las capas de su red neuronal (menos la capa de entrada que no tiene parámetros), “n” es el número de unidades de la capa y “activation” el nombre de la función de activación que usó en dicha capa son “relu” y "softmax", “w” es la matriz W para dicha capa representada como un arreglo de arreglos, cada arreglo dentro de dicho arreglo representa una fila de la matriz W. De manera similar a “w”, “b” contiene el vector b para cada capa.

Los datos para entrenar la red neuronal son x_train (atributos) y y_train (etiquetas).
