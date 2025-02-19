# Red Neuronal Profunda para la clasificación de lecturas de los latidos del corazón
## Proyecto de clase: Inteligencia Artificial UNAH
Implementación de un Red Neuronal Profunda para la clasificación de lecturas de los latidos del corazón humano. Este es un problema de clasificación multiclase ya que se cuenta con 3 posibles tipos de latidos (Normal (N), Anormal (A), Inclasificable (U)).


Para realizar el entrenamiento se codifica las etiquetas utilizando la técnica conocida como One Hot Encoding de clasificación multiclase, el modelo guarda los parametros e hiperparámetros en un archivo en formato JSON nombrado params.json.
El archivo contiene un arreglo nombrado “dnn_layers” con los datos de las capas de su red neuronal (menos la capa de entrada que no tiene parámetros), “n” es el número de unidades de la capa y “activation” el nombre de la función de activación que usó en dicha capa son “relu” y "softmax", “w” es la matriz W para dicha capa representada como un arreglo de arreglos, cada arreglo dentro de dicho arreglo representa una fila de la matriz W. De manera similar a “w”, “b” contiene el vector b para cada capa.

![image](https://github.com/SevenDogsNTwoCats/Red-Neuronal-Profunda-para-la-clasificaci-n-de-lecturas-de-los-latidos-del-coraz-n/assets/78670212/996286b2-e271-432d-b4c1-5eaa204577db)
![image](https://github.com/SevenDogsNTwoCats/Red-Neuronal-Profunda-para-la-clasificaci-n-de-lecturas-de-los-latidos-del-coraz-n/assets/78670212/4d90f02c-5c60-4112-af88-9fb1c7f911a2)
![image](https://github.com/SevenDogsNTwoCats/Red-Neuronal-Profunda-para-la-clasificaci-n-de-lecturas-de-los-latidos-del-coraz-n/assets/78670212/864d84ce-af6e-4c6a-9ada-c17cee6165ce)
![image](https://github.com/SevenDogsNTwoCats/Red-Neuronal-Profunda-para-la-clasificaci-n-de-lecturas-de-los-latidos-del-coraz-n/assets/78670212/250b84a3-3819-4434-8c6f-ce021c803fe9)



Los datos para entrenar y probar la red neuronal en https://drive.google.com/drive/folders/1EmW_qaj1VhNa45A7R-leTDb6-mLl7Kk5?usp=drive_link
