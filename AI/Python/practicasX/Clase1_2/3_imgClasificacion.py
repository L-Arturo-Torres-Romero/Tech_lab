"""
Descripción general del archivo
-------------------------------

Este script implementa un ejemplo básico de clasificación de imágenes usando
Deep Learning con TensorFlow/Keras.

El objetivo es entrenar una red neuronal para reconocer prendas de vestir
usando el dataset Fashion-MNIST. Este dataset contiene imágenes en escala
de grises de tamaño 28x28 píxeles que representan diferentes categorías
de ropa (camisetas, zapatos, pantalones, etc.).

El flujo típico de un proyecto de deep learning que se muestra aquí es:

1. Cargar el dataset
2. Visualizar ejemplos
3. Preprocesar los datos (normalización)
4. Definir la arquitectura de la red neuronal
5. Compilar el modelo (definir función de pérdida, optimizador, métricas)
6. Entrenar el modelo
7. Evaluar el modelo con datos no vistos
8. Realizar predicciones

Este ejemplo usa una red neuronal densa (fully connected neural network),
una de las arquitecturas más simples en deep learning.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Cargar datos

print("cargando datos")

# Fashion-MNIST es un dataset estándar de visión por computadora usado para
# aprender clasificación de imágenes con deep learning.
# Contiene 70,000 imágenes (60k para entrenamiento y 10k para prueba).
fashion_mnist = tf.keras.datasets.fashion_mnist

# load_data() descarga el dataset y lo divide automáticamente en:
# train_images / train_labels -> datos usados para entrenar el modelo
# test_images / test_labels   -> datos usados para evaluar el modelo
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#visualizar datos
print("visualizando datos")

# Visualizamos una imagen del dataset para entender el tipo de datos
# que recibirá la red neuronal. Cada imagen es una matriz de 28x28 píxeles.
plt.imshow(train_images[20],cmap='gray')

# plt.show(block=False)
plt.show()

# La etiqueta (label) indica la clase real de la imagen.
# En Fashion-MNIST las clases son números del 0 al 9.
print(f"Label: {train_labels[20]}")

print("normalizando train_images, test_images")

# Normalización de datos
# ----------------------
# Los valores de los píxeles están originalmente entre 0 y 255.
# Dividir entre 255 los convierte a un rango [0,1].
#
# Esto es una práctica estándar en deep learning porque:
# - mejora la estabilidad numérica
# - acelera la convergencia del entrenamiento
train_images=train_images/255.0
test_images=test_images/255.0

print("creando el  model using keras.sequential")

# Definición del modelo
# ---------------------
# keras.Sequential crea una red neuronal donde las capas están
# conectadas secuencialmente (una después de otra).
model= keras.Sequential([

    # Flatten transforma la imagen 28x28 en un vector de 784 elementos.
    # Esto es necesario porque las capas Dense esperan vectores 1D.
    keras.layers.Flatten(input_shape=(28,28)),

    # Capa densa (fully connected layer)
    # Cada neurona está conectada con todas las entradas.
    # Tiene 15 neuronas y usa la función de activación ReLU.
    #
    # ReLU introduce no linealidad, lo cual permite que la red
    # aprenda representaciones más complejas.
    keras.layers.Dense(15, activation=tf.nn.relu),

    # Capa de salida con 10 neuronas (una por cada clase).
    #
    # Softmax convierte los valores en probabilidades que
    # suman 1. Esto permite interpretar la salida como la
    # probabilidad de pertenecer a cada clase.
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

print("compilando el modelo")

# Compilación del modelo
# ----------------------
# Aquí se definen tres componentes fundamentales del entrenamiento:

model.compile(

    # Adam es un optimizador muy usado en deep learning.
    # Ajusta los pesos de la red usando gradiente descendente
    # con adaptación automática del learning rate.
    optimizer = tf.optimizers.Adam(),

    # Función de pérdida (loss function)
    # sparse_categorical_crossentropy se usa cuando:
    # - hay múltiples clases
    # - las etiquetas están representadas como enteros
    loss = "sparse_categorical_crossentropy",

    # Métrica para monitorear el desempeño durante entrenamiento
    metrics=['accuracy']
)

print("entrenando el modelo con model.fit")

# Entrenamiento del modelo
# ------------------------
# model.fit realiza el entrenamiento usando backpropagation.
#
# Durante cada epoch:
# 1. Se pasan las imágenes por la red (forward pass)
# 2. Se calcula el error usando la función de pérdida
# 3. Se calculan gradientes (backpropagation)
# 4. El optimizador ajusta los pesos
model.fit(train_images, train_labels, epochs=5)

print("evaluando el modelo con model.evaluate")

# Evaluación del modelo
# ---------------------
# Se prueba el modelo con datos que NO se usaron para entrenar.
# Esto permite medir la capacidad de generalización.
model.evaluate(test_images, test_labels)

print("model.predict")

# Predicción
# ----------
# model.predict devuelve las probabilidades predichas para cada clase.
classifications = model.predict(test_images)

print("imprimir clasificaciones")

# Este vector contiene 10 valores (uno por cada clase).
# Cada valor representa la probabilidad estimada por la red.
print(classifications[20])

print("imprimir test_labels")

# Aquí mostramos la clase real para comparar con la predicción.
print(test_labels[20])
