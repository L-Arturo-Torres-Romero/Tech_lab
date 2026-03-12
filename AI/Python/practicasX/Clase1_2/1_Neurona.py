# ---------------------------------------------------------------------------
# GENERAL DESCRIPTION OF THIS FILE
# ---------------------------------------------------------------------------
#
# This script demonstrates the most basic workflow of training a neural
# network using a supervised learning approach. The goal is to show how
# a neural network can learn the relationship between inputs and outputs
# from example data.




# Se importa la capa Dense. En deep learning, una capa "Dense"
# representa una capa completamente conectada de neuronas artificiales.
# Cada neurona recibe todas las entradas de la capa anterior
# y aprende pesos para combinar esas entradas.
from tensorflow.keras.layers import Dense

# Input permite definir explícitamente la forma de los datos de entrada
# que recibirá la red neuronal.
from tensorflow.keras import Input

# Numpy se usa aquí simplemente para definir los datos de entrenamiento.
import numpy as np

# Herramienta para visualizar la arquitectura de la red neuronal.
from tensorflow.keras.utils import plot_model

from tensorflow import keras


# ------------------------------------------------------------------
# DATASET DE ENTRENAMIENTO
# ------------------------------------------------------------------
# Se define un pequeño conjunto de datos de entrada (x).
# En deep learning, estos son los "features" o variables de entrada.
x = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype=float)

# Se define la salida esperada (y) para cada entrada.
# Estas son las "labels" o valores objetivo que el modelo debe aprender.
y = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0], dtype=float)

# Observando los datos se puede notar que existe una relación lineal:
# y = 2x - 1
# El objetivo del modelo de deep learning será aprender automáticamente
# esta relación a partir de los datos.


# ------------------------------------------------------------------
# DEFINICIÓN DEL MODELO DE RED NEURONAL
# ------------------------------------------------------------------
# Se crea un modelo secuencial.
# Esto significa que las capas se organizan una después de otra
# formando un flujo de datos simple desde la entrada hasta la salida.
model2 = keras.Sequential()

# Se define la capa de entrada.
# shape=(1,) significa que cada muestra de datos tiene una sola variable.
model2.add(Input(shape=(1,)))

# Se añade una capa densa con una sola neurona.
# Matemáticamente esta neurona aprende una función:
#
# y_hat = w*x + b
#
# donde:
# w = peso aprendido
# b = bias aprendido
#
# Es decir, esta red neuronal es equivalente a una regresión lineal.
model2.add(Dense(1))


# ------------------------------------------------------------------
# CONFIGURACIÓN DEL PROCESO DE ENTRENAMIENTO
# ------------------------------------------------------------------
# Se configura cómo se entrenará la red.
model2.compile(
    optimizer="sgd",               # Algoritmo de optimización (Stochastic Gradient Descent)
    loss="mean_squared_error"      # Función de costo usada para medir el error
)

# El optimizador SGD ajustará los pesos w y b
# utilizando el gradiente del error para minimizar la función de pérdida.


# Muestra un resumen de la arquitectura del modelo:
# número de capas, parámetros entrenables, etc.
model2.summary()


# ------------------------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# ------------------------------------------------------------------
# Se entrena la red neuronal usando los pares (x,y).
#
# Durante el entrenamiento:
# 1. El modelo hace una predicción inicial.
# 2. Se calcula el error con respecto al valor real.
# 3. Se calcula el gradiente del error.
# 4. Se ajustan los pesos.
#
# Este proceso se repite múltiples veces sobre el dataset.
model2.fit(x,y, epochs=10)

# "epochs=10" significa que el dataset completo se procesa
# 10 veces para refinar los pesos del modelo.


# ------------------------------------------------------------------
# PREDICCIÓN
# ------------------------------------------------------------------
# Una vez entrenado, el modelo puede usarse para inferencia.
# Aquí se solicita una predicción para un valor nuevo de x.
yp = model2.predict([3.8])

# Si el modelo aprendió correctamente la relación:
# y = 2x - 1
# entonces el resultado debería aproximarse a:
#
# y ≈ 2(3.8) - 1 = 6.6
print (yp)


# ------------------------------------------------------------------
# VISUALIZACIÓN DE LA RED
# ------------------------------------------------------------------
# Genera un diagrama de la arquitectura de la red neuronal,
# mostrando las capas y las dimensiones de los tensores.
plot_model(model2, show_shapes=True)

