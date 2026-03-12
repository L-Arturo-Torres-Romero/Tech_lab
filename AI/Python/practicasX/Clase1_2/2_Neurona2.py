from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
import numpy as np
import keras
import pydot
import graphviz
from tensorflow.keras.utils import plot_model
#from keras.utils.vis_utils import plot_model
#import tensorflow.keras.utils as util
#from keras.utils.vis_utils import plot_model


x = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype=float)
y = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0], dtype=float)


model2 = keras.Sequential()

model2.add(Input(shape=(1,)))
model2.add(Dense(1))

model2.compile(optimizer="sgd", loss="mean_squared_error")
model2.summary()
model2.fit(x,y, epochs=10)


yp = model2.predict([3.8])
print (yp)

img='model.png'
plot_model(model2, show_shapes=False, to_file=img )
