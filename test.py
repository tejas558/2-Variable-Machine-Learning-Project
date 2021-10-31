#inporting tensor flow and relevantt libraries into python
import tensorflow as tf
import numpy as np
from tensorflow import keras

#creating the neural network model in tensor flow
#nodes = 1
#model is the sgd model and the loss is measured by mean squared error
model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

#array of x and y values that we want to perform regression on
#data is in the form y = 3x+4
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([1.0, 4.0, 7.0, 10.0, 13.0, 16.0])

#calculating the model
model.fit(x, y, epochs=1000)

#guessing for value x=10 and predicting a relevant y value
print(model.predict([10.0]))