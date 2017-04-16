'''generic_application'''

# pylint: disable=E0401
# pylint: disable=C0103

# Imports.
import numpy as np
import tensorflow.contrib.keras as keras

# the four different states of the XOR gate
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

# the four expected results in the same order
target_data = np.array([[0], [1], [1], [0]], "float32")

model = keras.models.Sequential()
model.add(keras.layers.Dense(16, input_dim=2, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=500, verbose=2)

print(model.predict(training_data).round())

