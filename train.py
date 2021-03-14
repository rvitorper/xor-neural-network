import keras
import pandas as pd
from keras.layers.core import Dense
from keras.optimizers import SGD

data = pd.read_csv('data.csv')

model = keras.Sequential()

n = 2
m = 1
num_hidden_neurons = n + m + 2

num_output_neurons = m
model.add(Dense(num_hidden_neurons, input_dim=n, activation='tanh'))
model.add(Dense(num_hidden_neurons, activation='tanh'))
model.add(Dense(num_output_neurons, activation='tanh'))

opt = SGD(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

accuracy = 0

while accuracy < 0.95:
    history = model.fit(data[['a', 'b']], data['r'], epochs=1500)
    accuracy = history.history['binary_accuracy'][-1]

model.save('model.keras')
