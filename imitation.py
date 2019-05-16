from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import gym
import time


model = Sequential()

model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(146, 144, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(units=3, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

X = np.load('X.npy')
Y = np.load('Y.npy')


model.fit(
    epochs=100,
    x=X,
    y=Y,
    batch_size=256
)

model.save('imitation.h5')
