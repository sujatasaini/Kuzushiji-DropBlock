# kNN with neighbors=4 benchmark for MNIST

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import keras
from keras.datasets import mnist

# Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Intialize the Model
clf = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1)
print('Fitting', clf)
clf.fit(x_train, y_train)
print('Evaluating', clf)

# Train the model
test_score = clf.score(x_test, y_test)
print('Test accuracy:', test_score)
