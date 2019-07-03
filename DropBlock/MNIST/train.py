#Importing Libraries
from __future__ import print_function
import keras
import datetime
import keras.backend as K
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, BatchNormalization
from keras_drop_block import DropBlock2D
from keras import utils as np_utils
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(2019)

now = datetime.datetime.now

# Load dataset as train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Set numeric type to float32 and Normalize value to [0,1]
x_train = np.expand_dims(x_train.astype(K.floatx()) / 255, axis=-1)
x_test = np.expand_dims(x_test.astype(K.floatx()) / 255, axis=-1)

# Transform lables to one-hot encoding
y_train, y_test = np.expand_dims(y_train, axis=-1), np.expand_dims(y_test, axis=-1)

train_num = round(x_train.shape[0])
x_train, x_valid = x_train[:train_num, ...], x_train[train_num:, ...]
y_train, y_valid = y_train[:train_num, ...], y_train[train_num:, ...] 

#Design the Model
model = Sequential()
model.add(ZeroPadding2D(input_shape=(28, 28, 1), name='Input'))

model.add(Conv2D(filters=32, strides=(1, 1), kernel_size=3, activation='relu', padding='same', name='Conv-1'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, strides=(1, 1), kernel_size=3, activation='relu', padding='same', name='Conv-2'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, name='Pool-1'))
model.add(DropBlock2D(block_size=7, keep_prob=0.8, name='Dropout-1'))

model.add(Conv2D(filters=64, strides=(1, 1), kernel_size=3, activation='relu', padding='same', name='Conv-3'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, name='Pool-2'))
model.add(DropBlock2D(block_size=7, keep_prob=0.8, name='Dropout-2'))

model.add(Conv2D(filters=64, strides=(1, 1), kernel_size=3, activation='relu', padding='same', name='Conv-4'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, name='Pool-3'))
model.add(DropBlock2D(block_size=7, keep_prob=0.8, name='Dropout-3'))

model.add(Conv2D(filters=128, strides=(1, 1), kernel_size=3, activation='relu', padding='same', name='Conv-5'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, name='Pool-4'))
model.add(DropBlock2D(block_size=7, keep_prob=0.8, name='Dropout-4'))

model.add(Flatten(name='Flatten'))
model.add(Dense(units=128, activation='relu', name='Dense-1'))
model.add(BatchNormalization())

model.add(Dropout(rate=0.5, name='Dense-Dropout'))
model.add(Dense(units=10, activation='softmax', name='Softmax'))

#train the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

t = now()
history = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(x_test, y_test))

# Calculating loss and accuracy
# train
tr_loss, tr_accuracy = model.evaluate(x_train, y_train)
# tr_loss = 0.039, tr_accurary = 0.98845
# test
te_loss, te_accuracy = model.evaluate(x_test, y_test)
# te_loss = 0.042, te_accurary = 0.9861

print('Training time: %s' % (now() - t))
print('Train loss:', tr_loss)
print('Train accuracy:', tr_accuracy)
print('Test loss:', te_loss)
print('Test accuracy:', te_accuracy)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
