# Import the libraries
from keras import backend as K
from keras.datasets import mnist
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant
from keras.optimizers import Adam
import matplotlib.pyplot as plt
 
# Load the dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
 
y_test = np_utils.to_categorical(y_test, 10)
y_train = np_utils.to_categorical(y_train, 10)

# Build the model
K.set_image_dim_ordering("th")
#LeNet
model = Sequential()
model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=(1,28,28)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
 
model.add(Conv2D(50, kernel_size=5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D())
 
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
 
model.add(Dense(10))
model.add(Activation("softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)
print("Test score:", score[0])
print("Test accuracy:", score[1])
print(history.history.keys())

# Plot the graph of the model
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
