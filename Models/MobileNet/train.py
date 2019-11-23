# Import the libraries
import numpy as np
import keras
import random
import psutil
from keras.datasets import mnist

from keras import backend as K
from tqdm import tqdm
from scipy import misc
import tensorflow as tf

np.random.seed(2017)
tf.set_random_seed(2017)

# Load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
height,width = 56,56

from keras.applications.mobilenet import MobileNet
from keras.layers import Input,Dense,Dropout,Lambda
from keras.models import Model
from keras import backend as K

# Build the model
input_image = Input(shape=(height,width))
input_image_ = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,3),3,3))(input_image)
base_model = MobileNet(input_tensor=input_image_, include_top=False, pooling='avg')
output = Dropout(0.5)(base_model.output)
predict = Dense(10, activation='softmax')(output)

# Compile the model
model = Model(inputs=input_image, outputs=predict)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

X_train = X_train.reshape((-1,28,28))
X_train = np.array([misc.imresize(x, (height,width)).astype(float) for x in tqdm(iter(X_train))])/255.

X_test = X_test.reshape((-1,28,28))
X_test = np.array([misc.imresize(x, (height,width)).astype(float) for x in tqdm(iter(X_test))])/255.

def random_reverse(x):
	if np.random.random() > 0.5:
		return x[:,::-1]
	else:
		return x
# Data-augmentation
def data_generator(X,Y,batch_size=100):
	while True:
		idxs = np.random.permutation(len(X))
		X = X[idxs]
		Y = Y[idxs]
		p,q = [],[]
		for i in range(len(X)):
			p.append(random_reverse(X[i]))
			q.append(Y[i])
			if len(p) == batch_size:
				yield np.array(p),np.array(q)
				p,q = [],[]
		if p:
			yield np.array(p),np.array(q)
			p,q = [],[]		
# Train the model
model.fit_generator(data_generator(X_train,y_train), steps_per_epoch=600, epochs=20, validation_data=data_generator(X_test,y_test), validation_steps=100)
