model = Sequential()
model.add(ZeroPadding2D(input_shape=(28, 28, 1), name='Input'))

model.add(Conv2D(filters=32, strides=(1, 1), kernel_size=3, activation='relu', padding='same', name='Conv-1'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, strides=(1, 1), kernel_size=3, activation='relu', padding='same', name='Conv-2'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, name='Pool-1'))
model.add(SpatialDropout2D(rate=0.8, name='Dropout-1'))

model.add(Conv2D(filters=64, strides=(1, 1), kernel_size=3, activation='relu', padding='same', name='Conv-3'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, name='Pool-2'))
model.add(SpatialDropout2D(rate=0.8, name='Dropout-2'))

model.add(Conv2D(filters=64, strides=(1, 1), kernel_size=3, activation='relu', padding='same', name='Conv-4'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, name='Pool-3'))
model.add(SpatialDropout2D(rate=0.8, name='Dropout-3'))

model.add(Conv2D(filters=128, strides=(1, 1), kernel_size=3, activation='relu', padding='same', name='Conv-5'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, name='Pool-4'))
model.add(SpatialDropout2D(rate=0.8, name='Dropout-4'))

model.add(Flatten(name='Flatten'))
model.add(Dense(units=128, activation='relu', name='Dense-1'))
model.add(BatchNormalization())

model.add(Dropout(rate=0.5, name='Dense-Dropout'))
model.add(Dense(units=10, activation='softmax', name='Softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

t = now()
history = model.fit(x=x_train, y=y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test))

# Calculating loss and accuracy
# train
tr_loss, tr_accuracy = model.evaluate(x_train, y_train)
# tr_loss = 0.039, tr_accurary = 0.98845
# test
te_loss, te_accuracy = model.evaluate(x_test, y_test)
# te_loss = 0.042, te_accurary = 0.9861
