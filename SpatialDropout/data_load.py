# Load dataset as train and test sets
# Set numeric type to float32 and Normalize value to [0,1]

x_train = np.expand_dims(x_train.astype(K.floatx()) / 255, axis=-1)
x_test = np.expand_dims(x_test.astype(K.floatx()) / 255, axis=-1)

# Transform lables to one-hot encoding
y_train, y_test = np.expand_dims(y_train, axis=-1), np.expand_dims(y_test, axis=-1)

#Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

train_num = round(x_train.shape[0])
x_train, x_valid = x_train[:train_num, ...], x_train[train_num:, ...]
y_train, y_valid = y_train[:train_num, ...], y_train[train_num:, ...] 
