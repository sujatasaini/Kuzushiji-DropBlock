#  Plot the graph

import matplotlib.pyplot as plt
%matplotlib inline

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
