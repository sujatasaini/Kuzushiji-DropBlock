# Import the library
from __future__ import print_function
import keras
import datetime
import keras.backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SpatialDropout2D
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, BatchNormalization
from keras import utils as np_utils
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(2019)

now = datetime.datetime.now
