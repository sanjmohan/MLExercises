
# coding: utf-8

# # Keras - imports, data formatting, and 7 lines of code

# In[1]:

import time
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.regularizers import l2

from mnist_loader import load


# In[2]:

# Hyper params and constants
IMAGE_SIZE = 28

INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE
HIDDEN_SIZE = 200
OUTPUT_SIZE = 10

LEARNING_RATE = 0.05
ALPHA = 0.0001
EPOCHS = 10
MINIBATCH_SIZE = 5

LOSS='categorical_crossentropy'

# File to save trained model to
SAVE_FILE = "C:/Users/Sanjay/Desktop/keras-mnist-model"


# In[3]:

# Load data -- from net_tf.py

path = "Data\mnist.pkl.gz"
train_data, validation_data, test_data = load(path)

def vectorize(labels):
    for i, label in enumerate(labels):
        vector = np.zeros(10)
        vector[label] = 1
        labels[i] = vector
    return labels

# Tuples of (img, label) --> list of imgs, list of labels
train_imgs, train_vals = zip(*train_data)
train_labels = vectorize(list(train_vals))
train_imgs = np.vstack(train_imgs)
train_labels = np.vstack(train_labels)

# Tuples of (img, label) --> list of imgs, list of labels
vali_imgs, vali_vals = zip(*validation_data)
vali_labels = vectorize(list(vali_vals))
vali_imgs = np.vstack(vali_imgs)
vali_labels = np.vstack(vali_labels)

# Tuples of (img, label) --> matrix of imgs, list of labels
test_imgs, test_labels = zip(*test_data)
test_imgs = np.vstack(test_imgs)

# Change index of train_imgs to see other images
plt.imshow(np.reshape(train_imgs[5], (IMAGE_SIZE, IMAGE_SIZE)))


# In[4]:

model = Sequential()

model.add(Dense(HIDDEN_SIZE, init='lecun_uniform', input_dim=784, W_regularizer=l2(ALPHA)))
model.add(Activation('relu'))
model.add(Dense(OUTPUT_SIZE, init='lecun_uniform', W_regularizer=l2(ALPHA)))
model.add(Activation('sigmoid'))


# In[5]:

# optimizer= takes either string or optimizer object
model.compile(loss=LOSS, optimizer=SGD(lr=LEARNING_RATE), metrics=['accuracy'])


# In[6]:

# verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
hist = model.fit(train_imgs, train_labels, nb_epoch=EPOCHS, batch_size=MINIBATCH_SIZE, 
          validation_data=(vali_imgs, vali_labels), shuffle=True, verbose=2)
print("Done training")


# In[7]:

print(hist.history)
validation_loss = list(hist.history.values())[0]
plt.plot(range(EPOCHS), validation_loss)


# In[ ]:



