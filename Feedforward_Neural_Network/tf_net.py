
# coding: utf-8

# # MNIST Classification with a Fully-Connected Net in TensorFlow 

# <h2>Imports / magic (%)</h2>

# In[4]:

import numpy as np
from mnist_loader import load

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import tensorflow as tf
# compiles operation graphs at intermediate steps (instead of at end)
sess = tf.InteractiveSession()


# <h2>Hyper Parameters</h2>

# In[19]:

# Hyper params
INPUT_SIZE = 784
HIDDEN_SIZE = 200
OUTPUT_SIZE = 10

LEARNING_RATE = 0.5
EPOCHS = None
MINIBATCH_SIZE = 5


# ## Load Data and Input Placeholders
# <p>The input matrix has
# <ul>
#     <li>First dimension = training examples
#     <li>Second dimension = features
# </ul></p>
# <p>ie. an m x n matrix has m training examples each with n features.</p>

# In[20]:

# Load data
path = "Data\mnist.pkl.gz"
train_data, validation_data, test_data = load(path)

# Placeholder vars
# float pixel values of images
inputs = tf.placeholder(tf.float32)
# integer target labels of each image
labels = tf.placeholder(tf.int32)

# Change first index of train_data to see other images
plt.imshow(np.reshape(train_data[0][0], (28, 28)))


# <h2>Initialize Layers:</h2>
# <p>Each weight matrix has:
# <ul>
#     <li>First dim = size of previous layer
#     <li>Second dim = size of current layer
# </ul></p>
# 
# <p>for multiplication with inputs. The individual weights are randomly initialized with:
# <ul>
#     <li>Mean = 0
#     <li>Standard devation = 1 / sqrt(n), n = size of previous layer
# </ul></p>
# <p>This initialization prevents saturation in activation function with a narrow distribution and (ideally) leads to faster learning.</p>
# <p>The biases are initialized as an all-zero column vector.</p>

# In[24]:

with tf.name_scope("hidden"):
    weights = tf.Variable(
        tf.truncated_normal([INPUT_SIZE, HIDDEN_SIZE], 
                             stddev=(1 / (INPUT_SIZE ** 0.5))),
        name="weights")
    biases = tf.Variable(tf.zeros([HIDDEN_SIZE]), name="biases")
    hidden = tf.sigmoid(tf.matmul(inputs, weights) + biases)

with tf.name_scope("output"):
    weights2 = tf.Variable(
        tf.truncated_normal([HIDDEN_SIZE, OUTPUT_SIZE], 
                             stddev=(1 / (HIDDEN_SIZE ** 0.5))),
        name="weights")
    biases2 = tf.Variable(tf.zeros([OUTPUT_SIZE]), name="biases")
    # logits: unscaled outputs of layer (ie no activation)
    # softmax scaling on output is computed along with error later on
    logits = tf.matmul(hidden, weights2) + biases2


# ## Initialize Training Operations

# In[26]:

# Softmax activation with cross entropy cost
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name="xent")
cost = tf.reduce_mean(xent, name="cost")
# Counter incremented each step of gradient descent (why?)
global_step = tf.Variable(0, name="global_step", trainable=False)
# Descent algorithm
sgd = tf.train.GradientDescentOptimizer(LEARNING_RATE)

train = sgd.minimize(cost, global_step=global_step)


# In[ ]:

# All variables should have been constructed by now
sess.run(tf.init_all_variables())


# ## Training

# In[2]:

for i in range(EPOCHS):
    # Returns activations from "train" and "cost"
    _, cost_val = sess.run([train, cost], feed_dict=)


# In[ ]:



