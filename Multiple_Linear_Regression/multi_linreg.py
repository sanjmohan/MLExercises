# Multiple Linear Regression on Housing Price given 13 attributes
# First time using theano!

# TODO: theano symbolic differentiation

import numpy as np
import theano
from theano import tensor as T
from theano import function
import matplotlib.pyplot as plt


def load_data():
    # load first 13 values as feature, load last value as target
    # matrix of all examples (row = example, column = feature)
    x = []
    # matrix of all target values
    y = []

    # read file of data
    with open('Housing/housing_data.txt', 'r') as f:
        for line in f:
            if line != "":
                ex = [float(i) for i in line.split(' ') if i != '']
                # first "attribute" is 1 as placeholder for theta_0
                x.append([1] + ex[:-1])
                y.append(ex[-1])

    return x, y


x_in, y_in = load_data()
num_test = 100
num_train = len(x_in) - num_test

# leave last num_test examples for testing
train_x = np.asarray(x_in[:-num_test])
train_y = np.asarray(y_in[:-num_test]).reshape(num_train, 1)

print("Data Size: %d" % len(x_in))
print("Number of Features: %d" % (len(x_in[0])-1)) # first "feature" is placeholder
print("Training Size: %d" % num_train)
print("Test Size: %d" % num_test)

# shared var - column vector with length = number of independent attributes
theta = theano.shared(np.zeros((train_x.shape[1], 1)))

# symbolic inputs to cost function
x = T.matrix('x')
y = T.matrix('y')

# Compute predictions (feedforward, hypothesis, etc.)
pred = T.dot(x, theta)

# least mean square cost function
c = (1 / (2 * num_train)) * ((pred - y) ** 2).sum()

# function([symbolic inputs], output, name=name)
cost = theano.function([x, y], c, name="cost")

# least mean square cost partial derivatives
# grad w/respect to theta_j = 1/m * sum( (x_i - y_i) * x_i_j )
# gc = 1/num_train * T.dot(x.T, (pred - y))

# gradient descent update function
lr = 0.000006
# (shared var to update, expression representing update)
updates = [(theta, theta - lr * T.grad(c, theta))]
grad_desc = theano.function([x, y], theta, updates=updates, name="grad_desc")

# iterate through gradient descent fixed number of times
# list of costs at each iteration
accuracy = []
iters = 3000
for i in range(iters):
    grad_desc(train_x, train_y)
    accuracy.append(cost(train_x, train_y))
    if i % (iters // 20) == 0:
        print("Iteration", i)

print("Minimum Cost: %d" % min(accuracy))
# show (hopefully) decreasing cost
plt.plot(range(iters), accuracy)
plt.show()