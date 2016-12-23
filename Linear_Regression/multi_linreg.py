# Multiple Linear Regression on Housing Price given 13 attributes
# First time using theano!

import numpy as np
import theano
from theano import tensor as T
from theano import function
import matplotlib.pyplot as plt


def load_data():
    # Features separated by space, examples separated by line breaks
    # Load first 13 values as feature, load last value as target
    # Matrix of all examples (row = example, column = feature)
    x = []
    # matrix of all target values
    y = []

    # read file of data
    with open('Data/housing_data.txt') as f:
        for line in f:
            if line != "":
                ex = [float(i) for i in line.split(' ') if i != '']
                # first "attribute" is 1 as placeholder for theta_0
                x.append([1] + ex[:-1])
                y.append(ex[-1])

    return x, y


def normalize(data):
    # Rescale features to lie on range [0, 1]
    # Transpose => each row is a feature
    xT = np.asarray(data).T
    # Skip first placeholder "feature"
    for i in range(1, len(xT)):
        feature = xT[i]
        min_val = min(feature)
        max_val = max(feature)
        feature = (feature - min_val) / (max_val - min_val)
        xT[i] = feature
    return (xT.T).tolist()


x_in, y_in = load_data()
x_in = normalize(x_in)
num_test = 100
num_train = len(x_in) - num_test

# leave last num_test examples for testing
train_x = np.asarray(x_in[:-num_test])
train_y = np.asarray(y_in[:-num_test]).reshape(num_train, 1)

test_x = np.asarray(x_in[-num_test:])
test_y = np.asarray(y_in[-num_test:]).reshape(num_test, 1)

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
c = 0.5 * T.mean((pred - y) ** 2)

# function([symbolic inputs], output, name=name)
cost = theano.function([x, y], c, name="cost")

# least mean square cost partial derivatives
# grad w/respect to theta_j = 1/m * sum( (x_i - y_i) * x_i_j )
# gc = 1/num_train * T.dot(x.T, (pred - y))

# gradient descent update function
# learning rate
lr = 0.01
print("Learning Rate: %f" % lr)
# update format: (shared var to update, expression representing update)
# featuring symbolic differentiation!
updates = [(theta, theta - lr * T.grad(c, theta))]
grad_desc = theano.function([x, y], theta, updates=updates, name="grad_desc")

# iterate through gradient descent fixed number of times
# list of costs at each iteration
accuracy = []
iters = 3000
for i in range(iters):
    grad_desc(train_x, train_y)
    accuracy.append(cost(train_x, train_y))
    if i % (iters // 20) == 0 or i == iters - 1:
        print("Iteration %d" % (i+1))

print("Minimum Training Cost: %f" % min(accuracy))
print("Test Cost: %f" % cost(test_x, test_y))
# show (hopefully) decreasing cost
plt.plot(range(iters), accuracy)
plt.show()


# 300 iters, lr = 0.000006: min cost = 36
# (higher lr explodes)
# 300 iters w/normalization, lr = 0.000006: min cost = 261
# 300 iters w/normalization, lr = 0.01: min cost = 35, test cost = 16
# (higher lr explodes)

# 3000 iters, lr = 0.000006: min cost = 27
# 3000 iters w/normalization: min cost = 178
# 3000 iters w/normalization, lr = 0.01: min cost = 15, test cost = 10