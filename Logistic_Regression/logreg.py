# Logistic Regression on Dota 2 win/losses given hero picks
# This makes some nice looking cost over training time graphs...

import sys
import numpy as np
import theano
import theano.tensor as T
from theano import function
import matplotlib.pyplot as plt


def load_data(file):
    """
    Read data from given file.
    Return features in x (rows hold examples, columns hold feature)
    Return targets in y
    """
    x, y = [], []
    with open(file, 'r') as f:
        for line in f:
            # Data values separated by spaces
            ex = [int(n) for n in line.split(',') if n != '']
            # [1] as placeholder for bias theta_0 in dot products
            x.append([1] + ex[1:])
            # target, or true output
            # change -1 cases to 0
            y.append(max(ex[0], 1))
    return x, y


def normalize(data):
    """
    Rescale features to lie on range [0, 1]
    Transpose => each row is a feature
    """
    xT = np.asarray(data).T
    # Skip first placeholder "feature"; normalize next 3 columns
    # (all other values are range [-1, 1] already)
    for i in range(1, 4):
        feature = xT[i]
        min_val = min(feature)
        max_val = max(feature)
        feature = (feature - min_val) / (max_val - min_val)
        xT[i] = feature
    return xT.T
    

print("Loading Data...")
# Loaded data in list structure
train_x_in, train_y_in = load_data("Data/dota2Train.csv")
test_x_in, test_y_in = load_data("Data/dota2Test.csv")

# Meta-info / hyper-params
num_train = 1000        # train size
batch_size = 499
num_test = len(test_x_in)   # test size
lr = 0.01             # learning rate
iters = 5000          # number of iterations of gradient descent

train_x = np.asfarray(train_x_in[:num_train])
train_y = np.asfarray(train_y_in[:num_train])
# Column vector
train_y = train_y.reshape(len(train_y), 1)

test_x = np.asfarray(test_x_in)
test_y = np.asfarray(test_y_in)
# Column vector
test_y = test_y.reshape(len(test_y), 1)

train_x = normalize(train_x)
test_x = normalize(test_x)

print("Data loaded\n")

print("Training Size:   %d" % num_train)
print("Batch Size:      %d" % batch_size)
print("Test Size:       %d" % num_test)
print("# of Features:   %d" % (len(train_x[0]) - 1))
print("Learning Rate:   %f" % lr)
print("# of Iterations: %d" % iters)
print()

# Weights (index 0 is essentially a bias) - Column vector
theta = theano.shared(np.zeros((len(train_x[0]), 1)))

# x = inputs, y = true outputs
x = T.matrix('x')
y = T.matrix('y')

# Prediction, hypothesis, logistic regression, the main thing, etc.
pred = T.nnet.sigmoid(T.dot(x, theta))
predfn = function([x], pred, name="predfn")

# Cross entropy cost function (TODO: regularization)
c = T.mean(T.nnet.binary_crossentropy(pred, y))
cost = function([x, y], c, name="cost")

# Gradient Descent
updates = [(theta, theta - lr * T.grad(c, theta))]
train = function([x, y], theta, updates=updates, name="train")
accuracy = []
for i in range(iters):
    if i % (iters//50) == 0 or i+1 == iters:
            print("Iteration: %d" % (i+1))
    # minibatch gradient descent
    # shuffle
    augmented = np.asarray((train_x.T.tolist() + train_y.T.tolist())).T
    np.random.shuffle(augmented)
    train_x = augmented.T[:-1].T
    train_y = augmented.T[-1].reshape(len(train_y), 1)
    # iterate through minibatches
    for rng in range(0, num_train - batch_size, batch_size):
        train(train_x, train_y)
        
    accuracy.append(cost(train_x, train_y))

print("Minimum training cost: %f" % min(accuracy))
print("Test Cost:             %f" % cost(test_x, test_y))

plt.plot(range(iters), accuracy)
plt.show()


# train size=1000, lr=0.01, iters=5000, batch=999, test cost = 0.01320
# train size=1000, lr=0.01, iters=5000, batch=499, test cost = 0.00665