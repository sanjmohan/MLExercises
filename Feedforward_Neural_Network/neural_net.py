# Fully connected feedforward neural network for classification on MNIST se0t

import gzip
import pickle
import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


def load_data(file_name):
    # Load, format mnist data
    def vectorize(digit):
        # Return length 10 column vector with 1 in digit place
        vector = np.zeros((10, 1))
        vector[digit] = 1
        return vector

    file = gzip.open(file_name, "rb")
    training, validation, test = pickle.load(file, encoding="latin1")
    file.close()

    # Training labels are vectors, other labels are digits
    training_images = [np.reshape(array, (784, 1)) for array in training[0]]
    training_labels = [vectorize(digit) for digit in training[1]]
    training_data = zip(training_images, training_labels)

    validation_images = [np.reshape(array, (784, 1)) for array in validation[0]]
    validation_labels = [vectorize(digit) for digit in validation[1]]
    validation_data = zip(validation_images, validation_labels)

    test_images = [np.reshape(array, (784, 1)) for array in test[0]]
    test_labels = [vectorize(digit) for digit in test[1]]
    test_data = zip(test_images, test_labels)

    return list(training_data), list(validation_data), list(test_data)


# Load data
print("Loading Data...")
training, validation, test = load_data("Data/mnist.pkl.gz")

print("Compiling Theano functions...")

# Input and target output matrices
x = T.matrix('x')
y = T.matrix('y')

# Initialize network
# layers - list of nodes in each layer (including input)
layers = [784, 100, 10]
# Params of network
weights = []
biases = []

# Recursively compose feedforward function over each layer
f = x
for i in range(1, len(layers)):
    w = theano.shared(np.random.randn(layers[i], layers[i - 1]))
    # By default, shared vars aren't broadcastable
    b = theano.shared(np.random.randn(layers[i], 1), broadcastable=(False, True))
    weights.append(w)
    biases.append(b)
    f = T.nnet.sigmoid(T.dot(w, f) + b)

feedforward = theano.function([x], f, name="feedforward")

# Regularization factor
alpha = 0.005
# MSE Cost function with L2 Regularization
#c = T.mean((f - y) ** 2) + T.mean(w ** 2)
# Cross Entropy cost function with L2 Regularization
c = T.mean(T.nnet.binary_crossentropy(f, y)) + alpha * T.mean(w ** 2)
cost = theano.function([x, y], c, name="cost")

# Backprop function
learning_rate = 0.4
updates = list((w, w - learning_rate * T.grad(c, w)) for w in weights)
updates += list((b, b - learning_rate * T.grad(c, b)) for b in biases)
backprop = theano.function([x, y], c, updates=updates, name="backprop")


def evaluate(data):
    # Evaluates the accuracy over given data
    accuracy = 0
    for ex in data:
        if feedforward(ex[0]).argmax() == ex[1].argmax():
            accuracy += 1
    return 100 * accuracy / len(data)


# Gradient Descent
epochs = 150
mb_size = 5

print("Epochs: %d" % epochs)
print("Learning Rate: %f" % learning_rate)
print("Alpha (L2): %f" % alpha)
print("Training Size: %d" % len(training))
print("Minibatch Size: %d" % mb_size)
print("Training Network", layers, "...")

# Training accuracy
train_accuracy = []
# Validation accuracy
val_accuracy = []

t_start = time.time()
for epoch in range(epochs):
    print("Epoch %d" % epoch)
    t_elapsed = time.time() - t_start
    if epoch != 0:
        print("Elapsed: %ds; Estimated Left: %ds" % (t_elapsed, t_elapsed/epoch * (epochs-epoch)))

    # Generate minibatches
    np.random.shuffle(training)
    minibatches = [training[i:i+mb_size] for i in range(0, len(training) - mb_size, mb_size)]

    # Update params
    for mb in minibatches:
        # Reformat from ordered pairs to list of inputs, list of outputs
        x, y = tuple(zip(*mb))
        # hstack - list of column vectors to 2d array
        backprop(np.hstack(x), np.hstack(y))

    # Intermediate training and validation accuracies
#    train_accuracy.append(evaluate(training))
    val_accuracy.append(evaluate(validation))

# Test evaluation
print("Test Accuracy: %f" % evaluate(test))

# TODO: Write params to file

# Plot
#plt.plot(range(epochs), train_accuracy, 'b-', label="training accuracy")
plt.plot(range(epochs), val_accuracy, 'r-', label="validation accuracy")
plt.legend()
plt.show()


# [784, 30, 10], mse, learning rate: 0.5, minibatch size: 10, epochs: 50 -- 92.62%, 247s
# [784, 300, 10], mse + l2 reg (alpha = 1), learning rate: 0.3, minibatch size: 5, epochs: 100 -- 93.34%, 2375s
# [784, 100, 10], cross-entropy + l2 reg (alpha = 0.1), learning rate: 0.3, minibatch size: 5, epochs: 150 -- 95.71%, 813s
# [784, 100, 10], cross-entropy + l2 reg (alpha = 0.005), learning rate: 0.4, minibatch size: 5, epochs: 150 -- 96.69%, 793s
