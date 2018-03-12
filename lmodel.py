import numpy as np
import h5py
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset


def load_data():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    test_dataset = h5py.File('test_catvnoncat.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def get_x_y_info(train_x_unshape, test_x_unshape, train_y, test_y):
    m_train = train_x_unshape.shape[0]
    num_px = train_x_unshape.shape[1]
    m_test = test_x_unshape.shape[0]

    print("Number of training examples: " + str(m_train))
    print("Number of testing examples: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_orig shape: " + str(train_x_unshape.shape))
    print("train_y shape: " + str(train_y.shape))
    print("test_x_orig shape: " + str(test_x_unshape.shape))
    print("test_y shape: " + str(test_y.shape))

    return None


def initialize_parameter(layers_dim):
    parameters = {}
    length = len(layers_dim)
    for j in range(1, length):
        a = layers_dim[j]
        b = layers_dim[j - 1]
        parameters["W" + str(j)] = np.random.randn(a, b) / np.sqrt(b)
        parameters["b" + str(j)] = np.zeros((a, 1))

    return parameters


def forward_sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))

    return A


def forward_relu(Z):
    A = np.maximum(0, Z)

    return A


def backward_last_da(Y, last_A):
    Y = Y.reshape(last_A.shape)
    dA = - (np.divide(Y, last_A) - np.divide(1 - Y, 1 - A_last))

    return dA


def backward_relu(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z < 0] = 0

    return dZ


def backward_sigmoid(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ


def forward_propagation_relu_sigmoid(parameters, X, layers_dim):
    length = len(layers_dim)
    storage = {"A0": X}

    for k in range(1, length):
        # storage["Z" + str(k)] = np.dot(parameters["W" + str(k)], storage["A" + str(k-1)])+parameters["b" + str(k)]
        storage["Z" + str(k)] = parameters["W" + str(k)].dot(storage["A" + str(k-1)]) + parameters["b" + str(k)]
        # print("Z: ")
        # print(storage["Z" + str(k)])
        storage["A" + str(k)] = forward_relu(storage["Z" + str(k)])
        # print("A: ")
    last_A = forward_sigmoid(storage["Z" + str(length - 1)])
    storage["A" + str(length - 1)] = last_A

    return storage, last_A


def compute_cost(last_A, Y):
    m = Y.shape[1]
    cost = - (np.dot(Y, np.log(last_A).T) + np.dot(1 - Y, np.log(1 - last_A).T)) / m
    cost = np.squeeze(cost)

    return cost


def update_parameter_sigmoid_relu(parameters, storage, rate, Y):
    length = len(parameters) // 2
    m = Y.shape[1]

    last_A = storage["A" + str(length)]
    last_dA = backward_last_da(Y, last_A)
    last_dZ = backward_sigmoid(last_dA, storage["Z" + str(length)])
    temp = {"dA" + str(length): last_dA,
            "dZ" + str(length): last_dZ}
    # print("lastA: ")
    # print(last_A)
    # print("last dZ: ")
    # print(last_dZ)
    # print("last dA: ")
    # print(last_dA)
    for l in reversed(range(length)):
        temp["dA" + str(l)] = np.dot(parameters["W" + str(l + 1)].T, temp["dZ" + str(l + 1)])
        temp["dW" + str(l + 1)] = np.dot(temp["dZ" + str(l + 1)], storage["A" + str(l)].T) / m
        temp["db" + str(l + 1)] = np.sum(temp["dZ" + str(l + 1)], axis=1, keepdims=True) / m
        if l > 0:
            temp["dZ" + str(l)] = backward_relu(temp["dA" + str(l)], storage["Z" + str(l)])
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - rate * temp["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - rate * temp["db" + str(l + 1)]
        # print("W: " + str(l + 1))
        # print(parameters["W" + str(l + 1)])
        # print("b: " + str(l + 1))
        # print(parameters["b" + str(l + 1)])
    return parameters, temp


def check_accuracy(prediction, Y):
    m = Y.shape[1]
    accuracy = np.sum(prediction == Y) / m * 100
    print("Accuracy: " + str(accuracy) + "%")
    return accuracy


# set default size of plots
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# START!
x_train_unshape, y_train, x_test_unshape, y_test, data_classes = load_data()
get_x_y_info(x_train_unshape, x_test_unshape, y_train, y_test)  # get original data information

# reshape the input X; "-1" -> remaining dimensions
x_train_flatten = x_train_unshape.reshape(x_train_unshape.shape[0], -1).T
x_test_flatten = x_test_unshape.reshape(x_test_unshape.shape[0], -1).T
x_train = x_train_flatten / 255
x_test = x_test_flatten / 255

# x_train, y_train = load_planar_dataset()

# set model structure
layers_dims = [x_train.shape[0], 80, 20, 7, 5, 1]
num_iterations = 3000
learning_rate = 0.0075

# get parameters
np.random.seed(0)
parameters_wb = initialize_parameter(layers_dims)
costs = []
for i in range(num_iterations):
    storage_za, A_last = forward_propagation_relu_sigmoid(parameters_wb, x_train, layers_dims)
    cost_un = compute_cost(A_last, y_train)
    parameters_wb, temp_dwdb = update_parameter_sigmoid_relu(parameters_wb, storage_za,
                                                             learning_rate, y_train)
    if i % 500 == 0:
        print("Cost after iteration %i: %s" % (i, str(cost_un)))
        costs.append(cost_un)
plt.plot(np.squeeze(costs))  # x - cost; y - iteration per tens
plt.show()

# check the train model
no_use_train, train_prediction = forward_propagation_relu_sigmoid(parameters_wb, x_train, layers_dims)
train_prediction = np.where(train_prediction > 0.5, 1, 0)
train_accuracy = check_accuracy(train_prediction, y_train)

# test on test data
no_use_test, test_prediction = forward_propagation_relu_sigmoid(parameters_wb, x_test, layers_dims)
test_prediction = np.where(test_prediction > 0.5, 1, 0)
test_accuracy = check_accuracy(test_prediction, y_test)
