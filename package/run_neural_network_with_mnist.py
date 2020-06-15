import os
import pickle

import numpy as np

from common import softmax, sigmoid
from dataset.mnist import load_mnist

def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, y_test

def init_network():
    abs_path = os.path.join(os.path.dirname(__file__), "dataset", "sample_weight.pkl")
    with open(abs_path, "rb") as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = x @ W1 + b1
    z1 = sigmoid(a1)
    a2 = z1 @ W2 + b2
    z2 = sigmoid(a2)
    a3 = z2 @ W3 + b3
    y = softmax(a3)

    return y

if __name__ =="__main__":
    # バッチなしで実行
    train, test = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(train)):
        y = predict(network, train[i])
        p = np.argmax(y)
        if p == test[i]:
            accuracy_cnt += 1

    print("no batch Accuracy: " + str(float(accuracy_cnt) / len(train)))

    
    # バッチありで実行
    train, test = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(train), batch_size):
        x_batch = train[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == test[i:i+batch_size])

    print("batch Accuracy: " + str(float(accuracy_cnt) / len(train)))