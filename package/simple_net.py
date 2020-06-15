import numpy as np

from common import (
    softmax, 
    cross_entropy_error,
    numerical_gradient
)

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return x @ self.W

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == "__main__":
    net = SimpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))

    t = np.array([0,0,1])
    print(net.loss(x, t))