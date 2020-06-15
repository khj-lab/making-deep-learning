import sys, os

import numpy as np
from PIL import Image

from dataset.mnist import load_mnist


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    img = x_train[0]
    label = y_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28,28)
    img_show(img)