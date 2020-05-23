import numpy as np

def AND(x1, x2):
    """
    ANDゲート

    Args:
        x1: 第一引数
        x2: 第二引数

    Returns:
        0 or 1:
    """
    x = np.array([x1, x2])
    weight = np.array([0.5, 0.5])
    bias = -0.7
    tmp = x @ weight + bias
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def NAND(x1, x2):
    """
    NANDゲート

    Args:
        x1: 第一引数
        x2: 第二引数

    Returns:
        0 or 1:
    """
    x = np.array([x1, x2])
    weight = np.array([-0.5, -0.5])
    bias = 0.7
    tmp = x @ weight + bias
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def OR(x1, x2):
    """
    ORゲート

    Args:
        x1: 第一引数
        x2: 第二引数

    Returns:
        0 or 1:
    """
    x = np.array([x1, x2])
    weight = np.array([0.5, 0.5])
    bias = -0.2
    tmp = x @ weight + bias
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def XOR(x1, x2):
    """
    XORゲート
    AND,NAND,ORと違い、XORは非線形な領域を作る。
    また、この実装は２層のパーセプトロンといえる。

    Args:
        x1: 第一引数
        x2: 第二引数

    Returns:
        0 or 1:
    """
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y