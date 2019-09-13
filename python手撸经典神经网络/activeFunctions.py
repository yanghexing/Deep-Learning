import numpy as np


def sigmoid(Z):
    """
    sigmoid active function
    """
    return 1/(1 + np.exp(-Z))


def sigmoid_backward(dA, Z):
    """
    sigmoid back propagation
    """
    y = sigmoid(Z)
    return dA * y * (1 - y)


def relu(Z):
    """
    relu active function
    """
    return np.maximum(0, Z)


def relu_backward(dA, Z):
    """
    relu back propagation
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def tanh(Z):
    """
    tanh active function
    """
    return 2*sigmoid(2*Z) - 1


def tanh_backward(dA, Z):
    """
    tanh back propagation
    """
    y = tanh(Z)
    return dA * (1 - y * y)


def softmax(Z):
    """
    softmax active function
    :param Z: np.array(m,n) n组数据，每组数据为维度为m的列向量
    """
    e_x = np.exp(Z - np.max(Z))
    sum = np.sum(e_x, axis=0)
    out = e_x/sum
    return out


def softmax_backward(dA, Z):
    """
    softmax back propagation
    :param dA: np.array(m,n) n组数据，每组数据为维度为m的列向量
    :param Z: np.array(m,n) n组数据，每组数据为维度为m的列向量
    """
    m = Z.shape[1]
    out = np.zeros_like(dA)
    for i in range(m):
        x = Z[:, i]
        y = softmax(x)
        out[:, i] = np.dot(np.diag(y)-np.outer(y, y), dA[:, i])
    return out