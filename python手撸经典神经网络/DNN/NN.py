import numpy as np
import matplotlib.pyplot as plt
from activeFunctions import *


def nn_init(nn_architecture, seed=99):
    """
    初始化神经网络，以字典形式返回初始化后的系数 W(i) b(i)
    :param nn_architecture:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']

        params_values['W' + str(layer_idx)] = np.random.rand(
            layer_output_size, layer_input_size)*0.1
        params_values['b' + str(layer_idx)] = np.random.rand(
            layer_output_size, 1)*0.1

    return params_values


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="sigmoid"):
    """
    单层神经网络前向传播
    :param A_prev: input a(l-1)
    :param W_curr: parameters W(l)
    :param b_curr: bias b(l)
    :param activation: active funcs
    :return: output a(l) z(l)   z(l) = np.dot(W_curr, A_prev) + b_curr
    """
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation == "sigmoid":
        active_func = sigmoid
    elif activation == "relu":
        active_func = relu
    elif activation == "softmax":
        active_func = softmax
    else:
        raise Exception('Non-supported active function')

    return active_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    """
    前向传播
    :param X: input features
    :param params_values: parameters of nn (W(i) b(i))
    :param nn_architecture:
    :return: output, state of hidden layers
    """
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1;
        A_prev = A_curr

        activation = layer["active_func"]
        W_curr = params_values['W' + str(layer_idx)]
        b_curr = params_values['b' + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activation)

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    return A_curr, memory


def get_cost_value(Y_hat, Y, lossfunc="CrossEntropy"):
    """
    计算损失函数
    :param Y_hat: 神经网络预测结果
    :param Y:  训练数据标签
    :return: loss
    """
    m = Y_hat.shape[1]
    if lossfunc == "CrossEntropy":
        cost = -1 / m * np.sum(np.multiply(Y, np.log(Y_hat)) + np.multiply(1-Y, np.log(1 - Y_hat)))
    elif lossfunc == "MSE":
        cost = np.sum((Y_hat - Y)*(Y_hat - Y))/(2 * m)
    elif lossfunc == "LogLikelihood":
        cost = -1 / m * np.sum(np.multiply(Y, np.log(Y_hat)))

    return cost


# def get_accuracy_value(Y_hat, Y):


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="sigmoid"):
    """
    单层神经网络反向传播
    :param dA_curr: 反向传播至当前层输出的梯度
    :param W_curr: 当前层的系数
    :param b_curr: 当前层偏置
    :param Z_curr: 当前层激活函数的输入
    :param A_prev: 上一层的输出
    :param activation:
    :return:
    """
    m = A_prev.shape[1]

    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    elif activation is "softmax":
        backward_activation_func = softmax_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture, lossfunc="CrossEntropy"):
    """
    整合后的反向传播
    """
    grads_values = {}
    m = Y_hat.shape[1]

    if lossfunc == "CrossEntropy":
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1-Y, 1-Y_hat))
    elif lossfunc == "MSE":
        dA_prev = Y_hat - Y
    elif lossfunc == "LogLikelihood":
        dA_prev = - np.divide(Y, Y_hat)

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        active_func = layer["active_func"]

        dA_curr = dA_prev

        A_prev = memory['A' + str(layer_idx_prev)]
        Z_curr = memory['Z' + str(layer_idx_curr)]
        W_curr = params_values['W' + str(layer_idx_curr)]
        b_curr = params_values['b' + str(layer_idx_curr)]
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, active_func)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    return grads_values


def grad_check(X, Y, params_values, grads_values, nn_architecture, loss_func):
    """
    以数值方法检测反向传播梯度是否计算正确
    """
    delta = 0.0001
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        for weight in ['W' + str(layer_idx), 'b' + str(layer_idx)]:
            n = params_values[weight].size
            for i in range(n):
                w = params_values[weight].flat[i]
                params_values[weight].flat[i] = w + delta
                Y_hat, _ = full_forward_propagation(X, params_values, nn_architecture)
                loss_positive = get_cost_value(Y_hat, Y, loss_func)

                params_values[weight].flat[i] = w - delta
                Y_hat, _ = full_forward_propagation(X, params_values, nn_architecture)
                loss_negative = get_cost_value(Y_hat, Y, loss_func)
                params_values[weight].flat[i] = w

                grad_numerical = (loss_positive - loss_negative) / (2 * delta)
                grad_analytic = grads_values['d'+weight].flat[i]

                # compare the relative error between analytical and numerical gradients
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic + 0.0000001)

                if rel_error > 0.01:
                    print('WARNING %f - %f => %e ' % (grad_numerical, grad_analytic, rel_error))


def update(params_values, grads_values, nn_architecture, learning_rate):
    """
    update parameters
    """
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values


def train(X, Y, nn_architecture, epoch, learning_rate, loss_func="CrossEntropy"):
    params_values = nn_init(nn_architecture, 10)
    cost_history = []

    for i in range(epoch):
        Y_hat, memory = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_cost_value(Y_hat, Y, loss_func)
        cost_history.append(cost)
        grads_values = full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture, loss_func)
        grad_check(X, Y, params_values, grads_values, nn_architecture, loss_func)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        if i % 50 == 0:
            print("Epoch: %d | Loss: %f" % (i, cost))
    return params_values, cost_history


# define architecture of neural network
layers = [
    {'input_dim': 2, 'output_dim': 2, 'active_func': "sigmoid"},
    {'input_dim': 2, 'output_dim': 2, 'active_func': "sigmoid"},
    {'input_dim': 2, 'output_dim': 2, 'active_func': "sigmoid"}
]

X = np.array([[1, 1, 0, 0], [0, 1, 0, 1]])
Y = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
train(X, Y, layers, 500, 0.1, "LogLikelihood")
# train(X, Y, layers, 500, 0.5, "MSE")
# train(X, Y, layers, 500, 0.1, "CrossEntropy")
