"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import random
from a3_helper import svm_loss, softmax_loss
from eecs598 import Solver


def hello_fully_connected_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from fully_connected_networks.py!")


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        #############################################################################
        # TODO: Implement the linear forward pass. Store the result in out. You     #
        # will need to reshape the input into rows.                                 #
        #############################################################################
        # Replace "pass" statement with your code
        x_tmp = x.clone().view(x.shape[0], -1)
        out = x_tmp @ w + b
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        #############################################################################
        # TODO: Implement the linear backward pass.                                 #
        #############################################################################
        # Replace "pass" statement with your code
        x_temp = x.clone().view(x.shape[0], -1)
        db = dout.sum(dim=0)
        dw = x_temp.T @ dout
        dx = dout @ w.T
        dx = dx.reshape(x.shape)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
        #############################################################################
        # TODO: Implement the ReLU forward pass.                                    #
        # You should not change the input tensor with an in-place operation.        #
        #############################################################################
        # Replace "pass" statement with your code
        out = x.clone()
        out[out < 0] = 0
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        #############################################################################
        # TODO: Implement the ReLU backward pass.                                   #
        # You should not change the input tensor with an in-place operation.        #
        #############################################################################
        relu_grad = (x > 0).float()

        dx = dout * relu_grad
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=torch.float32,
        device="cpu",
    ):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be performed using
          this datatype. float is faster but less accurate, so you should use
          double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        ###########################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights   #
        # should be initialized from a Gaussian centered at 0.0 with              #
        # standard deviation equal to weight_scale, and biases should be          #
        # initialized to zero. All weights and biases should be stored in the     #
        # dictionary self.params, with first layer weights                        #
        # and biases using the keys 'W1' and 'b1' and second layer                #
        # weights and biases using the keys 'W2' and 'b2'.                        #
        ###########################################################################
        # Replace "pass" statement with your code
        self.params["W1"] = weight_scale * torch.randn(
            input_dim, hidden_dim, dtype=dtype, device=device
        )
        self.params["W2"] = weight_scale * torch.randn(
            hidden_dim, num_classes, dtype=dtype, device=device
        )

        # Initialize biases for the first layer (b1) and second layer (b2) to zeros
        self.params["b1"] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params["b2"] = torch.zeros(num_classes, dtype=dtype, device=device)
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################

    def save(self, path):
        checkpoint = {
            "reg": self.reg,
            "params": self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint["params"]
        self.reg = checkpoint["reg"]
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Tensor of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ###########################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the   #
        # class scores for X and storing them in the scores variable.             #
        ###########################################################################
        # Replace "pass" statement with your code
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        h1, h1_cache = Linear_ReLU.forward(X, W1, b1)
        h2, h2_cache = Linear.forward(h1, W2, b2)
        scores = h2
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ###########################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss #
        # in the loss variable and gradients in the grads dictionary. Compute data#
        # loss using softmax, and make sure that grads[k] holds the gradients for #
        # self.params[k]. Don't forget to add L2 regularization!                  #
        #                                                                         #
        # NOTE: To ensure that your implementation matches ours and you pass the  #
        # automated tests, make sure that your L2 regularization does not include #
        # a factor of 0.5.                                                        #
        ###########################################################################
        # Replace "pass" statement with your code
        from a3_helper import svm_loss, softmax_loss

        softmax_loss, dx = softmax_loss(scores, y)
        dx, dw2, db2 = Linear.backward(dx, h2_cache)
        dx, dw1, db1 = Linear_ReLU.backward(dx, h1_cache)

        loss = softmax_loss + self.reg * (torch.sum(W1**2) + torch.sum(W2**2))

        dw1 += 2 * self.reg * W1
        dw2 += 2 * self.reg * W2

        grads = {"W1": dw1, "W2": dw2, "b1": db1, "b2": db2}
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################

        return loss, grads
import torch

# 假设这些辅助函数存在于 a3_helper.py 或类似文件中
# from a3_helper import softmax_loss, Linear, Linear_ReLU, Dropout

class FullyConnectedNet(object):
    """
    一个具有任意数量隐藏层、ReLU非线性激活函数和Softmax损失函数的
    全连接神经网络。
    对于一个L层的网络，其架构为：

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    其中 dropout 是可选的，并且 {...} 模块重复 L - 1 次。

    与上面的 TwoLayerNet 类似，可学习的参数存储在 self.params 字典中，
    并将使用 Solver 类进行学习。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=0.0,
        reg=0.0,
        weight_scale=1e-2,
        seed=None,
        dtype=torch.float,
        device="cpu",
    ):
        """
        初始化一个新的 FullyConnectedNet。

        输入:
        - hidden_dims: 一个整数列表，给出每个隐藏层的大小。
        - input_dim: 一个整数，给出输入的大小。
        - num_classes: 一个整数，给出要分类的类别数。
        - dropout: 0到1之间的标量，给出使用dropout的概率。
          如果dropout=0，则网络不应使用dropout。
        - reg: 标量，给出L2正则化强度。
        - weight_scale: 标量，给出权重随机初始化的标准差。
        - seed: 如果不是None，则将此随机种子传递给dropout层。
          这将使dropout层具有确定性，以便我们可以对模型进行梯度检查。
        - dtype: 一个torch数据类型对象；所有计算都将使用此数据类型执行。
          float更快但不太精确，因此您应该使用double进行数值梯度检查。
        - device: 用于计算的设备，'cpu'或'cuda'。
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: 初始化网络的参数，将所有值存储在 self.params 字典中。         #
        # 将第一层的权重和偏置存储在 W1 和 b1 中；第二层使用 W2 和 b2，依此类推。 #
        # 权重应从以0为中心、标准差等于 weight_scale 的正态分布中初始化。     #
        # 偏置应初始化为零。                                                    #
        ############################################################################
        # 将所有层的维度整合到一个列表中，方便循环处理
        all_dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(self.num_layers):
            # 第 i+1 层的权重和偏置
            W_key = f"W{i+1}"
            b_key = f"b{i+1}"

            # 获取当前层的输入和输出维度
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]

            # 初始化权重 W
            self.params[W_key] = weight_scale * torch.randn(
                in_dim, out_dim, device=device, dtype=dtype
            )

            # 初始化偏置 b
            self.params[b_key] = torch.zeros(out_dim, device=device, dtype=dtype)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # 使用dropout时，我们需要为每个dropout层传递一个dropout_param字典，
        # 以便该层知道dropout概率和模式（训练/测试）。
        # 你可以为每个dropout层传递相同的dropout_param。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

    def save(self, path):
        checkpoint = {
            "reg": self.reg,
            "dtype": self.dtype,
            "params": self.params,
            "num_layers": self.num_layers,
            "use_dropout": self.use_dropout,
            "dropout_param": self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint["params"]
        self.dtype = dtype
        self.reg = checkpoint["reg"]
        self.num_layers = checkpoint["num_layers"]
        self.use_dropout = checkpoint["use_dropout"]
        self.dropout_param = checkpoint["dropout_param"]

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        为全连接网络计算损失和梯度。
        输入/输出：与上面的TwoLayerNet相同。
        """
        X = X.to(self.dtype)
        mode = "test" if y is None else "train"

        # 为dropout参数设置训练/测试模式，因为它们在训练和测试期间的行为不同。
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: 实现全连接网络的前向传播，为X计算类别分数并将其存储在scores变量中。#
        #                                                                          #
        # 使用dropout时，你需要将self.dropout_param传递给每个dropout前向传播。     #
        ############################################################################
        # 存储每一层的缓存(cache)，以便在反向传播中使用
        caches = {}
        current_input = X

        # 对前 L-1 层进行前向传播 (Linear -> ReLU -> Dropout)
        for i in range(1, self.num_layers):
            W, b = self.params[f"W{i}"], self.params[f"b{i}"]

            # Linear-ReLU 层
            current_input, layer_cache = Linear_ReLU.forward(current_input, W, b)
            caches[f"layer{i}"] = layer_cache

            # 可选的 Dropout 层
            if self.use_dropout:
                current_input, dropout_cache = Dropout.forward(
                    current_input, self.dropout_param
                )
                caches[f"dropout{i}"] = dropout_cache

        # 对最后一层进行前向传播 (Linear)
        W_last = self.params[f"W{self.num_layers}"]
        b_last = self.params[f"b{self.num_layers}"]
        scores, last_layer_cache = Linear.forward(current_input, W_last, b_last)
        caches[f"layer{self.num_layers}"] = last_layer_cache
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # 如果是测试模式，提前返回
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: 实现全连接网络的反向传播。将损失存储在loss变量中，将梯度存储在   #
        # grads字典中。使用softmax计算数据损失，并确保grads[k]持有self.params[k] #
        # 的梯度。不要忘记添加L2正则化！                                       #
        # 注意：为确保你的实现与我们的匹配并通过自动化测试，请确保你的L2正则化   #
        # 包含一个0.5的因子，以简化梯度的表达式。                              #
        ############################################################################
        # 假设 a3_helper.py 中有 softmax_loss 函数
        from a3_helper import svm_loss, softmax_loss

        # 1. 计算损失 (数据损失 + 正则化损失)
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.0
        for i in range(1, self.num_layers + 1):
            W = self.params[f"W{i}"]
            reg_loss += 0.5 * self.reg * torch.sum(W * W)
        loss = data_loss + reg_loss

        # 2. 反向传播
        # 从最后一层开始
        cache_last = caches[f"layer{self.num_layers}"]
        dx, dW, db = Linear.backward(dscores, cache_last)
        grads[f"W{self.num_layers}"] = dW
        grads[f"b{self.num_layers}"] = db

        # 循环反向传播 L-1 到 1 层
        for i in range(self.num_layers - 1, 0, -1):
            # 可选的 Dropout 反向传播
            if self.use_dropout:
                dropout_cache = caches[f"dropout{i}"]
                dx = Dropout.backward(dx, dropout_cache)

            # Linear-ReLU 反向传播
            layer_cache = caches[f"layer{i}"]
            dx, dW, db = Linear_ReLU.backward(dx, layer_cache)
            grads[f"W{i}"] = dW
            grads[f"b{i}"] = db

        # 3. 为权重梯度添加正则化项
        for i in range(1, self.num_layers + 1):
            W = self.params[f"W{i}"]
            grads[f"W{i}"] += self.reg * W
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    ##############################################################################
    # TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
    # 50% accuracy on the validation set.                                        #
    ##############################################################################
    solver = None
    
    solver = Solver(model, data_dict,
                   optim_config={"learning_rate": 1,},
                   lr_decay=0.95,
                   batch_size = 1000,
                   num_epochs = 50,
                   print_every = 100,
                   print_acc_every =100,
                   verbose = True,
                   device = device,
    )
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return solver


def get_three_layer_network_params():
    ############################################################################
    # TODO: Change weight_scale and learning_rate so your model achieves 100%  #
    # training accuracy within 20 epochs.                                      #
    ############################################################################
    weight_scale = 6e-2  # Experiment with this!
    learning_rate = 0.84  # Experiment with this!
    # Replace "pass" statement with your code
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return weight_scale, learning_rate


def get_five_layer_network_params():
    ############################################################################
    # TODO: Change weight_scale and learning_rate so your model achieves 100%  #
    # training accuracy within 20 epochs.                                      #
    ############################################################################
    learning_rate = 0.464  # Experiment with this!
    weight_scale = 8e-2  # Experiment with this!
    # Replace "pass" statement with your code
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", torch.zeros_like(w))

    next_w = None
    #############################################################################
    # TODO: Implement the momentum update formula. Store the updated value in   #
    # the next_w variable. You should also use and update the velocity v.       #
    #############################################################################
    # Replace "pass" statement with your code
    lr = config["learning_rate"]
    mu = config["momentum"]

    v = mu * v - lr * dw
    next_w = w + v
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", torch.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # Replace "pass" statement with your code
    lr = config["learning_rate"]
    decay_rate = config["decay_rate"]
    eps = config["epsilon"]
    cache = config["cache"]

    # 更新二阶矩（平方梯度的移动平均）
    cache = decay_rate * cache + (1 - decay_rate) * (dw**2)

    # 参数更新
    next_w = w - (lr * dw) / (torch.sqrt(cache) + eps)
    
    config["cache"] = cache
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", torch.zeros_like(w))
    config.setdefault("v", torch.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    #############################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in   #
    # the next_w variable. Don't forget to update the m, v, and t variables     #
    # stored in config.                                                         #
    #                                                                           #
    # NOTE: In order to match the reference output, please modify t _before_    #
    # using it in any calculations.                                             #
    #############################################################################
    # Replace "pass" statement with your code
    
    lr = config["learning_rate"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    eps = config["epsilon"]
    m = config["m"]
    v = config["v"]
    t = config["t"]
    t += 1

    # 一阶、二阶动量
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)

    # 偏差修正
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # 参数更新
    next_w = w - lr * m_hat / (torch.sqrt(v_hat) + eps)

    # 保存更新值
    config["m"] = m
    config["v"] = v
    config["t"] = t
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with probability p.
          - mode: 'test' or 'train'. If the mode is train, then perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed makes this
          function deterministic, which is needed for gradient checking but not
          in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
          mask that was used to multiply the input; in test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping** a neuron
        output; this might be contrary to some sources, where it is referred to
        as the probability of keeping a neuron output.
        """
        p, mode = dropout_param["p"], dropout_param["mode"]
        if "seed" in dropout_param:
            torch.manual_seed(dropout_param["seed"])

        mask = None
        out = None

        if mode == "train":
            ###########################################################################
            # TODO: Implement training phase forward pass for inverted dropout.       #
            # Store the dropout mask in the mask variable.                            #
            ###########################################################################
            # Replace "pass" statement with your code
            keep_prob = 1 - p
            # 创建一个与输入 x 形状相同的二元掩码 (binary mask)
            # 掩码中的值为 1 (保留) 的概率为 keep_prob
            # 我们需要将布尔掩码转换为与 x 相同的数据类型以便进行乘法运算
            mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
            # 应用掩码，并除以 keep_prob 来进行缩放 (这是“反向 dropout”的关键)
            # 这样做的目的是为了在测试时不需要进行任何操作
            out = (x * mask) / keep_prob
            ###########################################################################
            #                             END OF YOUR CODE                            #
            ###########################################################################
        elif mode == "test":
            ###########################################################################
            # TODO: Implement the test phase forward pass for inverted dropout.       #
            ###########################################################################
            # Replace "pass" statement with your code
            out = x
            ###########################################################################
            #                             END OF YOUR CODE                            #
            ###########################################################################

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param["mode"]

        dx = None
        if mode == "train":
            ###########################################################################
            # TODO: Implement training phase backward pass for inverted dropout       #
            ###########################################################################
            # Replace "pass" statement with your code
            # 计算保留概率
            keep_prob = 1 - dropout_param["p"]
            # 只将梯度传回给在前向传播中被“保留”的神经元
            # 同样需要进行与前向传播中相同的缩放
            dx = (dout * mask) / keep_prob
            ###########################################################################
            #                            END OF YOUR CODE                             #
            ###########################################################################
        elif mode == "test":
            dx = dout
        return dx
