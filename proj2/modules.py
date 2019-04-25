from torch import empty
import math

class Module(object):
    '''
    basic Module class for advanced modules such as ReLu, Linear and Sequential
    '''

    def forward(self, *input):
        # must override
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        # must override
        raise NotImplementedError

    def param(self):
        # only override when the module has parameters
        return []

    def zero_grad(self):
        # only override when the module has parameters
        pass

    def __call__(self, *input):
        if len(input)==1: # for normal modules
            return self.forward(input[0])
        elif len(input)==2: # for MSE loss. the 1st input is prediction and the second one is ground truth
            return self.forward(input[0], input[1])


class ReLU(Module):
    '''
    ReLU module
    '''

    def __init__(self):
        self.input = None

    def forward(self, input):
        '''
        ReLU(x) = max(0, x)
        :param input: Tensor with any shape
        :return: ReLU(x): Tensor with the same shape as the input
        '''
        self.input = input
        return input * (input>0).float()

    def backward(self, gradwrtoutput):
        '''
        ReLU backward:
        The gradient of ReLU is 0 if input<0 else 1
        :param gradwrtoutput: dL/d(output) Tensor with the same shape as input
        :return: dL/d(input): Tensor with the same shape as input
        '''
        grad = empty(*gradwrtoutput.shape)
        grad[self.input>0] = 1
        grad[self.input<=0] = 0
        return grad * gradwrtoutput


class tanh(Module):
    '''
    tanh module
    '''

    def __init__(self):
        self.input = None

    def forward(self, input):
        '''
        tanh(x)
        :param input: Tensor with any shape
        :return: tanh(x): Tensor with the same shape as the input
        '''
        self.input = input
        return input.tanh()

    def backward(self, gradwrtoutput):
        '''
        tanh backward:
        The gradient of ReLU is 1-(tanh(x))^2
        :param gradwrtoutput: dL/d(output) Tensor with the same shape as input
        :return: dL/d(input): Tensor with the same shape as input
        '''
        grad = 1 - (self.input.tanh())**2
        return grad * gradwrtoutput

class Linear(Module):
    '''
    Fully connected layer Module
    '''

    def __init__(self, input_num, output_num, bias=True):
        '''
        Initialize the layer
        :param input_num: the input feature dimension
        :param output_num: the output feature dimension
        :param bias: if bias is needed (default: True)
        '''
        self.input = None
        self.input_num = input_num
        self.output_num = output_num
        self.ifbias = bias

        bound = 1/math.sqrt(self.input_num)
        self.weight = empty((self.input_num, self.output_num)).uniform_(-bound, bound)
        if self.ifbias:
            self.bias = empty((1, self.output_num)).uniform_(-bound, bound)

        self.weight_grad = 0
        if self.ifbias:
            self.bias_grad = 0

    def forward(self, input):
        '''
        Linear forward: XW + b
        X's shape: (batchsize, input_feature_dimension)
        W's shape: (input_feature_dimension, output_feature_dimension)
        b's shape: (1, output_feature_dimension)

        :param input: X (batchsize, input_feature_dimension)
        :return: XW + b (batchsize, output_feature_dimension)
        '''

        self.input = input
        if self.ifbias:
            return input.matmul(self.weight) + self.bias
        else:
            return input.matmul(self.weight)

    def backward(self, gradwrtoutput):
        '''
        Linear backward
        :param gradwrtoutput dL/dy: (batchsize, output_num)
        :return: dL/dX: (batchsize, input_num)
        '''
        if self.ifbias:
            ones = empty((1, gradwrtoutput.shape[0])).fill_(1)
            self.bias_grad += ones.matmul(gradwrtoutput)  # dL/db = 1^T * dL/dy
        self.weight_grad += self.input.t().matmul(gradwrtoutput)  # dL/dW = X^T * dL/dy
        return gradwrtoutput.matmul(self.weight.t())  # dL/dX = dL/dy * W^T

    def param(self):
        '''
        return the parameters and the corresponding gradient
        :return: a list [[weight, weight gradient], [bias, bias gradient]]
        '''
        if self.ifbias:
            return [[self.weight, self.weight_grad], [self.bias, self.bias_grad]]
        else:
            return [[self.weight, self.weight_grad]]

    def zero_grad(self):
        '''
        set all gradients zero
        :return:
        '''
        self.weight_grad = 0
        if self.ifbias:
            self.bias_grad = 0

class Sequential(Module):
    '''
    Sequential module: connect modules in a sequence
    '''

    def __init__(self, *modules):
        self.modules = modules
        self.input = None

    def forward(self, input):
        '''
        Data flows through all the module in Sequential
        :param input: (batchsize, input_feature_dimension)
        :return: output: (batchsize, output_feature_dimension)
        '''
        self.input = input
        y = input
        for m in self.modules:
            y = m(y)
        return y

    def backward(self, gradwrtoutput):
        '''
        Gradients flow back through all modules
        :param gradwrtoutput: (batchsize, output_feature_dimension)
        :return: gradwrtinput: (batchsize, input_feature_dimension)
        '''
        reversed_modules = self.modules[::-1]
        grad = gradwrtoutput
        for m in reversed_modules:
            grad = m.backward(grad)
        return grad

    def param(self):
        '''
        :return: all the parameters and parameter gradients for each module
        (modules with no parameters will return empty list []
        '''
        params = []
        for m in self.modules:
            params.append(m.param())
        return params

    def zero_grad(self):
        '''
        zero all parameter gradient for each module
        '''
        for m in self.modules:
            m.zero_grad()

class LossMSE(Module):
    '''
    Mean squre error(MSE) loss module
    '''
    def __init__(self,model):
        '''
        :param model: a module is needed for gradient backward
        '''
        self.pred = None
        self.gt = None
        self.model = model

    def forward(self, pred, gt):
        '''
        MSE forward
        :param pred: (batchsize, feature_dimension)
        :param gt: (batchsize, feature_dimension)
        :return: MSE scalar
        '''
        self.pred = pred
        self.gt = gt
        return ((pred-gt)**2).mean()

    def backward(self):
        '''
        MSE backward
        :return: gradient w.r.t prediction (batchsize, feature_dimension)
                 2*(pred-gt)/(feature_dimension*batchsize)
        '''
        Nb = self.pred.shape[0] # batchsize
        Nf = self.pred.shape[1] # feature size
        self.model.backward((self.pred-self.gt)*2/Nb/Nf)

class SGD():
    '''
    stochastic gradient descent optimizer
    '''
    def __init__(self, lr, model):
        '''
        initialize the optimizer
        :param lr: learning rate
        :param model: the model to be optimized
        '''
        self.lr = lr
        self.model = model

    def step(self):
        '''
        update parameters with corresponding gradients
        '''
        for m in self.model.modules:
            if m.param()!=[]:
                m.weight -= self.lr * m.weight_grad
                m.bias -= self.lr * m.bias_grad