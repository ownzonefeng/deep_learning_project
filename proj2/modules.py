from torch import empty
import math

class Module(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        pass

    def __call__(self, *input):
        if len(input)==1: # for normal modules
            return self.forward(input[0])
        elif len(input)==2: # for MSE loss. the 1st input is prediction and the second one is ground truth
            return self.forward(input[0], input[1])


class ReLU(Module):

    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return input * (input>0).float()

    def backward(self, gradwrtoutput):
        grad = empty(*gradwrtoutput.shape)
        grad[self.input>0] = 1
        grad[self.input<=0] = 0
        return grad * gradwrtoutput


class tanh(Module):

    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return input.tanh()

    def backward(self, gradwrtoutput):
        grad = 1 - (self.input.tanh())**2
        return grad * gradwrtoutput

class Linear(Module):

    def __init__(self, input_num, output_num):
        self.input = None
        self.input_num = input_num
        self.output_num = output_num
        # TODO: INITIALIZATION
        self.weight = empty((self.input_num, self.output_num)).uniform_(-1/math.sqrt(self.input_num), 1/math.sqrt(self.input_num))#.normal_()/5
        self.bias = empty((1, self.output_num)).uniform_(-1/math.sqrt(self.input_num), 1/math.sqrt(self.input_num))#.normal_()/5

        self.weight_grad = 0
        self.bias_grad = 0

    def forward(self, input):
        '''

        :param input: (batchsize, input_num)
        :return: (batchsize, output_num)
        '''

        self.input = input
        return input.matmul(self.weight) + self.bias

    def backward(self, gradwrtoutput):
        '''
        :param gradwrtoutput dL/dy: (batchsize, output_num)
        :return: dL/dX: (batchsize, input_num)
        '''
        ones = empty((1, gradwrtoutput.shape[0])).fill_(1)
        self.bias_grad += ones.matmul(gradwrtoutput)  # dL/db = 1^T * dL/dy
        self.weight_grad += self.input.t().matmul(gradwrtoutput)  # dL/dW = X^T * dL/dy
        return gradwrtoutput.matmul(self.weight.t())  # dL/dX = dL/dy * W^T

    def param(self):
        return [[self.weight, self.weight_grad], [self.bias, self.bias_grad]]

    def zero_grad(self):
        self.weight_grad = 0
        self.bias_grad = 0

class Sequential(Module):

    def __init__(self, *modules):
        self.modules = modules
        self.input = None

    def forward(self, input):
        self.input = input
        y = input
        for m in self.modules:
            y = m(y)
        return y

    def backward(self, gradwrtoutput):
        reversed_modules = self.modules[::-1]
        grad = gradwrtoutput
        for m in reversed_modules:
            grad = m.backward(grad)
        return grad

    def param(self):
        params = []
        for m in self.modules:
            params.append(m.param())
        return params

    def zero_grad(self):
        for m in self.modules:
            m.zero_grad()

class LossMSE(Module):

    def __init__(self,model):
        self.pred = None
        self.gt = None
        self.model = model

    def forward(self, pred, gt):
        self.pred = pred
        self.gt = gt
        return ((pred-gt)**2).mean()

    def backward(self):
        Nb = self.pred.shape[0] # batchsize
        Nf = self.pred.shape[1] # feature size
        self.model.backward((self.pred-self.gt)*2/Nb/Nf)

class SGD():
    def __init__(self, lr, model):
        self.lr = lr
        self.model = model

    def step(self):
        for m in self.model.modules:
            if m.param()!=[]:
                m.weight -= self.lr * m.weight_grad
                m.bias -= self.lr * m.bias_grad