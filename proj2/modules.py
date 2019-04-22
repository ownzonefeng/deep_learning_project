from torch import empty


class Module(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

class ReLU(Module):

    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return input * (input>0)

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
        self.weight = empty((self.input_num, self.output_num))
        self.bias = empty((1, self.output_num))
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
        :param gradwrtoutput: (batchsize, output_num)
        :return: dL/dX: (batchsize, input_num)
        '''
        ones = empty((1, gradwrtoutput.shape[0])).fill_(1)
        self.bias_grad += ones.matmul(gradwrtoutput)
        self.weight_grad += self.input.t().matmul(gradwrtoutput)
        return gradwrtoutput.matmul(self.weight.t())