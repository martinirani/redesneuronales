from numpy import array, dot

# [CC501] Ex.1. Perceptron
# Tarea Lunes
# Usar Github


class Perceptron:


    def __init__(self, x, bias, weight):
        self.x = x
        self.config(bias,weight)

    def output(self):
        if (dot(self.x, self.weight) + self.bias) > 0:
            return 1
        else:
            return 0

    def set_bias(self, bias):
        self.bias = bias

    def set_weight(self, weight):
        self.weight = weight

class PerceptronNand(Perceptron):
    def __init__(self, x):
        self.x = x
        self.bias= len(x) * 1
        self.weight= len(x) * [-1]


class PerceptronAnd(Perceptron):
    def __init__(self, x):
        self.x = x
        self.bias= len(x) * -1 + 1
        self.weight = len(x) * [1]


class PerceptronOr(object, Perceptron):
    def __init__(self, x):
        self.x = x
        self.bias = 0
        self.weight = len(x) * [1]

