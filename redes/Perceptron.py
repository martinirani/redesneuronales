import numpy


# [CC501] Ex.1. Perceptron
# Tarea Lunes
# Usar Github


class Perceptron:
    def __init__(self, x, weight, bias):
        self.x = x
        self.weight = weight
        self.bias = bias

    def output(self):
        if (numpy.dot(self.x, self.weight) + self.bias) > 0:
            return 1
        else:
            return 0


class PerceptronNand(Perceptron):
    def __init__(self, x):
        self.x = x
        self.bias = len(x) * 1
        self.weight = len(x) * [-1]


class PerceptronAnd(Perceptron):
    def __init__(self, x):
        self.x = x
        self.bias = len(x) * -1 + 1
        self.weight = len(x) * [1]


class PerceptronOr(object, Perceptron):
    def __init__(self, x):
        self.x = x
        self.bias = 0
        self.weight = len(x) * [1]
