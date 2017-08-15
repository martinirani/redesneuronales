import numpy as np
import matplotlib.pyplot as plt


# [CC501] Ex.1. Perceptron
# Tarea Lunes
# Usar Github


class Perceptron:
    def __init__(self, x, weight, bias):
        self.x = x
        self.weight = weight
        self.bias = bias

    def feedforward(self):
        return np.dot(self.x, self.weight) + self.bias

    def output(self):
        if Perceptron.feedforward(self) > 0:
            return 1
        else:
            return 0

class PerceptronNand(object, Perceptron):
    """only admits inputs values 0 and 1"""
    def __init__(self, x):
        self.x = x
        self.bias = len(x) * 1
        self.weight = len(x) * [-1]


class PerceptronAnd(object, Perceptron):
    """only admits inputs values 0 and 1"""
    def __init__(self, x):
        self.x = x
        self.bias = len(x) * -1 + 1
        self.weight = len(x) * [1]


class PerceptronOr(object, Perceptron):
    """only admits inputs values 0 and 1"""
    def __init__(self, x):
        self.x = x
        self.bias = 0
        self.weight = len(x) * [1]


class LearningPerceptron(object, Perceptron):
    def __init__(self, x, y, bias, learningrate, train_num):
        self.x = x
        self.y = y
        self.bias = bias
        self.weight = np.random.rand(1, 1)     # weights are picked randomly to start

        self.learningrate = learningrate
        self.train_num = train_num

    def train(self):

        for i in xrange(0, self.train_num):

            guess = Perceptron.feedforward(self.x[i])
            desired = self.y[i]
            error = desired - guess

            if error > 0:
                self.weight = self.weight + self.learningrate * self.x[i]
            elif error < 0:
                self.weight = self.weight - self.learningrate * self.x[i]
            else:
                self.weight = self.weight

    def draw(self):
        plt.scatter(self.x, self.y)
        plt.plot(self.x, Perceptron(self.x, self.weight, self.bias).feedforward())
        plt.show()

x = np.random.rand(50, 1)
y = np.random.rand(50, 1)
train_num = 50
b = 1
learningrate = 0.01

LearningPerceptron(x, y, b, learningrate, train_num).draw()



