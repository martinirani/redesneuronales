import numpy as np
import matplotlib.pyplot as plt


# [CCn_points1] Ex.1. Perceptron
# Tarea Lunes
# Usar Github


class Perceptron:
    def __init__(self, x, weight, bias):
        self.x = x
        self.weight = weight
        self.bias = bias

    def feedforward(self):
        value = np.dot(self.x, self.weight) + self.bias
        return value

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


n_points = 50

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
        self.weight = np.random.rand(2, 1)     # weights are picked randomly to start
        self.learningrate = learningrate
        self.train_num = train_num
        self.xy = np.concatenate((x, y), axis=1)
        self.guess = np.zeros(np.shape(y))

    def train(self):

        for ii in xrange(0, self.train_num):
            for i in xrange(0, np.shape(self.x)[0]):

                self.guess[i] = Perceptron(self.xy[i, 0:2], self.weight, self.bias).output()
                #print self.guess[i]

                desired = self.xy[i, 2:3]
                sign = desired-self.guess[i]
                self.weight = self.weight + sign*np.transpose([self.xy[i, 0:2]])*self.learningrate

                #print self.weight

    def color(self,c):
        if c ==0:
            return 'red'
        else:
            return 'blue'

    def draw(self):
        for i in range(0,n_points):
            plt.scatter(self.x[i,0:1], self.x[i,1:2], c='blue', marker='o')
        for i in range(n_points,2*n_points):
            plt.scatter(self.x[i,0:1], self.x[i,1:2], c='red', marker='o')

        plt.plot(self.x[:,0:1], self.x[:,0:1]*self.weight[0] + self.bias)
        plt.show()

x_population1 = np.random.normal(np.array([5, 5]), 2.0, np.array([n_points, 2]))
x_pop1_label = np.zeros((n_points, 1)) + 1

x_population2 = np.random.normal(np.array([0, 0]), 2.0, np.array([n_points, 2]))
x_pop2_label = np.zeros((n_points, 1))

x = np.concatenate((x_population1, x_population2), axis=0)
y = np.concatenate((x_pop1_label, x_pop2_label), axis=0)

train_num = 200
b = 1
learningrate = 0.01


percyval = LearningPerceptron(x, y, b, learningrate, train_num)
percyval.train()
percyval.draw()


