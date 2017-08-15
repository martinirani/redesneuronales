import numpy as np
import matplotlib.pyplot as plt


class SigmoidNeuron:

    def __init__(self, x, bias, weight):

        self.x = x
        self.bias = bias
        self.weight = weight

    def zvalue(self):
        return np.dot(self.x, self.weight) + self.bias

    def output(self):
        return 1/(1 + np.exp(SigmoidNeuron(self.x, self.bias, self.weight).zvalue()))


class LearningSigmoid(object, SigmoidNeuron):
    def __init__(self, x, y, bias, learningrate, train_num):
        self.x = x
        self.y = y
        self.bias = bias
        self.weight = np.random.rand(1, 1)     # weights are picked randomly to start
        self.learningrate = learningrate
        self.train_num = train_num

    def train(self):

        for i in xrange(0, self.train_num):

            guess = SigmoidNeuron.output(self.x[i])
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
        plt.plot(self.x, SigmoidNeuron(self.x, self.bias, self.weight).output())
        plt.show()

x = np.random.rand(50, 1)
y = np.random.rand(50, 1)
train_num = 50
b = 1
learningrate = 0.01

LearningSigmoid(x, y, b, learningrate, train_num).draw()




