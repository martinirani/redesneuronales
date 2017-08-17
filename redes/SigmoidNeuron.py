import numpy as np
import matplotlib.pyplot as plt

n_points = 100


class SigmoidNeuron:

    def __init__(self, x, bias, weight, threshold):

        self.x = x
        self.bias = bias
        self.weight = weight
        self.threshold = threshold


    def zvalue(self):
        value = np.dot(self.x, self.weight) + self.bias
        print value
        return value

    def sigma(self):
        sigma = 1/(1 + np.exp(-(self.zvalue())))
        return sigma

    def output(self):
        if self.sigma > self.threshold:
            return 1
        else:
            return 0


class LearningSigmoid(object, SigmoidNeuron):
    def __init__(self, x, y, bias, learningrate, train_num, threshold):
        self.x = x
        self.y = y
        self.bias = bias
        self.weight = np.random.rand(2, 1)  # weights are picked randomly to start
        self.learningrate = learningrate
        self.train_num = train_num
        self.xy = np.concatenate((x, y), axis=1)
        self.guess = np.zeros(np.shape(y))
        self.threshold = threshold

    def train(self):

        for ii in xrange(0, self.train_num):
            for i in xrange(0, np.shape(self.x)[0]):

                self.guess[i] = SigmoidNeuron(self.xy[i, 0:2], self.weight, self.bias, self.threshold).output()

                desired = self.xy[i, 2:3]
                sign = desired-self.guess[i]
                self.weight = self.weight + sign*np.transpose([self.xy[i, 0:2]])*self.learningrate

                #print self.weight

    def color(self, c):
        if c ==0:
            return 'red'
        else:
            return 'blue'

    def draw(self):
        for i in range(0,n_points):
            plt.scatter(self.x[i,0:1], self.x[i,1:2], c='blue', marker='o')
        for i in range(n_points, 2*n_points):
            plt.scatter(self.x[i, 0:1], self.x[i,1:2], c='red', marker='o')

        plt.plot(self.x[:,0:1], self.x[:,0:1]*self.weight[0] + self.bias)
        plt.show()

x_population1 = np.random.normal(np.array([20, 20]), 50.0, np.array([n_points, 2]))
x_pop1_label = np.zeros((n_points, 1)) + 1

x_population2 = np.random.normal(np.array([0, 0]), 50.0, np.array([n_points, 2]))
x_pop2_label = np.zeros((n_points, 1))

x = np.concatenate((x_population1, x_population2), axis=0)
y = np.concatenate((x_pop1_label, x_pop2_label), axis=0)

train_num = 500
b = 1
learningrate = 0.001
threshold = 0.5

percyval = LearningSigmoid(x, y, b, learningrate, train_num, threshold)
percyval.train()
percyval.draw()



