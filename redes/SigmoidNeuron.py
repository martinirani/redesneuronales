import numpy as np
import matplotlib.pyplot as plt


class SigmoidNeuron:
    def Z(self, X, w, b):
        return np.dot(X, w) + b

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_with_zeros(self, dim):
        w = np.zeros([dim])
        b = 0
        return w, b

    def propagation(self, w, b, X, Y):
        m = X.shape[1]
        A = self.sigmoid(self.Z(X, w, b))
        cost = (-1 / m) * np.sum([(Y * np.log(A)) + (1 - Y) * np.log(1 - A)])
        dw = (1 / m) * np.dot(X, [A - Y])
        db = (1 / m) * np.sum([A - Y])
        return dw, db, cost

    def optimization(self, w, b, X, Y, num_iterations, learning_rate):


("def __init__(self, x, bias, weight, threshold):\n"
 "\n"
 "        self.x = x\n"
 "        self.bias = bias\n"
 "        self.weight = weight\n"
 "        self.threshold = threshold\n"
 "\n"
 "\n"
 "\n"
 "\n"
 "class LearningSigmoid(object, SigmoidNeuron):\n"
 "    def __init__(self, x, y, bias, learningrate, train_num, threshold):\n"
 "        self.x = x\n"
 "        self.y = y\n"
 "        self.bias = bias\n"
 "        self.weight = np.random.rand(2, 1)  # weights are picked randomly to start\n"
 "        self.learningrate = learningrate\n"
 "        self.train_num = train_num\n"
 "        self.xy = np.concatenate((x, y), axis=1)\n"
 "        self.guess = np.zeros(np.shape(y))\n"
 "        self.threshold = threshold\n"
 "\n"
 "    def train(self):\n"
 "\n"
 "        for ii in xrange(0, self.train_num):\n"
 "            for i in xrange(0, np.shape(self.x)[0]):\n"
 "                self.guess[ii] = SigmoidNeuron(self.xy[ii, 0:2], self.weight, self.bias, self.threshold).sigma(self)\n"
 "\n"
 "                self.guess[i] = SigmoidNeuron(self.xy[i, 0:2], self.weight, self.bias, self.threshold).output()\n"
 "\n"
 "                desired = self.xy[i, 2:3]\n"
 "                sign = desired-self.guess[i]\n"
 "                self.weight = self.weight + sign*np.transpose([self.xy[i, 0:2]])*self.learningrate\n"
 "\n"
 "                #print self.weight\n"
 "\n"
 "    def color(self, c):\n"
 "        if c ==0:\n"
 "            return 'red'\n"
 "        else:\n"
 "            return 'blue'\n"
 "\n"
 "    def draw(self):\n"
 "        for i in range(0,n_points):\n"
 "            plt.scatter(self.x[i,0:1], self.x[i,1:2], c='blue', marker='o')\n"
 "        for i in range(n_points, 2*n_points):\n"
 "            plt.scatter(self.x[i, 0:1], self.x[i,1:2], c='red', marker='o')\n"
 "\n"
 "        plt.plot(self.x[:,0:1], self.x[:,0:1]*self.weight[0] + self.bias)\n"
 "        plt.show()\n"
 "\n"
 "\n"
 "x_population1 = np.random.normal(np.array([5, 5]), 1.0, np.array([n_points, 2]))\n"
 "x_pop1_label = np.zeros((n_points, 1)) + 1\n"
 "\n"
 "x_population2 = np.random.normal(np.array([0, 0]), 1.0, np.array([n_points, 2]))\n"
 "x_pop2_label = np.zeros((n_points, 1))\n"
 "\n"
 "x = np.concatenate((x_population1, x_population2), axis=0)\n"
 "y = np.concatenate((x_pop1_label, x_pop2_label), axis=0)\n"
 "\n"
 "train_num = 20\n"
 "b = 1\n"
 "learningrate = 0.01\n"
 "threshold = 0.5\n"
 "\n"
 "SigNeuron = LearningSigmoid(x, y, b, learningrate, train_num, threshold)\n"
 "SigNeuron.train()\n"
 "SigNeuron.draw()")
