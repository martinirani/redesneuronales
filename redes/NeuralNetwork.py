import numpy as np


class NeuralNetwork:
    def __init__(self, numberOfInputs, numberOfNeuronsInHiddenLayer, numberOfOutputs):
        self.numberOfInputs = numberOfInputs
        self.numberOfNeuronsInHiddenLayer = numberOfNeuronsInHiddenLayer
        self.numberOfHiddenLayers = len(self.numberOfNeuronsInHiddenLayer)
        self.numberOfOutputs = numberOfOutputs
        self.hiddenLayer = [None] * len(self.numberOfNeuronsInHiddenLayer)

    def createNetwork(self):

        if self.numberOfHiddenLayers is 1:

            self.hiddenLayer[0] = NeuronLayer(self.numberOfInputs, self.numberOfNeuronsInHiddenLayer[0])
            self.outputLayer = NeuronLayer(self.numberOfNeuronsInHiddenLayer[0], self.numberOfOutputs)
            self.hiddenLayer[0].nextLayer(self.outputLayer)
            self.outputLayer.nextLayer(None)
            self.outputLayer.previousLayer(self.hiddenLayer[0])
            self.hiddenLayer[0].previousLayer(None)

        else:
            for L in range(self.numberOfHiddenLayers):
                if L is 0:
                    self.hiddenLayer[L] = NeuronLayer(self.numberOfInputs, self.numberOfNeuronsInHiddenLayer[L])
                elif 0 < L <= self.numberOfHiddenLayers-1:
                    self.hiddenLayer[L] = NeuronLayer(self.numberOfNeuronsInHiddenLayer[L-1], self.numberOfNeuronsInHiddenLayer[L])
            self.outputLayer = NeuronLayer(self.numberOfNeuronsInHiddenLayer[self.numberOfHiddenLayers-1], self.numberOfOutputs)

            for L in range(self.numberOfHiddenLayers):
                if L is 0:
                    self.hiddenLayer[L].nextLayer(self.hiddenLayer[L + 1])
                    self.hiddenLayer[L].previousLayer(None)
                elif 0 < L < self.numberOfHiddenLayers - 1:
                    self.hiddenLayer[L].nextLayer(self.hiddenLayer[L + 1])
                    self.hiddenLayer[L].previousLayer(self.hiddenLayer[L - 1])
                elif L is self.numberOfHiddenLayers - 1:
                    self.hiddenLayer[L].previousLayer(self.hiddenLayer[L - 1])
                    self.hiddenLayer[L].nextLayer(self.outputLayer)

            self.outputLayer.nextLayer(None)
            self.outputLayer.previousLayer(self.hiddenLayer[self.numberOfHiddenLayers-1])

    def feed(self, someInputValues):
        outputValues = self.hiddenLayer[0].feedForward(someInputValues)
        return outputValues

    def train(self, someInputValues, expectedValues, learningRate, epochs):
        self.createNetwork()
        for i in range(epochs):
            for j in range(np.shape(someInputValues)[0]):
                self.feed(someInputValues[j])
                self.outputLayer.backPropagationOutputLayer(expectedValues[j])
                self.hiddenLayer[self.numberOfHiddenLayers-1].backPropagationHiddenLayer()
                self.hiddenLayer[0].updateWeights(someInputValues[j], learningRate)
                self.hiddenLayer[0].updateBias(learningRate)
                self.outputLayer.updateWeights(someInputValues[j], learningRate)
                self.outputLayer.updateBias(learningRate)
                self.hiddenLayer[0].resetOutputs()

    def performance(self, outputValues, expectedValues):

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(expectedValues)):
            if outputValues[i] == expectedValues[i] == 1:
                TP += 1
        for i in range(len(expectedValues)):
            if expectedValues[i] == 1 and outputValues[i] != expectedValues[i]:
                FP += 1
        for i in range(len(expectedValues)):
            if outputValues[i] == expectedValues[i] == 0:
                TN += 1
        for i in range(len(expectedValues)):
            if expectedValues[i] == 0 and outputValues != expectedValues[i]:
                FN += 1

        TPRate = TP/(TP+FN)
        FPRate = FP/(FP+TN)
        Precision = (FP+TN)/(len(expectedValues))

        return TP, FP, TN, FN, TPRate, FPRate, Precision

class NeuronLayer:
    def __init__(self, numberOfInputs, numberOfNeuronsInLayer):
        self.numberOfInputs = numberOfInputs
        self.numberOfNeuronsInLayer = numberOfNeuronsInLayer
        self.neuronsInLayer = [Neuron(self.numberOfInputs) for i in range(self.numberOfNeuronsInLayer)]
        self.someOutputs = []

    def feedForward(self, someInputValues):  # Feed the neuron layer with some inputs
        for i in range(self.numberOfNeuronsInLayer):  # loop for feeding every neuron in the layer
            self.someOutputs.append(self.neuronsInLayer[i].output(someInputValues))
        if self.__nextLayer is None:  # if there is not next layer
            return self.someOutputs
        else:  # if there is a layer, pass it to the next layer
            return self.__nextLayer.feedForward(self.someOutputs)

    def nextLayer(self, aLayer):
        self.__nextLayer = aLayer

    def previousLayer(self, aLayer):
        self.__previousLayer = aLayer

    def backPropagationOutputLayer(self, expectedValue):
        theError = np.subtract(expectedValue, self.someOutputs)
        for i in range(self.numberOfNeuronsInLayer):
            self.neuronsInLayer[i].adjustDeltaWith(self.someOutputs[i], theError[0])

    def backPropagationHiddenLayer(self):
        theError = np.dot(self.__nextLayer.get_deltas(), self.__nextLayer.get_weights())
        if self.__previousLayer is None:
            for i in range(self.numberOfNeuronsInLayer):
                self.neuronsInLayer[i].adjustDeltaWith(self.someOutputs[0], theError[0])
        else:
            for i in range(self.numberOfNeuronsInLayer):
                self.neuronsInLayer[i].adjustDeltaWith(self.someOutputs[0], theError[0])
            self.__previousLayer.backPropagationHiddenLayer()

    def get_deltas(self):
        deltas = np.array([self.neuronsInLayer[i].get_delta_value() for i in range(self.numberOfNeuronsInLayer)])
        return deltas

    def get_weights(self):
        weights = np.array([self.neuronsInLayer[i].get_weight_value() for i in range(self.numberOfNeuronsInLayer)])
        return weights

    def updateWeights(self, inputs, learningRate):
        if self.__previousLayer is None:
            for i in range(self.numberOfNeuronsInLayer):
                self.neuronsInLayer[i].adjustWeightWithInput(inputs, learningRate)
        else:
            for i in range(self.numberOfNeuronsInLayer):
                self.neuronsInLayer[i].adjustWeightWithInput(self.__previousLayer.someOutputs, learningRate)

    def updateBias(self, learningRate):
        for i in range(self.numberOfNeuronsInLayer):
            self.neuronsInLayer[i].adjustBiasUsingLearningRate(learningRate)

    def resetOutputs(self):
        if self.__nextLayer is None:
            self.someOutputs = []
        else:
            self.someOutputs = []
            self.__nextLayer.resetOutputs()


class Neuron:
    def __init__(self, numberOfInputs):
        self.weightValues = np.random.randn(numberOfInputs)
        self.bias = np.random.randn(1)

    def output(self, inputValues):
        Z = np.dot(inputValues, self.weightValues) + self.bias
        self.outputValue = 1 / (1 + np.exp(-Z))
        return self.outputValue[0]

    def get_output_value(self):
        outputValue = self.outputValue
        return outputValue

    def adjustBiasUsingLearningRate(self, learningRate):
        self.bias = self.bias + (learningRate * self.delta)
        return self.bias

    def adjustDeltaWith(self, output, anError):
        self.delta = anError * self.transferDerivative(output)
        return self.delta

    def adjustWeightWithInput(self, inputs, learningRate):
        for i in range(len(inputs)):
            self.weightValues[i] = self.weightValues[i] + (learningRate * self.delta * inputs[i])

    def get_delta_value(self):
        return self.delta

    def get_weight_value(self):
        return self.weightValues

    def transferDerivative(self, output):
        return output * (1 - output)




# XOR
"""Net = NeuralNetwork(2, [3], 1)
Input = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
Expect = np.array([[1], [0], [0], [1]])
learningRate = 0.01
epochs = 100

Net.train(Input, Expect, learningRate, epochs)
A = Net.feed(np.array([1, 1]))"""

