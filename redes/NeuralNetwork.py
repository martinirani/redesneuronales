import numpy as np


class NeuralNetwork:
    def __init__(self, numberOfInputs, numberOfNeuronsInHiddenLayer, numberOfOutputs):
        self.numberOfInputs = numberOfInputs
        self.numberOfNeuronsInHiddenLayer = numberOfNeuronsInHiddenLayer
        self.numberOfOutputs = numberOfOutputs
        self.hiddenLayer = NeuronLayer(numberOfInputs, numberOfNeuronsInHiddenLayer)
        self.outputLayer = NeuronLayer(numberOfNeuronsInHiddenLayer, numberOfOutputs)
        self.hiddenLayer.nextLayer(self.outputLayer)
        self.outputLayer.nextLayer(None)
        self.outputLayer.previousLayer(self.hiddenLayer)
        self.hiddenLayer.previousLayer(None)

    def feed(self, someInputValues):
        outputValues = self.hiddenLayer.feedForward(someInputValues)
        return outputValues

    def train(self, someInputValues, expectedValues, learningRate, epochs):
        for i in range(epochs):
            for j in range(np.shape(someInputValues)[0]):
                self.feed(someInputValues[j])
                self.outputLayer.backPropagationOutputLayer(expectedValues[j])
                self.hiddenLayer.backPropagationHiddenLayer()
                self.hiddenLayer.updateWeights(someInputValues[j], learningRate)
                self.hiddenLayer.updateBias(learningRate)
                self.outputLayer.updateWeights(someInputValues[j], learningRate)
                self.outputLayer.updateBias(learningRate)
                self.hiddenLayer.resetOutputs()



class NeuronLayer:
    def __init__(self, numberOfInputs, numberOfNeuronsInLayer):
        self.numberOfInputs = numberOfInputs
        self.numberOfNeuronsInLayer = numberOfNeuronsInLayer
        self.neuronsInLayer = [Neuron(self.numberOfInputs) for i in range(self.numberOfNeuronsInLayer)]
        self.someOutputs = []

    def feedForward(self, someInputValues):  # Feed the neuron layer with some inputs
        for i in range(self.numberOfNeuronsInLayer):  # loop for feeding every neuron in the layer
            print 'we are in neuron ' + str(i + 1)
            self.someOutputs.append(self.neuronsInLayer[i].output(someInputValues))
        if self.__nextLayer is None:  # if there is not next layer
            print self.someOutputs
            return self.someOutputs
        else:  # if there is a layer, pass it to the next layer
            return self.__nextLayer.feedForward(self.someOutputs)

    def nextLayer(self, aLayer):
        self.__nextLayer = aLayer

    def previousLayer(self, aLayer):
        self.__previousLayer = aLayer

    def backPropagationOutputLayer(self, expectedValue):
        theError = expectedValue - self.someOutputs
        for i in range(self.numberOfNeuronsInLayer):
            print 'the Error at the output layer is ' + str(theError[0])
            print self.someOutputs[0]
            self.neuronsInLayer[i].adjustDeltaWith(self.someOutputs[i], theError[0])

    def backPropagationHiddenLayer(self):
        theError = np.dot(self.__nextLayer.get_deltas(), self.__nextLayer.get_weights())
        for i in range(self.numberOfNeuronsInLayer):
            self.neuronsInLayer[i].adjustDeltaWith(self.someOutputs[0], theError[0])

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
        if self.__nextLayer is None:  # if there is not next layer
            self.someOutputs = []
        else:  # if there is a layer, clean next layer
            self.someOutputs = []
            self.__nextLayer.resetOutputs()


class Neuron:
    def __init__(self, numberOfInputs):
        self.weightValues = np.random.randn(numberOfInputs)
        self.bias = np.random.rand(1)

    def output(self, inputValues):
        print 'input Values : ' + str(inputValues)
        print 'weight Values :' + str(self.weightValues)
        Z = np.dot(inputValues, self.weightValues) + self.bias
        self.outputValue = 1 / (1 + np.exp(-Z))
        print 'output Values' + str(self.outputValue)
        return self.outputValue[0]

    def get_output_value(self):
        outputValue = self.outputValue
        return outputValue

    def adjustBiasUsingLearningRate(self, learningRate):
        self.bias = self.bias + (learningRate * self.delta)
        return self.bias

    def adjustDeltaWith(self, output, anError):
        self.delta = anError * self.transferDerivative(output)
        print 'At this layer, delta is ' + str(self.delta)
        return self.delta

    def adjustWeightWithInput(self, inputs, learningRate):
        print 'we are now adjusting the weights'
        print 'learning rate: ' + str(learningRate)
        print 'delta :' + str(self.delta)
        print 'inputs :' + str(inputs)
        print 'weightValues before :' + str(self.weightValues)
        for i in range(len(inputs)):
            self.weightValues[i] = self.weightValues[i] + (learningRate * self.delta * inputs[i])
        print ' weightValues after :' + str(self.weightValues)

    def get_delta_value(self):
        return self.delta

    def get_weight_value(self):
        return self.weightValues

    def transferDerivative(self, output):
        return output * (1 - output)

        # XOR


Net = NeuralNetwork(2, 3, 1)
Input = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
Expect = np.array([[1], [0], [0], [1]])
learningRate = 0.005
epochs = 50

Net.train(Input, Expect, learningRate, epochs)
print Net.feed(np.array([1, 1]))



#en vez de arreglos usar for  -> tarea 01 (ultima prioridad)
#agregar flexibilidad con las capas -> tarea 01
#agregar threshold  -> tarea 01
#variables temporales
#backprop
