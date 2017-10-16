from unittest import TestCase
import NeuralNetwork as NN
import numpy as np

# Test for a 2-Layer Neural Network: 2 inputs, 3 hidden neurons in hidden layer, 1 output layer

# -------------------- XOR  ------------------------------

Net = NN.NeuralNetwork(2, [3], 1)
Input = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
Expect = np.array([[1], [0], [0], [1]])
learningRate = 0.01


class TestNeuralNetwork(TestCase):
    def testNeuralNetworkParameters(self):
        self.assertEqual(Net.numberOfInputs, 2)
        self.assertEqual(Net.numberOfNeuronsInHiddenLayer[0], 3)
        self.assertEqual(Net.numberOfHiddenLayers, 1)
        self.assertEqual(len(Net.hiddenLayer), 1)
        self.assertEqual(Net.numberOfOutputs, 1)
        self.assertEqual(Net.hiddenLayer[0].__nextLayer, Net.outputLayer)
        self.assertEqual(Net.outputLayer, Net.hiddenLayer[0].nextLayer)

    def testNeuralNetworkFeed(self):
        self.assertEqual(Net.hiddenLayer[0].numberOfInputs, 2)
        self.assertEqual(len(Net.hiddenLayer[0].neuronsInLayer), 3)
        self.assertEqual(len(Net.outputLayer.neuronsInLayer[0].ouput), 1)
        self.assertEqual()

    def testVerifyDeltas(self):
        self.assertEqual()

    def testBiasUpdate(self):
        pass

    def testFinalWeightsAndBias(self):
        pass
