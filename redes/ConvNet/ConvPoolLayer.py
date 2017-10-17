import numpy as np
import time

class ConvNet:
    def __init__(self,
                 inputChannels,  # input Channels from the EEG
                 numberOfClasses,  # number of Classes
                 inputTimeLength,  # EEG total length
                 numberOfTimeFilters=25,  # number of Filters used in the time convolution
                 numberOfSpaceFilters=25,  # number of Filters used in the spatial convolution
                 filterTimeLength=10,  # length of the Filter used in the time convolution
                 poolTimeLength=3,  # pool length used in the first pool layer
                 poolTimeLengthStride=3,  # stride used in the first pool layer
                 numberOfFilters2=50,  # number of filters used in the second convolution
                 filterLength2=10,  # length of filter used in the second convolution
                 numberOfFilters3=100,  # number of filters used in the third convolution
                 filterLength3=10,  # length of filters used in the third convolution
                 numberOfFilters4=200,  # number of filters used in the forth convolution
                 convStride=1,
                 numberOfFCLayers=2,
                 numberOfNeuronsInLayer=200,
                 dropProbability=0.5  # probability used for drop out
                 ):

        # 1. First Conv-Pool Block

        # Time Convolution
        print "initializing time conv layer"
        timeInputShape = (inputChannels, 1, 1, inputTimeLength)
        timeKernelSize = (1, filterTimeLength)
        self.timeConvLayer = Conv2DLayer(timeInputShape, timeKernelSize, numberOfTimeFilters, stride=(1, 1),
                                         zeroPadding=0, activationFunction='elu', alpha=1)

        # Spatial Convolution
        print "initializing space conv layer"
        spaceInputShape = (1, numberOfTimeFilters, inputChannels, self.timeConvLayer.outputValues.shape[3])
        spaceKernelSize = (1, inputChannels)
        self.spaceConvLayer = Conv2DLayer(spaceInputShape, spaceKernelSize, numberOfSpaceFilters,
                                          stride=(convStride, 1),
                                          zeroPadding=0, activationFunction='elu', alpha=1)

        # First pool Layer

        poolInputShape_1 = (1, numberOfSpaceFilters, 1, self.timeConvLayer.outputValues.shape[3])
        poolKernelSize_1 = (1, poolTimeLength)
        self.poolLayer_1 = PoolLayer(poolInputShape_1, poolKernelSize_1, stride=(1, poolTimeLengthStride))

        # 2. Second Conv-Pool Block
        print "initializing third conv layer"

        convInputShape_2 = (
        1, numberOfSpaceFilters, self.poolLayer_1.outputValues.shape[2], self.poolLayer_1.outputValues.shape[3])
        kernelSizeConv_2 = (1, filterTimeLength)
        self.convLayer_2 = Conv2DLayer(convInputShape_2, kernelSizeConv_2, numberOfFilters2, stride=(convStride, 1),
                                       zeroPadding=0, activationFunction='elu', alpha=1)

        poolInputShape_2 = (
        1, numberOfFilters2, self.convLayer_2.outputValues.shape[2], self.convLayer_2.outputValues.shape[3])
        poolKernelSize_2 = (1, poolTimeLength)
        self.poolLayer_2 = PoolLayer(poolInputShape_2, poolKernelSize_2, stride=(1, poolTimeLengthStride))

        # 3. Third Conv-Pool Block
        print "initializing fourth conv layer"
        convInputShape_3 = (
        1, numberOfFilters2, self.poolLayer_2.outputValues.shape[2], self.poolLayer_2.outputValues.shape[3])
        kernelSizeConv_3 = (1, filterLength2)
        self.convLayer_3 = Conv2DLayer(convInputShape_3, kernelSizeConv_3, numberOfFilters3, stride=(convStride, 1),
                                       zeroPadding=0, activationFunction='elu', alpha=1)

        poolInputShape_3 = (
        1, numberOfFilters3, self.convLayer_3.outputValues.shape[2], self.convLayer_3.outputValues.shape[3])
        poolKernelSize_3 = (1, poolTimeLength)
        self.poolLayer_3 = PoolLayer(poolInputShape_3, poolKernelSize_3, stride=(1, poolTimeLengthStride))


        # 4. Fourth Conv-Pool Block
        print "fifth conv layer"
        convInputShape_4 = (
            1, numberOfFilters3, self.poolLayer_3.outputValues.shape[2], self.poolLayer_3.outputValues.shape[3])
        kernelSizeConv_4 = (1, filterLength3)
        self.convLayer_4 = Conv2DLayer(convInputShape_4, kernelSizeConv_4, numberOfFilters4, stride=(convStride, 1),
                                       zeroPadding=0, activationFunction='elu', alpha=1)

        poolInputShape_4 = (
        1, numberOfFilters4, self.convLayer_4.outputValues.shape[2], self.convLayer_4.outputValues.shape[3])
        poolKernelSize_4 = (1, poolTimeLength)
        self.poolLayer_4 = PoolLayer(poolInputShape_4, poolKernelSize_4, stride=(1, poolTimeLengthStride))

        # 5. Classification Layer
        self.numberOfFCLayers = numberOfFCLayers
        self.aFCLayer = np.zeros(numberOfFCLayers)

        for layerIndex in range(numberOfFCLayers):
            inputShape = self.aFCLayer[layerIndex].__previousLayer.outputValues.shape
            while layerIndex is 0:
                self.aFCLayer[layerIndex] = FCLayer(inputShape,
                                                    numberOfNeuronsInLayer,
                                                    layerIndex,
                                                    activationFunction='relu',
                                                    firstLayer=True, dropoutProbability=dropProbability)

                self.aFCLayer[layerIndex].__previousLayer(self.poolLayer_4)
                self.aFCLayer[layerIndex].__nextLayer(layerIndex + 1)
            while 0 < layerIndex < numberOfFCLayers - 1:
                self.aFCLayer[layerIndex] = FCLayer(inputShape,
                                                    numberOfNeuronsInLayer,
                                                    layerIndex,
                                                    activationFunction='relu', dropoutProbability=dropProbability)

                self.aFCLayer[layerIndex].__previousLayer(layerIndex - 1)
                self.aFCLayer[layerIndex].__nextLayer(layerIndex + 1)
            while layerIndex is numberOfFCLayers:
                self.aFCLayer[layerIndex] = FCLayer(inputShape,
                                                    numberOfClasses,
                                                    layerIndex,
                                                    activationFunction='relu',
                                                    outputLayer=True, dropoutProbability=dropProbability)
                self.aFCLayer[layerIndex].__previousLayer(layerIndex - 1)

        # Connect Layers

        self.timeConvLayer.nextLayer(self.spaceConvLayer)
        self.timeConvLayer.previousLayer(None)
        self.spaceConvLayer.previousLayer(self.timeConvLayer)
        self.spaceConvLayer.nextLayer(self.poolLayer_1)
        self.poolLayer_1.previousLayer(self.spaceConvLayer)
        self.poolLayer_1.nextLayer(self.convLayer_2)
        self.convLayer_2.previousLayer(self.poolLayer_1)
        self.convLayer_2.nextLayer(self.poolLayer_2)
        self.poolLayer_2.previousLayer(self.convLayer_2)
        self.poolLayer_2.nextLayer(self.convLayer_3)
        self.convLayer_3.previousLayer(self.poolLayer_2)
        self.convLayer_3.nextLayer(self.poolLayer_3)
        self.poolLayer_3.previousLayer(self.convLayer_3)
        self.poolLayer_3.nextLayer(self.convLayer_4)
        self.convLayer_4.previousLayer(self.poolLayer_3)
        self.convLayer_4.nextLayer(self.poolLayer_4)
        self.poolLayer_4.previousLayer(self.convLayer_4)
        self.poolLayer_4.nextLayer(self.aFCLayer[0])

    def forward(self, someInputValues):
        return self.timeConvLayer.conv2D(someInputValues)

    def backward(self, expectedValues):
        return self.aFCLayer[self.numberOfFCLayers].backpropagation(expectedValues)

    def updateParameters(self, learningRate):
        return self.timeConvLayer.updateParams(learningRate)

    def performance(self):
        pass

    def training(self, someInputValues, expectedValues, learningRate):
        timeA = time.time()
        self.forward(someInputValues)
        self.backward(expectedValues)
        self.updateParameters(learningRate)
        timeB = time.time()
        Error = np.subtract(expectedValues, self.forward(someInputValues))
        print 'the training took ' + str(timeB - timeA) + ' seconds'

    def test(self):
        pass

    def confusalMatrix(self):
        pass


class Conv2DLayer:
    def __init__(self, inputShape, kernelSize, numberOfFilters, stride, zeroPadding, activationFunction, alpha=1):

        """
        Intializes layer parameters

        Arguments:

        :param inputShape: Input dimensions [numberOfInput, numberOfInputChannels, Heigth, Width]
        :type inputShape: np.array
        :param kernelSize: Kernel dimensions [Width, Heigth]
        :type kernelSize: np.array
        :param numberOfFilters: number of Kernels used in the Layer
        :type numberOfFilters: int
        :param stride: Stride of the convolution
        :type stride: np.array of dimensions [Heigth stride, Width stride]
        :param activationFunction: activation Function of the layer
        :type activationFunction: string - 'sigmoid', 'elu', 'relu'
        :param alpha: alpha value for elu activation Function
        :type alpha: int value
        :param zeroPadding: Zero padding added to both sides of the input
        :type zeroPadding: int
        """

        self.kernelSize = kernelSize
        self.inputShape = inputShape
        self.numberOfFilters = numberOfFilters
        self.weights = np.random.randn(numberOfFilters, self.inputShape[1], self.kernelSize[0],
                                       self.kernelSize[1])  # initializes random values for the kernels
        self.bias = np.random.randn(numberOfFilters)  # initializes random values for the biases
        self.stride = stride
        self.zeroPadding = zeroPadding
        self.activationFunction = activationFunction
        self.alpha = alpha

        # Computing dimensions of output
        print self.inputShape[2]
        print self.kernelSize[0]
        print self.stride[0]
        print self.inputShape[3]
        print self.kernelSize[1]
        print self.stride[1]

        if self.inputShape[2] == self.kernelSize:
            outputHeight = self.inputShape[2]
            outputWidth = (self.inputShape[3] - (self.kernelSize[0]) / self.stride[1] + 1)

            self.outputValues = np.zeros((self.inputShape[0], self.numberOfFilters, outputHeight, outputWidth))
        else:
            outputHeight = (self.inputShape[2] + 2 * self.zeroPadding - (self.kernelSize[0]) / self.stride[0] + 1)
            outputWidth = (self.inputShape[3] + 2 * self.zeroPadding - (self.kernelSize[0]) / self.stride[1] + 1)

            self.outputValues = np.zeros((self.inputShape[0], self.numberOfFilters, outputHeight, outputWidth))

    def conv2D(self, someInputs):

        """
        Applies a 2D convolution over an input signal composed of several input planes.

        Arguments:
        :param someInputs: array of dimensions [numberOfInputs, inputChannels, Height, Weigth]
        :type someInputs: np.array
        :return OutputValues: array of dimensions [numberOfOutputs, outputChannels, Height, Weight]
        :type OutputValues: np.array
        """
        self.someInputs = someInputs

        #  Adds Zero Padding
        if self.zeroPadding is 0:  # no padding added
            self.inputs = someInputs.reshape(self.inputShape[0], self.inputShape[1], self.inputShape[2],
                                             self.inputShape[3])
        elif self.zeroPadding > 0:  # adds padding
            self.inputs = np.zeros((self.inputShape[0], self.inputShape[1], self.inputShape[2] + 2 * self.zeroPadding,
                                    self.inputShape[
                                        3] + 2 * self.zeroPadding))  # creates a zeros vector with the shape of the padded inputs
            for n in range(self.inputShape[0]):  # does the padding along the W dimension
                for cin in range(self.inputShape[1]):
                    for h in range(self.inputShape[2]):
                        self.inputs[n, cin, h, :] = np.lib.pad(self.someInputs[n, cin, h, :],
                                                               (self.zeroPadding, self.zeroPadding),
                                                               'constant', constant_values=(0, 0))
            for n in range(self.inputShape[0]):  # does the padding along the H dimmension
                for cin in range(self.inputShape[1]):
                    for w in range(self.inputShape[3]):
                        self.inputs[n, cin, :, w + self.zeroPadding] = np.lib.pad(self.someInputs[n, cin, :, w],
                                                                                  (self.zeroPadding, self.zeroPadding),
                                                                                  'constant', constant_values=(0, 0))

        # Do the convolution
        for n in range(self.inputShape[0]):
            for cout in range(self.numberOfFilters):
                for cin in range(self.inputShape[1]):
                    for w in np.arange(0, self.inputs.shape[3] - self.kernelSize[1] + 1, self.stride[1]):
                        for h in np.arange(0, self.inputs.shape[2] - self.kernelSize[0] + 1, self.stride[0]):
                            activationMap = self.inputs[n, cin, h:h + self.kernelSize[0],
                                            w:w + self.kernelSize[1]]  # Portion of the input feature map convolved
                            kernel = self.weights[cout, cin, :, :]  # kernel used for the convolution
                            self.outputValues[n, cout, h, w] += np.sum(activationMap * kernel) + self.bias[
                                cout]  # convolution

        # Applies the activation function to the resultant matrix
        if self.activationFunction is 'relu':  # Applies reLU function
            return self.relu(self.outputValues)
        elif self.activationFunction is 'elu':  # Applies eLU function
            return self.elu(self.outputValues, self.alpha)
        elif self.activationFunction is 'sigmoid':  # Applies sigmoid function
            return self.sigmoid(self.outputValues)

    def backPropagationConvLayer(self):

        """
        backward pass of Conv layer.

        :param deltasNext: derivatives from next layer of shape (N, K, WF, HF)
        :type deltasNext: np.array

        :return self.deltaWeights
        :return self.deltaBiases
         """

        # Compute Deltas
        self.deltas = []
        for n in range(self.inputShape[0]):
            for nf in range(self.numberOfFilters):
                deltas_i = self.activationFunctionDerivative(self.inputs[n, nf], self.activationFunction) * \
                           self.__nextLayer.getDeltas()[n][nf]
                self.deltas.append(deltas_i)

        # Compute delta Biases
        deltaBiases = []
        for delta in self.deltas:
            deltaBiases.append(np.sum(delta))

            # Compute delta Kernels
        deltaKernel = np.zeros(self.weights)
        for n in range(self.inputShape[0]):
            for nf in range(self.numberOfFilters):
                flippedDelta = self.flipArray(self.deltas[n, nf, :, :])  # Flips Kernel for the convolution
                for cin in range(self.inputShape[1]):
                    for w in np.arange(0, self.inputs.shape[3] - self.kernelSize[1] + 1, self.stride[1]):
                        for h in np.arange(0, self.inputs.shape[2] - self.kernelSize[0] + 1, self.stride[0]):
                            activationMap = self.inputs[n, cin, h, w]  # Input Map used for the convolution
                            deltaKernel[n, nf, w, h] = np.sum(activationMap * flippedDelta)  # Convolution

        self.deltaWeights = deltaKernel
        self.deltaBiases = deltaBiases

        return self.deltas, self.deltaWeights, self.deltaBiases

    def updateParams(self, learningRate):
        """
        :param learningRate: value of learning Rate

        :return self.weights: updated weights
        :return self.bias: updated biases
        """
        self.weights -= learningRate * (self.deltaWeights * self.weights)
        self.bias -= learningRate * self.deltaBiases

    def elu(self, outputValues, alpha):
        self.outputs = np.maximum(outputValues, 0) + alpha * (np.exp(np.minimum(outputValues, 0)) - 1)
        return self.outputs

    def relu(self, outputValues):
        self.outputs = np.maximum(outputValues, 0)
        return self.outputs

    def sigmoid(self, outputValues):
        self.outputs = 1 / (1 + np.exp(-outputValues))
        return self.outputs

    def activationFunctionDerivative(self, outputValues, activationFunction):
        if activationFunction is 'sigmoid':
            self.transferDerivative = outputValues * (1 - outputValues)
            return self.transferDerivative
        if activationFunction is 'relu':
            self.transferDerivative = 1. * (outputValues > 0)
        if activationFunction is 'elu':
            self.transferDerivative = (outputValues < 0) * self.elu(outputValues, self.alpha) + self.alpha

    def nextLayer(self, aLayer):
        self.__nextLayer = aLayer

    def previousLayer(self, aLayer):
        self.__previousLayer = aLayer

    def flipArray(self, anArray):
        return np.fliplr(np.flipud(anArray))

    def getDeltas(self):
        return self.deltas


class PoolLayer:

    def __init__(self, inputShape, kernelSize, stride):

        """
        Initializes layer parameters

        Arguments:
            :param inputShape: dimensions of input [numberOfInputs, numberOfInputChannels, Height, Width]
            :param kernelSize: np.array of dimensions [kernelHeight, kernelWidth]
            :param stride: int or tuple
        """
        self.inputShape = inputShape
        self.kernelSize = kernelSize
        self.stride = stride

        # compute dimensions of outputValues

        if self.kernelSize[0] == self.inputShape[2]:
            outputHeight = self.inputShape[2]
            outputWidth = ((self.inputShape[3] - self.kernelSize[1]) / self.stride[1] + 1)
            self.outputValues = np.zeros((self.inputShape[0], self.inputShape[1], outputHeight, outputWidth))
        else:
            outputHeight = ((self.inputShape[2] - self.kernelSize[0]) / self.stride[0] + 1)
            outputWidth = ((self.inputShape[3] - self.kernelSize[1]) / self.stride[1] + 1)

            self.outputValues = np.zeros((self.inputShape[0], self.inputShape[1], outputHeight, outputWidth))

    def maxPool2d(self, someInputs):

        """
        :param someInputs: input Values of dimensions [numberOfInputs, Height, Width]

        :return self.outputValues: outputValues after pooling
        """

        self.someInputs = someInputs.reshape(self.inputShape[0], self.inputShape[1], self.inputShape[2],
                                             self.inputShape[3])

        self.maxIdx = []

        # Do the pooling
        for n in range(self.inputShape[0]):
            for c in range(self.inputShape[1]):
                for w in np.arange(0, self.inputShape[3] - self.kernelSize[1] + 1, self.stride[1]):
                    for h in np.arange(0, self.inputShape[2] - self.kernelSize[0] + 1, self.stride[0]):
                        activationMap = self.someInputs[n, c, h:h + self.kernelSize[0], w:w + self.kernelSize[1]]
                        self.outputValues[n, c, w, h] += activationMap.argmax()
                        self.maxIdx.append(np.unravel_index(activationMap.argmax(), activationMap.shape))

        return self.outputValues

    def backPropagationMaxPool(self, deltasNext):
        """
        Computes the backward pass of MaxPool Layer.
        Input:
        delta: delta values of shape (N, D, H/factor, W/factor)
        """

        self.deltas = np.zeros_like(self.someInputs)
        print self.deltas.shape
        # for para dar los valores del delta siguiente a los maximos
        idx = 0
        for n in range(self.inputShape[0]):
            for c in range(self.inputShape[1]):
                for w in range(deltasNext.shape[3]):
                    for h in range(deltasNext.shape[2]):
                        self.deltas[n, c, self.maxIdx[idx][0], self.maxIdx[idx][1]] = self.__nextLayer.getDeltas()[
                            n, c, h, w]
                        idx += 1

        self.deltas = self.deltas
        return self.deltas

    def nextLayer(self, aLayer):
        self.__nextLayer = aLayer

    def previousLayer(self, aLayer):
        self.__previousLayer = aLayer

    def getDeltas(self):
        return self.deltas

class FCLayer:
    def __init__(self, inputShape, numberOfNeuronsInLayer, layerIndex, activationFunction, dropoutProbability, alpha=1,
                 trainMode=False, outputLayer=False, firstLayer=False):

        self.numberOfNeurons = numberOfNeuronsInLayer
        self.activationFunction = activationFunction
        self.weights = np.random.rand((numberOfNeuronsInLayer, inputShape))
        self.biases = np.random.rand(numberOfNeuronsInLayer)
        self.deltaWeights = np.zeros(self.weights.shape)
        self.deltaBias = np.zeros(self.biases.shape)
        self.alpha = alpha
        self.outputLayer = outputLayer
        self.firstLayer = firstLayer
        self.layerIndex = layerIndex
        self.inputShape = inputShape
        self.trainMode = trainMode
        self.dropoutVector = np.random.binomial(1, dropoutProbability, size=numberOfNeuronsInLayer) / dropoutProbability

    def forward(self, someInputs):

        z = np.dot(self.weights, someInputs) + self.biases

        if self.trainMode is False:

            if self.activationFunction is 'elu':
                self.outputValues = self.elu(z, self.alpha)
            elif self.activationFunction is 'relu':
                self.outputValues = self.relu(z)
            elif self.activationFunction is 'sigmoid':
                self.outputValues = self.sigmoid(z)

            if self.outputLayer is True:
                return self.outputValues
            else:
                return self.__nextLayer.forward(self.outputValues)

        elif self.trainMode is True:

            z *= self.dropoutVector

            if self.activationFunction is 'elu':
                self.outputValues = self.elu(z, self.alpha)
            elif self.activationFunction is 'relu':
                self.outputValues = self.relu(z)
            elif self.activationFunction is 'sigmoid':
                self.outputValues = self.sigmoid(z)

            if self.outputLayer is True:
                return self.outputValues
            else:
                return self.__nextLayer.forward(self.outputValues)

    def backPropagation(self, expectedValues):

        if self.trainMode is False:
            if self.outputLayer is True:
                Error = np.subtract(self.outputValues, expectedValues)
                self.deltas = Error * self.activationFunctionDerivative(self.outputValues, self.activationFunction)
                return self.__previousLayer.backPropagation(None)

            elif self.firstLayer is False and self.outputLayer is False:
                weightsNextLayer = self.__nextLayer.getWeights()
                deltasNextLayer = self.__nextLayer.getDeltas()
                self.deltas = np.dot(weightsNextLayer, deltasNextLayer)
                return self.__previousLayer.backPropagation(None)
            else:
                weightsNextLayer = self.__nextLayer.getWeights()
                deltasNextLayer = self.__nextLayer.getDeltas()
                self.deltas = np.dot(weightsNextLayer, deltasNextLayer)
                return self.deltas

        elif self.trainMode is True:
            if self.outputLayer is True:
                Error = np.subtract(self.outputValues, expectedValues)
                self.deltas = Error * self.activationFunctionDerivative(self.outputValues, self.activationFunction)
                self.deltas *= self.dropoutVector
                return self.__previousLayer.backPropagation(None)

            elif self.firstLayer is False and self.outputLayer is False:
                weightsNextLayer = self.__nextLayer.getWeights()
                deltasNextLayer = self.__nextLayer.getDeltas()
                self.deltas = np.dot(weightsNextLayer, deltasNextLayer)
                self.deltas *= self.dropoutVector
                return self.__previousLayer.backPropagation(None)
            else:
                weightsNextLayer = self.__nextLayer.getWeights()
                deltasNextLayer = self.__nextLayer.getDeltas()
                self.deltas = np.dot(weightsNextLayer, deltasNextLayer)
                self.deltas *= self.dropoutVector
                return self.deltas

    def updateParameters(self, someInputs, learningRate):

        if self.firstLayer is True:
            self.deltaBiases = learningRate * self.deltas
            self.deltaWeights = learningRate * self.deltas * someInputs

            self.weights += self.deltaWeights
            self.biases += self.deltaBiases

            self.__nextLayer.updateParameters(self.outputValues, learningRate)

        elif self.firstLayer is not True and self.outputLayer is not True:
            self.deltaBiases = learningRate * self.deltas
            self.deltaWeights = learningRate * self.deltas * someInputs

            self.weights += self.deltaWeights
            self.biases += self.deltaBiases

            self.__nextLayer.updateParameters(self.outputValues, learningRate)

        elif self.outputLayer is True:

            self.deltaBiases = learningRate * self.deltas
            self.deltaWeights = learningRate * self.deltas * someInputs

            self.weights += self.deltaWeights
            self.biases += self.deltaBiases

    def activationFunctionDerivative(self, outputValues, activationFunction):
        if activationFunction is 'sigmoid':
            self.transferDerivative = outputValues * (1 - outputValues)
            return self.transferDerivative
        if activationFunction is 'relu':
            self.transferDerivative = 1. * (outputValues > 0)
        if activationFunction is 'elu':
            self.transferDerivative = (outputValues < 0) * self.elu(outputValues, self.alpha) + self.alpha


    def nextLayer(self, aLayer):
        self.__nextLayer = aLayer

    def previousLayer(self, aLayer):
        self.__previousLayer = aLayer

    def elu(self, outputValues, alpha):
        self.outputs = np.maximum(outputValues, 0) + alpha * (np.exp(np.minimum(outputValues, 0)) - 1)
        return self.outputs

    def relu(self, outputValues):
        self.outputs = np.maximum(outputValues, 0)
        return self.outputs

    def sigmoid(self, outputValues):
        self.outputs = 1 / (1 + np.exp(-outputValues))
        return self.outputs

    def getWeights(self):
        return self.weights

    def getDeltas(self):
        return self.deltas


"""inputShape = [1, 2, 3, 3]
kernelSize = [2, 2]
numberOfFilters = 5
pading = 1
stride = [1, 1]

ConvL = Conv2DLayer(inputShape, kernelSize, numberOfFilters, stride, pading, activationFunction='elu')

input = np.random.rand(1, 2, 3, 3)
X = ConvL.conv2D(input)
PoolL = PoolLayer(X.shape, [2, 2], stride)
print PoolL.maxPool2d(X)"""
