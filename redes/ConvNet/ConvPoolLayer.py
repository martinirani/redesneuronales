import numpy as np
import time
import matplotlib.pyplot as plt
import math

class ConvNet:

    def __init__(self,
                 inputChannels,  # input Channels from the EEG
                 numberOfClasses,  # number of Classes
                 inputTimeLength,  # EEG total length
                 numberOfTimeFilters=25,  # number of Filters used in the time convolution
                 numberOfSpaceFilters=25,  # number of Filters used in the spatial convolution
                 filterTimeLength=5,  # length of the Filter used in the time convolution
                 poolTimeLength=3,  # pool length used in the first pool layer
                 poolTimeStride=3,  # stride used in the first pool layer
                 numberOfFilters2=50,  # number of filters used in the second convolution
                 filterLength2=5,  # length of filter used in the second convolution
                 numberOfFilters3=100,  # number of filters used in the third convolution
                 filterLength3=5,  # length of filters used in the third convolution
                 numberOfFilters4=200,  # number of filters used in the forth convolution
                 filterLength4=3,
                 convStride=1,
                 numberOfFCLayers=2,
                 numberOfNeuronsInLayer=200,
                 dropoutProbability=0.5  # probability used for drop out
                 ):

        self.__dict__.update(locals())
        del self.self

        # 1. First Conv-Pool Block

        # Time Convolution
        print "initializing convolution layer over time"
        timeInputShape = (inputChannels, 1, 1, inputTimeLength)
        timeKernelSize = (1, filterTimeLength)
        self.timeConvLayer = Conv2DLayer(timeInputShape, timeKernelSize, numberOfTimeFilters, stride=(1, 1),
                                         zeroPadding=0, activationFunction='relu', alpha=1)

        # Spatial Convolution
        # print "initializing convolution layer over space"
        spaceInputShape = (self.timeConvLayer.outputValues.shape[3], numberOfTimeFilters, inputChannels, 1)
        spaceKernelSize = (inputChannels, 1)
        self.spaceConvLayer = Conv2DLayer(spaceInputShape, spaceKernelSize, numberOfSpaceFilters,
                                          stride=(convStride, 1),
                                          zeroPadding=0, activationFunction='relu', alpha=1, spaceConv=True)

        # First pool Layer
        # print "intializing first pool layer"

        poolInputShape_1 = (1, numberOfSpaceFilters, 1, self.timeConvLayer.outputValues.shape[3])
        poolKernelSize_1 = (1, poolTimeLength)
        self.poolLayer_1 = PoolLayer(poolInputShape_1, poolKernelSize_1, stride=(1, poolTimeStride))

        # 2. Second Conv-Pool Block

        # print "initializing second convolutional layer"

        convInputShape_2 = (1, numberOfSpaceFilters,
                            self.poolLayer_1.outputValues.shape[2], self.poolLayer_1.outputValues.shape[3])
        kernelSizeConv_2 = (1, filterLength2)
        self.convLayer_2 = Conv2DLayer(convInputShape_2, kernelSizeConv_2, numberOfFilters2, stride=(1, convStride),
                                       zeroPadding=0, activationFunction='relu', alpha=1)

        # print "intializing second pool layer"
        poolInputShape_2 = (
        1, numberOfFilters2, self.convLayer_2.outputValues.shape[2], self.convLayer_2.outputValues.shape[3])
        poolKernelSize_2 = (1, poolTimeLength)
        self.poolLayer_2 = PoolLayer(poolInputShape_2, poolKernelSize_2, stride=(1, poolTimeStride))

        # 3. Third Conv-Pool Block
        # print "initializing third convolutional Layer"
        convInputShape_3 = (
        1, numberOfFilters2, self.poolLayer_2.outputValues.shape[2], self.poolLayer_2.outputValues.shape[3])
        kernelSizeConv_3 = (1, filterLength3)
        self.convLayer_3 = Conv2DLayer(convInputShape_3, kernelSizeConv_3, numberOfFilters3, stride=(1, convStride),
                                       zeroPadding=0, activationFunction='relu', alpha=1)

        # print "intializing third pool layer"
        poolInputShape_3 = (
        1, numberOfFilters3, self.convLayer_3.outputValues.shape[2], self.convLayer_3.outputValues.shape[3])
        poolKernelSize_3 = (1, poolTimeLength)
        self.poolLayer_3 = PoolLayer(poolInputShape_3, poolKernelSize_3, stride=(1, poolTimeStride))


        # 4. Fourth Conv Block
        # print "intializing fourth convolutional Layer"
        convInputShape_4 = (
            1, numberOfFilters3, self.poolLayer_3.outputValues.shape[2], self.poolLayer_3.outputValues.shape[3])
        kernelSizeConv_4 = (1, filterLength4)
        self.convLayer_4 = Conv2DLayer(convInputShape_4, kernelSizeConv_4, numberOfFilters4, stride=(1, convStride),
                                       zeroPadding=0, activationFunction='relu', alpha=1)

        # 5. Classification Layer
        self.numberOfFCLayers = numberOfFCLayers
        self.aFCLayer = [[] for i in range(numberOfFCLayers)]

        for layerIndex in range(numberOfFCLayers):

            if layerIndex is 0:

                inputShape_i = np.squeeze(self.convLayer_4.outputValues)
                numberOfInputs = inputShape_i.size

                # print "initializing first full connected layer"
                self.aFCLayer[layerIndex] = FCLayer(numberOfInputs,
                                                    numberOfNeuronsInLayer,
                                                    layerIndex,
                                                    dropoutProbability,
                                                    activationFunction='softmax',
                                                    firstLayer=True)

            elif 0 < layerIndex < (numberOfFCLayers - 1):
                # print "initializing second full connected layer"

                self.aFCLayer[layerIndex] = FCLayer(numberOfNeuronsInLayer,
                                                    numberOfNeuronsInLayer,
                                                    layerIndex,
                                                    dropoutProbability,
                                                    activationFunction='softmax')

            else:
                # print "initializing ouput Layer"
                self.aFCLayer[layerIndex] = FCLayer(numberOfNeuronsInLayer,
                                                    numberOfClasses,
                                                    layerIndex,
                                                    dropoutProbability,
                                                    activationFunction='softmax',
                                                    outputLayer=True)

        # Connecting Layers

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
        self.convLayer_4.nextLayer(self.aFCLayer[0])

        for layerIndex in range(numberOfFCLayers):

            if layerIndex is 0:
                self.aFCLayer[layerIndex].previousLayer(self.convLayer_4)
                self.aFCLayer[layerIndex].nextLayer(self.aFCLayer[layerIndex + 1])
            elif 0 < layerIndex < numberOfFCLayers - 1:
                self.aFCLayer[layerIndex].previousLayer(self.aFCLayer[layerIndex - 1])
                self.aFCLayer[layerIndex].nextLayer(self.aFCLayer[layerIndex + 1])
            else:
                self.aFCLayer[layerIndex].previousLayer(self.aFCLayer[layerIndex - 1])
                self.aFCLayer[layerIndex].nextLayer(None)

    def forward(self, someInputValues):
        # print "We are now in the forward step"
        someInputValues -= np.mean(someInputValues)
        someInputValues /= np.std(someInputValues)
        self.timeConvLayer.forward(someInputValues)

    def backward(self, expectedValues):
        # print "We are now in the backward step"
        return self.aFCLayer[self.numberOfFCLayers - 1].backpropagation(expectedValues)

    def updateParameters(self, learningRate):
        return self.timeConvLayer.updateParams(learningRate)

    def training(self, someInputValues, expectedValues, learningRate):
        # print "Training phase started"
        self.forward(someInputValues)
        self.backward(expectedValues)
        self.updateParameters(learningRate)

    def test(self, testInputValues, labels):
        self.forward(testInputValues)
        outputs = self.aFCLayer[self.numberOfFCLayers - 1].errorlist
        convArray = self.getConfMat(outputs, labels)
        self.confusalMatrix(convArray)

    def getConfMat(self, outputs, labels):
        convArray = np.zeros(4, 4)

        for x in range(len(labels)):
            if math.isnan(outputs[x]):
                pass
            else:
                convArray[outputs[x], labels[x]] += 1

        return convArray

    def confusalMatrix(self, ConvArray):

        norm_conf = []
        figcount = 0
        for i in ConvArray:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j) / float(a))
            norm_conf.append(tmp_arr)

        fig = plt.figure()
        figcount += 1
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                        interpolation='nearest')

        width, height = ConvArray.shape

        for x in list(range(width)):
            for y in list(range(height)):
                ax.annotate(str(ConvArray[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

        cb = fig.colorbar(res)
        alphabet = '123456789LMNOPQRSTUVWXYZ'
        plt.xticks(range(width), alphabet[:width])
        plt.yticks(range(height), alphabet[:height])
        plt.suptitle('Matriz de Confusion')

    def learningCurve(self):
        plt.plot(self.aFCLayer[self.numberOfFCLayers - 1].errorlist)
        plt.show()  # plots learning curve


class Conv2DLayer:

    def __init__(self, inputShape, kernelSize, numberOfFilters, stride, zeroPadding, activationFunction, alpha=1,
                 spaceConv=False):

        """
        Intializes layer parameters

        Arguments:

        :param inputShape: Input dimensions [numberOfInput, numberOfInputChannels, Heigth, Width]
        :param kernelSize: Kernel dimensions [Width, Heigth]
        :param numberOfFilters: number of Kernels used in the Layer
        :param stride: Stride of the convolution
        :param activationFunction: activation Function of the layer
        :param alpha: alpha value for elu activation Function
        :param zeroPadding: Zero padding added to both sides of the input
        """
        self.__dict__.update(locals())
        del self.self

        self.weights = np.random.rand(numberOfFilters, self.kernelSize[0],
                                      self.kernelSize[1])  # initializes random values for the kernels
        self.bias = np.random.rand(numberOfFilters)  # initializes random values for the biases

        # Computing dimensions of output

        if self.inputShape[2] == self.kernelSize:
            outputHeight = self.inputShape[2]
            outputWidth = (self.inputShape[3] - (self.kernelSize[1]) / self.stride[1] + 1)

            self.outputValues = np.zeros((self.inputShape[0], self.numberOfFilters, outputHeight, outputWidth))
        else:
            outputHeight = (self.inputShape[2] + 2 * self.zeroPadding - (self.kernelSize[0]) / self.stride[0] + 1)
            outputWidth = (self.inputShape[3] + 2 * self.zeroPadding - (self.kernelSize[1]) / self.stride[1] + 1)

            self.outputValues = np.zeros((self.inputShape[0], self.numberOfFilters, outputHeight, outputWidth))
            self.outputShape = self.outputValues.shape

    def forward(self, someInputs):

        """
        Applies a 2D convolution over an input signal composed of several input planes.

        Arguments:
        :param someInputs: array of dimensions [numberOfInputs, inputChannels, Height, Weight]
        :return OutputValues: array of dimensions [numberOfOutputs, outputChannels, Height, Weight]
        """

        if self.spaceConv is True:
            someInputs = self.SpaceConvMatrixTranspose(someInputs)
            if self.outputValues.shape == self.outputShape:
                pass
            else:
                self.outputValues = np.transpose(self.outputValues, (3, 1, 2, 0))
        else:
            someInputs = np.reshape(someInputs, (self.inputShape))

        assert someInputs.shape == self.inputShape

        #  Adds Zero Padding
        if self.zeroPadding is 0:  # no padding added
            self.inputs = someInputs

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
        print "Performing convolution"
        timeA = time.time()
        for n in range(self.inputShape[0]):
            for cout in range(self.numberOfFilters):
                for cin in range(self.inputShape[1]):
                    nh = 0
                    for h in np.arange(0, self.inputShape[2] - self.kernelSize[0] + 1, self.stride[0]):
                        nw = 0
                        for w in np.arange(0, self.inputShape[3] - self.kernelSize[1] + 1, self.stride[1]):
                            activationMap = self.inputs[n, cin, h:h + self.kernelSize[0],
                                            w:w + self.kernelSize[1]]  # Portion of the input feature map convolved
                            kernel = self.weights[cout, :, :]  # kernel used for the convolution
                            self.outputValues[n, cout, nh, nw] = np.sum(activationMap * kernel) + self.bias[
                                cout]  # convolution
                            nw += 1
                        nh += 1

        timeB = time.time()

        if self.spaceConv is True:
            self.outputValues = np.transpose(self.outputValues, (3, 1, 2, 0))

        # print "Convolution took " + str(timeB - timeA) + " seconds"

        # Applies the activation function to the resultant matrix
        if self.activationFunction is 'relu':
            self.outcome = self.relu(self.outputValues)
            # Applies reLU function
            if self.__nextLayer is None:
                return self.outcome
            else:
                return self.__nextLayer.forward(self.outcome)  # Applies eLU function

        elif self.activationFunction is 'elu':
            self.outcome = self.elu(self.outputValues, self.alpha)
            if self.__nextLayer is None:
                return self.outcome
            else:
                return self.__nextLayer.forward(self.outcome)

        elif self.activationFunction is 'sigmoid':  # Applies sigmoid function

            self.outcome = self.sigmoid(self.outputValues)
            if self.__nextLayer is None:
                return self.outcome
            else:
                return self.__nextLayer.forward(self.outcome)

    def backpropagation(self):

        """
        backward pass of Conv layer.

        :param deltasNext: derivatives from next layer of shape (N, K, HF, WF)

        :return self.deltaWeights
        :return self.deltaBiases
         """

        print "backpropagation in Convlayer"

        if self.__nextLayer.__class__.__name__ is 'FCLayer':
            WF = self.__nextLayer.numberOfNeuronsInLayer
            dNext = np.reshape(self.__nextLayer.getDeltas(), (1, 1, 1, WF))
        else:
            dNext = self.__nextLayer.getDeltas()

        self.deltas = np.zeros(self.outputValues.shape)

        # Compute Deltas
        if self.__nextLayer.__class__.__name__ is 'FCLayer':
            for n in range(self.outputValues.shape[0]):
                for nf in range(self.numberOfFilters):
                    for h in range(self.outputValues.shape[2]):
                        for w in range(self.outputValues.shape[3]):
                            deltas_i = self.activationFunctionDerivative(self.outputValues)[n, nf, h, w] * dNext[
                                                                                                           :, :, :, nf]
                            self.deltas[n, nf, h, w] += deltas_i

        elif self.__previousLayer is None:
            for n in range(self.outputValues.shape[0]):
                deltas_i = self.activationFunctionDerivative(self.outputValues)[n] * dNext
                self.deltas[n] += deltas_i[0]

        else:
            for n in range(self.outputValues.shape[0]):
                deltas_i = self.activationFunctionDerivative(self.outputValues)[n] * dNext
                self.deltas[n] += deltas_i[n]

        # print "shape of delta is " + str(self.deltas.shape)

        if self.spaceConv is True:
            self.deltas = np.transpose(self.deltas, (3, 1, 2, 0))
        else:
            pass

        # Compute delta Biases
        deltaBiases = (np.sum(self.deltas, axis=(0, 2, 3)))
        assert deltaBiases.shape == self.bias.shape

        # Compute delta Kernels

        deltaKernel = np.zeros(self.weights.shape)

        for ninp in range(self.inputShape[0]):
            for nf in range(self.numberOfFilters):
                flippedDelta = self.flipArray(self.deltas[ninp, nf, :, :])  # Flips Kernel for the convolution
                for cin in range(self.inputShape[1]):
                    nh = 0
                    for h in np.arange(0, self.inputs.shape[2] - flippedDelta.shape[0] + 1, self.stride[0]):
                        nw = 0
                        for w in np.arange(0, self.inputs.shape[3] - flippedDelta.shape[1] + 1, self.stride[1]):
                            activationMap = self.inputs[ninp, cin,
                                            h:h + flippedDelta.shape[0],
                                            w:w + flippedDelta.shape[1]]  # Input Map used for the convolution
                            deltaKernel[nf, nh, nw] = np.sum(activationMap * flippedDelta)  # Convolution
                            nw += 1
                        nh += 1

        if self.spaceConv is True:
            self.deltas = np.transpose(self.deltas, (3, 1, 2, 0))
        else:
            pass

        self.deltaWeights = deltaKernel
        self.deltaBiases = deltaBiases

        if self.__previousLayer is None:
            return self.deltas, self.deltaWeights, self.deltaBiases
        else:
            return self.__previousLayer.backpropagation()


    def updateParams(self, learningRate):
        """
        :param learningRate: value of learning Rate

        :return self.weights: updated weights
        :return self.bias: updated biases
        """
        # print "we are now updating parameters"
        self.weights += learningRate * (self.deltaWeights * self.weights)
        self.bias += learningRate * self.deltaBiases

        if self.__nextLayer is None:
            return self.weights, self.bias
        else:
            self.__nextLayer.updateParams(learningRate)

    def elu(self, outputValues, alpha):
        self.outputs = np.maximum(outputValues, 0) + (alpha * (np.exp(np.minimum(outputValues, 0)) - 1))
        return self.outputs

    def relu(self, outputValues):
        self.outputs = np.maximum(outputValues, 0)
        return self.outputs

    def sigmoid(self, outputValues):
        self.outputs = 1 / (1 + np.exp(-outputValues))
        return self.outputs

    def activationFunctionDerivative(self, outputValues):
        if self.activationFunction is 'sigmoid':
            self.transferDerivative = outputValues * (1 - outputValues)
            return self.transferDerivative
        if self.activationFunction is 'relu':
            self.transferDerivative = 1. * (outputValues > 0)
            return self.transferDerivative
        if self.activationFunction is 'elu':
            self.transferDerivative = np.multiply(np.maximum(outputValues, 0),
                                                  self.elu(outputValues, self.alpha)) + self.alpha
            return self.transferDerivative

    def nextLayer(self, aLayer):
        self.__nextLayer = aLayer

    def previousLayer(self, aLayer):
        self.__previousLayer = aLayer

    def flipArray(self, anArray):
        return np.fliplr(np.flipud(anArray))

    def getDeltas(self):
        return self.deltas

    def SpaceConvMatrixTranspose(self, someInputValues):

        transposedValues = np.transpose(someInputValues, (3, 1, 0, 2))
        return transposedValues

class PoolLayer:

    def __init__(self, inputShape, kernelSize, stride):

        """
        Initializes layer parameters

        Arguments:
            :param inputShape: dimensions of input [numberOfInputs, numberOfInputChannels, Height, Width]
            :param kernelSize: np.array of dimensions [kernelHeight, kernelWidth]
            :param stride: int or tuple
        """
        self.__dict__.update(locals())
        del self.self

        # compute dimensions of outputValues

        if self.kernelSize[0] == self.inputShape[2]:
            outputHeight = self.inputShape[2]
            outputWidth = ((self.inputShape[3] - self.kernelSize[1]) / self.stride[1] + 1)
            self.outputValues = np.zeros((self.inputShape[0], self.inputShape[1], outputHeight, outputWidth))
        else:
            outputHeight = ((self.inputShape[2] - self.kernelSize[0]) / self.stride[0] + 1)
            outputWidth = ((self.inputShape[3] - self.kernelSize[1]) / self.stride[1] + 1)

            self.outputValues = np.zeros((self.inputShape[0], self.inputShape[1], outputHeight, outputWidth))

    def forward(self, someInputs):

        """
        :param someInputs: input Values of dimensions [numberOfInputs, Height, Width]

        :return self.outputValues: outputValues after pooling
        """

        print "Forward in pooling layer"

        self.someInputs = someInputs
        # print "The shape of the Input is " + str(self.someInputs.shape)
        # print "and the ouput " + str(self.outputValues.shape)

        someInputs = np.reshape(someInputs, (self.inputShape))

        assert someInputs.shape == self.inputShape

        #  list with the posititon of the
        self.maxIdx = []

        # Do the pooling
        for n in range(self.inputShape[0]):
            for c in range(self.inputShape[1]):
                nh = 0
                for h in np.arange(0, self.inputShape[2] - self.kernelSize[0] + 1, self.stride[0]):
                    nw = 0
                    for w in np.arange(0, self.inputShape[3] - self.kernelSize[1] + 1, self.stride[1]):
                        activationMap = self.someInputs[n, c, h:h + self.kernelSize[0], w:w + self.kernelSize[1]]
                        self.outputValues[n, c, nh, nw] += activationMap.argmax()
                        self.maxIdx.append(np.unravel_index(activationMap.argmax(), activationMap.shape))
                        nw += 1
                    nh += 1

        if self.__nextLayer is None:
            return self.outputValues
        else:
            return self.__nextLayer.forward(self.outputValues)

    def backpropagation(self):

        """
        Computes the backward pass of MaxPool Layer.
        Input:
        delta: delta values of shape (N, D, H/factor, W/factor)
        """

        print "Backpropagation in pool layer"
        deltasNext = self.__nextLayer.getDeltas()
        self.deltas = np.zeros(self.inputShape)


        # for para dar los valores del delta siguiente a los maximos
        idx = 0
        for n in range(self.inputShape[0]):
            for c in range(self.inputShape[1]):
                nh = 0
                for h in range(self.inputShape[2], self.inputShape[2] - self.kernelSize[0] + 1, self.stride[0]):
                    nw = 0
                    for w in range(self.inputShape[3], self.inputShape[3] - self.kernelSize[1] + 1, self.stride[1]):
                        self.deltas[n, c, w + self.maxIdx[idx][0], h + self.maxIdx[idx][1]] = deltasNext[
                                                                                              n, c,
                                                                                              nh: nh + self.kernelSize[
                                                                                                  0],
                                                                                              nw:nw + self.kernelSize[
                                                                                                  1]]
                        idx += 1


        if self.__previousLayer is None:
            return self.deltas
        else:
            return self.__previousLayer.backpropagation()

    def updateParams(self, learningRate):
        return self.__nextLayer.updateParams(learningRate)

    def nextLayer(self, aLayer):
        self.__nextLayer = aLayer

    def previousLayer(self, aLayer):
        self.__previousLayer = aLayer

    def getDeltas(self):
        return self.deltas


class FCLayer:
    def __init__(self, numberOfInputs, numberOfNeuronsInLayer, layerIndex, dropoutProbability, activationFunction,
                 alpha=1,
                 trainMode=False, outputLayer=False, firstLayer=False):
        """

        :param numberOfInputs:
        :param numberOfNeuronsInLayer:
        :param layerIndex:
        :param dropoutProbability:
        :param activationFunction:
        :param alpha:
        :param trainMode:
        :param outputLayer:
        :param firstLayer:
        """

        self.__dict__.update(locals())
        del self.self

        self.weights = np.random.rand(numberOfNeuronsInLayer, numberOfInputs)
        self.biases = np.random.rand(numberOfNeuronsInLayer)
        self.deltaWeights = np.zeros(self.weights.shape)
        self.deltaBias = np.zeros(self.biases.shape)
        self.dropoutVector = np.random.binomial(1, self.dropoutProbability,
                                                size=numberOfNeuronsInLayer) / self.dropoutProbability
        self.errorlist = []

    def forward(self, someInputs):
        """

        :param someInputs:
        :return:
        """
        print "Forward in a FC layer"
        # print "The number of the Input is " + str(self.numberOfInputs)
        someInputs = someInputs.reshape(self.numberOfInputs)
        z = np.dot(someInputs, self.weights.T) + self.biases

        if self.trainMode is False:

            if self.activationFunction is 'elu':
                self.outputValues = self.elu(z, self.alpha)
            elif self.activationFunction is 'relu':
                self.outputValues = self.relu(z)
            elif self.activationFunction is 'sigmoid':
                self.outputValues = self.sigmoid(z)
            elif self.activationFunction is 'softmax':
                self.outputValues = self.softmax(z)


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
            elif self.activationFunction is 'softmax':
                self.outputValues = self.softmax(z)

            if self.outputLayer is True:
                return self.outputValues
            else:
                return self.__nextLayer.forward(self.outputValues)

    def backpropagation(self, expectedValues):
        """

        :param expectedValues:
        :return:
        """

        print "Backpropagation in fc layer"

        if self.trainMode is False:

            if self.outputLayer is True:
                Error = self.error(self.outputValues, expectedValues)
                self.deltas = np.multiply(Error, self.activationFunctionDerivative(self.outputValues))
                return self.__previousLayer.backpropagation(None)

            elif self.firstLayer is False and self.outputLayer is False:
                weightsNextLayer = self.__nextLayer.getWeights()
                deltasNextLayer = self.__nextLayer.getDeltas()
                self.deltas = np.dot(weightsNextLayer, deltasNextLayer)
                return self.__previousLayer.backpropagation(None)
            else:
                if self.__previousLayer is None:
                    weightsNextLayer = self.__nextLayer.getWeights()
                    deltasNextLayer = self.__nextLayer.getDeltas()
                    self.deltas = np.dot(weightsNextLayer, deltasNextLayer)
                    return self.deltas
                else:
                    weightsNextLayer = self.__nextLayer.getWeights()
                    deltasNextLayer = self.__nextLayer.getDeltas()
                    self.deltas = np.dot(weightsNextLayer.T, deltasNextLayer)
                    return self.__previousLayer.backpropagation()

        elif self.trainMode is True:
            if self.outputLayer is True:
                Error = np.subtract(self.outputValues, expectedValues)
                self.deltas = Error * self.activationFunctionDerivative(self.outputValues)
                self.deltas *= self.dropoutVector
                return self.__previousLayer.backPropagation(None)

            elif self.firstLayer is False and self.outputLayer is False:
                weightsNextLayer = self.__nextLayer.getWeights()
                deltasNextLayer = self.__nextLayer.getDeltas()
                self.deltas = np.dot(weightsNextLayer, deltasNextLayer)
                self.deltas *= self.dropoutVector
                return self.__previousLayer.backpropagation(None)
            else:

                if self.__previousLayer is None:
                    weightsNextLayer = self.__nextLayer.getWeights()
                    deltasNextLayer = self.__nextLayer.getDeltas()
                    self.deltas = np.dot(weightsNextLayer, deltasNextLayer)
                    self.deltas *= self.dropoutVector
                    return self.deltas
                else:
                    weightsNextLayer = self.__nextLayer.getWeights()
                    deltasNextLayer = self.__nextLayer.getDeltas()
                    self.deltas = np.dot(weightsNextLayer.T, deltasNextLayer)
                    return self.__previousLayer.backpropagation()

    def updateParams(self, learningRate):
        """

        :param someInputs:
        :param learningRate:
        :return:
        """

        if self.firstLayer is True:

            someInputs = self.__previousLayer.outcome.reshape(self.numberOfInputs)
            self.deltaBiases = learningRate * self.deltas

            for nn in range(self.numberOfNeuronsInLayer):
                self.deltaWeights[nn, :] = learningRate * self.deltas[nn] * someInputs

            self.weights += self.deltaWeights
            self.biases += self.deltaBiases

            self.__nextLayer.updateParams(learningRate)

        elif self.firstLayer is not True and self.outputLayer is not True:

            self.deltaBiases = learningRate * self.deltas
            for nn in range(self.numberOfNeuronsInLayer):
                self.deltaWeights = learningRate * self.deltas[nn] * self.__previousLayer.outputValues,

            self.weights += self.deltaWeights
            self.biases += self.deltaBiases

            self.__nextLayer.updateParams(self.outputValues, learningRate)

        elif self.outputLayer is True:

            self.deltaBiases = learningRate * self.deltas

            for nn in range(self.numberOfNeuronsInLayer):
                self.deltaWeights = learningRate * self.deltas[nn] * self.__previousLayer.outputValues

            self.weights += self.deltaWeights
            self.biases += self.deltaBiases

    def activationFunctionDerivative(self, outputValues):
        """

        :param outputValues:
        :return:
        """
        if self.activationFunction is 'sigmoid':
            self.transferDerivative = outputValues * (1 - outputValues)
            return self.transferDerivative
        if self.activationFunction is 'relu':
            self.transferDerivative = 1. * (outputValues > 0)
            return self.transferDerivative
        if self.activationFunction is 'elu':
            self.transferDerivative = (outputValues < 0) * self.elu(outputValues, self.alpha) + self.alpha
            return self.transferderivative
        if self.activationFunction is 'softmax':
            self.transferDerivative = outputValues * (1 - outputValues)
            return self.transferDerivative


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

    def softmax(self, outputValues):
        exps = np.exp(outputValues - np.max(outputValues))
        self.outputs = exps / np.sum(exps)
        return self.outputs

    def getWeights(self):
        return self.weights

    def getDeltas(self):
        return self.deltas

    def error(self, outputValues, expectedValues):
        """

        :param outputValues:
        :param expectedValues:
        :return: Cross Entropy Error
        """
        # Creates an array of 0 with the length of outputValues
        expectedValue = np.zeros(len(outputValues))
        #  Adds a 1 in the position of the expected Value
        expectedValue[expectedValues.astype(int) - 1] += 1
        log = np.log(outputValues)
        #  Compute Error
        error = - (1.0 / len(outputValues)) * np.dot(expectedValue, log)
        self.errorlist.append(error)
        # print self.errorlist
        # print outputValues
        print "the expected value is " + str(expectedValues)
        print "the output value is " + str(np.unravel_index(outputValues.argmax(), outputValues.shape)[0] + 1)
        print "the error is " + str(error)
        return error
