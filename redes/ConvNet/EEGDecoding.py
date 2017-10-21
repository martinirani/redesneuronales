import ConvPoolLayer as nn
import BCICompetition4 as BCI
import numpy as np
import mne
import time
import matplotlib


class EEGDecoding:

    def __init__(self, numberOfSubjects):
        """

        :param numberOfSubjects:
        """
        self.numberOfSubjects = numberOfSubjects

    def loadData(self):
        """

        :return EEGdata: a 4D array containing the whole data
        :type EEGdata: np.array of dimensions [Subject, Event, EEGChannel, time]
        """

        loadinit = time.time()
        Raw = [[] for i in range(self.numberOfSubjects)]  # list of number of subjects with raw data
        events = [[] for i in range(self.numberOfSubjects)]  # list of events arrays with size of of number of subjects
        eegEpochsArray = [[] for i in range(self.numberOfSubjects)]
        eegData = np.zeros((self.numberOfSubjects, 288, 26, 176))
        eventsArray = np.zeros((self.numberOfSubjects, 288))

        # create list with raw and events

        n = 1
        for j in range(self.numberOfSubjects):

            datafile = './BCICIV_2a_gdf/A0' + str(n) + 'T.gdf'  # directory
            Raw[j], events[j] = BCI.BCICompetition4Set2A(datafile).load()  # load raw data with removed artifacts
            if j == 0:
                eventsArray[j, :] = np.squeeze(np.asanyarray(events[j], list))[:, 2]
            elif j > 0:
                eventsArray[j, :] = np.squeeze(np.asanyarray(events[j], list))[:, 2]
            n += 1

        # gets a 4D array with dimensions [Subject, Event, EEGChannel, Time]

        for i in range(self.numberOfSubjects):
            eegEpochsArray[i] = mne.Epochs(Raw[i], events[i])
            eegEpochsArray[i].drop_bad()  # drop bad epochs
            eegData[i, :, :, :] = eegEpochsArray[i].get_data()

        eegData = eegData[:, :, 0:22, :]  # just 22 electrodes are for recording
        labels = eventsArray

        print "EEG shape is " + str(eegData.shape)
        print "Labels shape is " + str(labels.shape)
        loadfinish = time.time()
        print "loading data took " + str(loadfinish - loadinit) + " secs."
        print str(eegData.shape[0]) + " subjects, " + str(eegData.shape[1]) + " trials, and " + str(
            eegData.shape[2]) + " channels found."
        return eegData, labels

    def train(self, eegData, labels):

        trainDataSet = eegData[:, ::2, :, :]  # even  - start at the beginning at take every second item
        trainLabels = labels[:, ::2]

        inputChannels = trainDataSet.shape[2]  # odd - start at second item and take every second item
        inputTimeLength = trainDataSet.shape[3]
        numberOfClasses = 4

        print "Initializing Convolutional Neural Network"

        ConvNetwork = nn.ConvNet(inputChannels, numberOfClasses, inputTimeLength)

        for s in range(eegData.shape[0]):  # number of subjects
            for e in range(eegData.shape[1]):  # number of events
                print trainLabels[s, e]
                ConvNetwork.training(trainDataSet[s, e, :, :], trainLabels[s, e], learningRate=0.05)

    def test(self, eegData):

        testDataSet = eegData[:, 1::2, :, :]
        testLabels = labels[:, 1::2]

        inputChannels = testDataSet.shape[2]  # odd - start at second item and take every second item
        inputTimeLength = testDataSet.shape[3]
        numberOfClasses = 4

        ConvNetwork = nn.ConvNet(inputChannels, numberOfClasses, inputTimeLength)

        for s in range(eegData.shape[0]):  # number of subjects
            for e in range(eegData.shape[1]):  # number of events
                print testDataSet[s, e, :, :].shape
                ConvNetwork.forward(testDataSet[s, e, :, :])
        pass

eegData, labels = EEGDecoding(1).loadData()
X = EEGDecoding(1).train(eegData, labels)
