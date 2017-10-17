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
        Raw = [[] for i in range(self.numberOfSubjects - 1)]
        events = [[] for i in range(self.numberOfSubjects - 1)]

        # create list with raw and events

        j = 0
        for i in np.arange(1, self.numberOfSubjects, 1):
            datafile = './BCICIV_2a_gdf/A0' + str(i) + 'T.gdf'  # directory
            Raw[j], events[j] = BCI.BCICompetition4Set2A(datafile).load()  # load raw data with removed artifacts
            j += 1

        eegEpochsArray = [[] for i in range(self.numberOfSubjects - 1)]
        eegData = np.zeros((self.numberOfSubjects, 288, 26, 176))
        # gets a 4D array with dimensions [Subject, Event, EEGChannel, Time]

        for i in range(self.numberOfSubjects - 1):
            eegEpochsArray[i] = mne.Epochs(Raw[i], events[i])
            eegEpochsArray[i].drop_bad()  # drop bad epochs
            eegData[i, :, :] = eegEpochsArray[i].get_data()

        eegData = eegData[:, :, 0:22, :]  # just 22 electrodes are for recording
        labels = np.array([[item[2] for item in events[i]] for i in range(self.numberOfSubjects - 1)])

        print eegData.shape
        print labels.shape
        loadfinish = time.time()
        print "loading data took " + str(loadfinish - loadinit) + " secs."
        print str(eegData.shape[0]) + " subjects, " + str(eegData.shape[1]) + " trials, and " + str(
            eegData.shape[2]) + " channels found."
        return eegData, labels

    def train(self, eegData, events):

        trainDataSet = eegData[:, ::2, :, :]  # even  - start at the beginning at take every second item
        trainLabels = events[:, ::2]

        inputChannels = trainDataSet.shape[2]  # odd - start at second item and take every second item
        inputTimeLength = trainDataSet.shape[3]
        numberOfClasses = 4

        print "Initializing Convolutional Neural Network"
        ConvNetwork = nn.ConvNet(inputChannels, numberOfClasses, inputTimeLength)

        for s in range(eegData.shape[0]):  # number of subjects
            for e in range(eegData.shape[1]):  # number of events
                ConvNetwork.training(trainDataSet[s, e, :, :], trainLabels[s, e], learningRate=0.05)

    def test(self, eegData, events):
        testDataSet = eegData[:, 1::2, :, :]
        testLabels = events[:, 1::2]
        pass


eegData, labels = EEGDecoding(9).loadData()  # funciona
X = EEGDecoding(9).train(eegData, labels)

"""file = './BCICIV_2a_gdf/A01T.gdf' # directory
A01T_raw, events  = BCI.BCICompetition4Set2A(file).load() #  load raw data with removed artifacts
A01T = mne.Epochs(A01T_raw, events)  # epoching the data
A01T.drop_bad() # drop bad epochs
A01TdataforEachTrial = A01T.get_data() # gets a 3D array containing EEG in each epoch
print A01TdataforEachTrial[1, :, :].shape"""

"""inputChannels = len(A.ch_names) - 4 # number Of Channels except last 4 (3 EOG and last stim)
numberOfInputs = A.shape[0]
inputTimeLength = A.n_times/A.n_events
numberOfClasses = 4
"""
