import sys
import pickle
from HMM import HMM
from DataSet import DataSet

if __name__ == '__main__':
    """Read in data, call code to train HMM, and save model."""

    # This function should be called with one argument: trainingdata.txt
    if (len(sys.argv) != 2):
        print("Usage: TrainMM.py trainingdata.txt")
        sys.exit(0)

    dataset = DataSet(sys.argv[1])
    dataset.readFile()

    hmm = HMM(dataset.envShape)
    hmm.train(dataset.observations) #, dataset.states)
    # hmm.viterbi(dataset.observations, dataset.states)

    # Save the model for future use
    fileName = "trained-model.pkl"
    print("Saving trained model as " + fileName)
    pickle.dump({'T': hmm.T, 'M': hmm.M, 'pi': hmm.pi}, open(fileName, "wb"))
