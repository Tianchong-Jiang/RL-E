import sys
import pickle
from HMM import HMM
from DataSet import DataSet


if __name__ == '__main__':
    """Call Viterbi implementation of HMM on a given set of observations."""
    # This function with the test data and (optionally) a model file
    if (len(sys.argv) < 2 or len(sys.argv) > 3):
        print("Usage: TrainMM.py testingdata.txt trained-model.pkl")
        sys.exit(0)

    dataset = DataSet(sys.argv[1])
    dataset.readFile()

    T = None
    M = None
    pi = None
    if (len(sys.argv) == 3):
        f = open(sys.argv[2], "rb")
        model = pickle.load(f)
        T = model['T']
        M = model['M']
        pi = model['pi']

    hmm = HMM(dataset.envShape, T, M, pi)

    totalCorrect = 0
    totalIncorrect = 0
    for i in range(len(dataset.observations)):
        predictedStates = hmm.viterbi(dataset.observations[i])
        if len(predictedStates) != len(dataset.states[i]):
            print("Length of predictedStates differs from dataset.states")
            sys.exit(-1)
        trueStates = dataset.states[i]

        numCorrect = 0
        for j in range(len(dataset.states[i])):
            if predictedStates[j] == dataset.states[i][j]:
                numCorrect += 1

        totalCorrect += numCorrect
        totalIncorrect += (len(dataset.observations[i]) - numCorrect)

    accuracy = totalCorrect/(totalCorrect + totalIncorrect)
    print("Accuracy: {0:.2f} percent".format(100*accuracy))
