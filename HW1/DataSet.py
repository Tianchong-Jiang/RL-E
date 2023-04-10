import numpy as np


class DataSet(object):
    """A class for reading in training and test data.

    Attributes
    ----------
    filename : str
        The name of the file containing the data
    envShape : list
        A two element list specifying the shape of the environment
    states : list
        A list of state sequences with each state specified by its (x, y)
        coordinates. Each states[i] denotes a distinct state sequence
    observations : list
        A list of observation sequences, where each observation is a string.
        Each observations[i] is a list denoting a distinct observation sequence

    Methods
    -------
    readFile()
        Reads data from filename
    """

    def __init__(self, filename):
        """Initialize the class.

        Parameters:
        ----------
        filename : str
            The name of the file containing the training data
        """
        # The following are some variables that may be necessary or
        # useful. You may find that you need/want to add other variables.
        self.filename = filename
        self.envShape = [4, 4]

        self.states = []  # An array of (x, y) coordinates
        self.observations = []

    def readFile(self):
        """Read in file and populate training state and output sequences."""
        states = []
        obs = []

        with open(self.filename, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                elif line[0] == '.':
                    self.states.append(states)
                    self.observations.append(obs)
                    states = []
                    obs = []
                else:
                    x, y, c = line.strip().split(',')
                    states.append((int(x), int(y)))
                    obs.append(c)
        if states != []:
            self.states.append(states)
            self.observations.append(obs)

    def sub2Ind(self, i, j):
        """Convert subscript (i,j) to linear index."""
        return (self.envShape[1]*i + j)
