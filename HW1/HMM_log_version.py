import numpy as np
import pdb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random


class HMM(object):
    """A class for implementing HMMs.

    Attributes
    ----------
    envShape : list
        A two element list specifying the shape of the environment
    states : list
        A list of states specified by their (x, y) coordinates
    observations : list
        A list specifying the sequence of observations
    T : numpy.ndarray
        An N x N array encoding the transition probabilities, where
        T[i,j] is the probability of transitioning from state i to state j.
        N is the total number of states (envShape[0]*envShape[1])
    M : numpy.ndarray
        An M x N array encoding the emission probabilities, where
        M[k,i] is the probability of observing k from state i.
    pi : numpy.ndarray
        An N x 1 array encoding the prior probabilities

    Methods
    -------
    train(observations)
        Estimates the HMM parameters using a set of observation sequences
    viterbi(observations)
        Implements the Viterbi algorithm on a given observation sequence
    setParams(T, M, pi)
        Sets the transition (T), emission (M), and prior (pi) distributions
    getParams
        Queries the transition (T), emission (M), and prior (pi) distributions
    sub2ind(i, j)
        Convert integer (i,j) coordinates to linear index.
    """

    def __init__(self, envShape, T=None, M=None, pi=None):
        """Initialize the class.

        Attributes
        ----------
        envShape : list
            A two element list specifying the shape of the environment
        T : numpy.ndarray, optional
            An N x N array encoding the transition probabilities, where
            T[i,j] is the probability of transitioning from state i to state j.
            N is the total number of states (envShape[0]*envShape[1])
        M : numpy.ndarray, optional
            An M x N array encoding the emission probabilities, where
            M[k,i] is the probability of observing k from state i.
        pi : numpy.ndarray, optional
            An N x 1 array encoding the prior probabilities
        """
        self.envShape = envShape
        self.numStates = envShape[0] * envShape[1]

        if T is None:
            # Initial estimate of the transition function
            # where T[sub2ind(i',j'), sub2ind(i,j)] is the likelihood
            # of transitioning from (i,j) --> (i',j')
            self.T = np.zeros((self.numStates, self.numStates), dtype=np.float128)

            # Self-transitions
            for i in range(self.numStates):
                self.T[i, i] = 0.2

            # Black rooms
            self.T[self.sub2ind(0, 0), self.sub2ind(0, 0)] = 1.0
            self.T[self.sub2ind(1, 1), self.sub2ind(1, 1)] = 1.0
            self.T[self.sub2ind(0, 3), self.sub2ind(0, 3)] = 1.0
            self.T[self.sub2ind(3, 2), self.sub2ind(3, 2)] = 1.0

            # (1, 0) -->
            self.T[self.sub2ind(2, 0), self.sub2ind(1, 0)] = 0.8

            # (2, 0) -->
            self.T[self.sub2ind(1, 0), self.sub2ind(2, 0)] = 0.8/3.0
            self.T[self.sub2ind(2, 1), self.sub2ind(2, 0)] = 0.8/3.0
            self.T[self.sub2ind(3, 0), self.sub2ind(2, 0)] = 0.8/3.0

            # (3, 0) -->
            self.T[self.sub2ind(2, 0), self.sub2ind(3, 0)] = 0.8/2.0
            self.T[self.sub2ind(3, 1), self.sub2ind(3, 0)] = 0.8/2.0

            # (0, 1) --> (0, 2)
            self.T[self.sub2ind(0, 2), self.sub2ind(0, 1)] = 0.8

            # (2, 1) -->
            self.T[self.sub2ind(2, 0), self.sub2ind(2, 1)] = 0.8/3.0
            self.T[self.sub2ind(3, 1), self.sub2ind(2, 1)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(2, 1)] = 0.8/3.0

            # (3, 1) -->
            self.T[self.sub2ind(2, 1), self.sub2ind(3, 1)] = 0.8/2.0
            self.T[self.sub2ind(3, 0), self.sub2ind(3, 1)] = 0.8/2.0

            # (0, 2) -->
            self.T[self.sub2ind(0, 1), self.sub2ind(0, 2)] = 0.8/2.0
            self.T[self.sub2ind(1, 2), self.sub2ind(0, 2)] = 0.8/2.0

            # (1, 2) -->
            self.T[self.sub2ind(0, 2), self.sub2ind(1, 2)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(1, 2)] = 0.8/3.0
            self.T[self.sub2ind(1, 3), self.sub2ind(1, 2)] = 0.8/3.0

            # (2, 2) -->
            self.T[self.sub2ind(1, 2), self.sub2ind(2, 2)] = 0.8/3.0
            self.T[self.sub2ind(2, 1), self.sub2ind(2, 2)] = 0.8/3.0
            self.T[self.sub2ind(2, 3), self.sub2ind(2, 2)] = 0.8/3.0

            # (1, 3) -->
            self.T[self.sub2ind(1, 2), self.sub2ind(1, 3)] = 0.8/2.0
            self.T[self.sub2ind(2, 3), self.sub2ind(1, 3)] = 0.8/2.0

            # (2, 3) -->
            self.T[self.sub2ind(1, 3), self.sub2ind(2, 3)] = 0.8/3.0
            self.T[self.sub2ind(3, 3), self.sub2ind(2, 3)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(2, 3)] = 0.8/3.0

            # (3, 3) --> (2, 3)
            self.T[self.sub2ind(2, 3), self.sub2ind(3, 3)] = 0.8
        else:
            self.T = T

        if M is None:
            # Initial estimates of emission likelihoods, where
            # M[k, sub2ind(i,j)]: likelihood of observation k from state (i, j)
            self.M = np.ones((4, 16)) * 0.1

            # Black states
            self.M[:, self.sub2ind(0, 0)] = 0.25
            self.M[:, self.sub2ind(1, 1)] = 0.25
            self.M[:, self.sub2ind(0, 3)] = 0.25
            self.M[:, self.sub2ind(3, 2)] = 0.25

            self.M[self.obs2ind('r'), self.sub2ind(0, 1)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(0, 2)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(1, 0)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(1, 2)] = 0.7
            self.M[self.obs2ind('r'), self.sub2ind(1, 3)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(2, 0)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(2, 1)] = 0.7
            self.M[self.obs2ind('r'), self.sub2ind(2, 2)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(2, 3)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(3, 0)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(3, 1)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(3, 3)] = 0.7
        else:
            self.M = M

        if pi is None:
            # Initialize estimates of prior probabilities where
            # pi[(i, j)] is the likelihood of starting in state (i, j)
            self.pi = np.ones((16, 1))/12
            self.pi[self.sub2ind(0, 0)] = 0.0
            self.pi[self.sub2ind(1, 1)] = 0.0
            self.pi[self.sub2ind(0, 3)] = 0.0
            self.pi[self.sub2ind(3, 2)] = 0.0
        else:
            self.pi = pi

    def overwriteBlack(self):
        self.M[:, self.sub2ind(0, 0)] = 0.25
        self.M[:, self.sub2ind(1, 1)] = 0.25
        self.M[:, self.sub2ind(0, 3)] = 0.25
        self.M[:, self.sub2ind(3, 2)] = 0.25

        self.pi[self.sub2ind(0, 0)] = 0.0
        self.pi[self.sub2ind(1, 1)] = 0.0
        self.pi[self.sub2ind(0, 3)] = 0.0
        self.pi[self.sub2ind(3, 2)] = 0.0

        self.T[self.sub2ind(0, 0), self.sub2ind(0, 0)] = 1.0
        self.T[self.sub2ind(1, 1), self.sub2ind(1, 1)] = 1.0
        self.T[self.sub2ind(0, 3), self.sub2ind(0, 3)] = 1.0
        self.T[self.sub2ind(3, 2), self.sub2ind(3, 2)] = 1.0

        self.T = np.nan_to_num(self.T)
        self.M = np.nan_to_num(self.M)
        self.pi = np.nan_to_num(self.pi)

    def setParams(self, T, M, pi):
        """Set the transition, emission, and prior probabilities."""
        self.T = T
        self.M = M
        self.pi = pi

    def getParams(self):
        """Get the transition, emission, and prior probabilities."""
        return (self.T, self.M, self.pi)

    # Estimate the transition and observation likelihoods and the
    # prior over the initial state based upon training data
    def train(self, observations, states):
        """Estimate HMM parameters from training data via Baum-Welch.

        Parameters
        ----------
        observations : list
            A list specifying a set of observation sequences
            where observations[i] denotes a distinct sequence
        """
        # This function should set self.T, self.M, and self.pi
        num_iters = 10
        observations = self.obs2ind_all(observations)
        states = self.sub2ind_all(states)
        likelihoods = np.zeros(num_iters, dtype=np.float128)

        for iter in range(num_iters):
            np.take(observations,np.random.permutation(observations.shape[0]),axis=0,out=observations)
            likelihoods_one_iter = np.zeros(observations.shape[0], dtype=np.float128)
            for step in range(observations.shape[0]):
                # print(np.max(self.M))
                # step = 2
                # computer greek letters
                alphas, scales = self.computeAlphas(observations[step])
                betas = self.computeBetas(observations[step], alphas)
                gammas = self.computeGammas(alphas, betas)
                xis = self.computeXis(alphas, betas, observations[step])

                # update params
                self.pi = gammas[:, 0, None]

                # pdb.set_trace()

                for i in range(self.numStates):
                    for j in range(self.numStates):
                        self.T[j, i] = sum(xis[j, i, :-1]) / sum(gammas[i, :-1])

                for m in range(self.M.shape[0]):
                    for i in range(self.numStates):
                        self.M[m, i] = np.dot(gammas[i], np.where(observations[step] == m, 1, 0)) / sum(gammas[i])

                if step % 10 == 0:
                    print(f"{iter}-th iter, {step} step finished")
                    print(f"The log likelihood is {np.sum(scales)}")
                self.overwriteBlack()

                likelihoods_one_iter[step] = np.sum(scales)

            likelihoods[iter] = np.mean(likelihoods_one_iter)
            print(f"Mean log likelihood of iter {iter} is {likelihoods[iter]}")

        self.plot_likelihood(likelihoods)

    def viterbi(self, observation):
        """Implement the Viterbi algorithm.

        Parameters
        ----------
        observations : list
            A list specifying the sequence of observations, where each o
            observation is a string (e.g., 'r')

        Returns
        -------
        states : list
            List of predicted sequence of states, each specified as (x, y) pair
        """
        # CODE GOES HERE
        # Return the list of predicted states, each specified as (x, y) pair
        observation = self.obs2ind_all(observation)

        predictions = np.zeros(observation.shape[0], dtype=np.float128)

        # compute delta and pre
        deltas = np.zeros([self.numStates, observation.shape[0]], dtype=np.float128)
        pres = np.zeros([self.numStates, observation.shape[0]], dtype=np.float128)

        deltas[:, 0] = self.M[observation[0], :] * self.pi[:, 0]
        pres[:, 0] = None

        for t in range(1, observation.shape[0]):

            for i in range(self.numStates):
                deltas[i, t] = self.M[observation[t], i]\
                    * np.max(self.T[i, :] * deltas[:, t-1])

                pres[i, t] = np.argmax(self.T[i, :] * deltas[:, t-1])

        # backtrack
        predictions[-1] = int(np.argmax(deltas[:, -1]))
        for t in range(observation.shape[0] - 2, -1, -1):
            predictions[t] = pres[int(predictions[t+1]) ,t+1]

        predictions = self.ind2sub_all(predictions)

        return predictions

    def computeAlphas(self, observation):
        """Get the alphas given an observation series"""
        alphas = np.zeros([self.numStates, observation.shape[0]], dtype=np.float128)
        log_scales = np.zeros(observation.shape[0], dtype=np.float128)

        # base
        alphas[:, 0] = self.logMult(self.M[observation[0], :], self.pi[:,0])

        # step
        for t in range(1, observation.shape[0]):
            for i in range(self.numStates):
                alphas[i, t] = self.M[observation[t], i]\
                * np.dot(self.T[i, :], alphas[:, t - 1])

            scale = np.sum(alphas[:, t])
            if scale > 0:
                alphas[:, t] = self.logDiv(alphas[:, t], scale)

            log_scales[t] = np.log(scale)

        return alphas, log_scales

    def computeBetas(self, observation, alphas):
        """Get the betas given an observation series"""
        betas = np.zeros([self.numStates, observation.shape[0]], dtype=np.float128)

        # base
        betas[:, -1] = 1.0

        # step
        for t in range(observation.shape[0] - 2, -1, -1):
            for i in range(self.numStates):
                for x in range(self.numStates):
                    betas[i, t] += self.M[observation[t+1], x]\
                    * self.T[x, i] * betas[x, t+1]

            scale = np.sum(betas[:, t])
            if scale > 0:
                betas[:, t] = self.logDiv(betas[:, t], scale)

        return betas

    def computeGammas(self, alphas, betas):
        """Compute P(X[t] | Z^T)."""
        gamma = np.zeros_like(alphas, dtype=np.float128)
        for t in range(alphas.shape[1]):
            scale = np.dot(alphas[:, t], betas[:, t])
            for i in range(alphas.shape[0]):
                if scale > 0:
                    gamma[i, t] = np.exp(np.log(alphas[i, t]) + np.log(betas[i, t]) - np.log(scale))
        # pdb.set_trace()
        return gamma

    def computeXis(self, alphas, betas, observation):
        """Compute xi as an array comprised of each xi-xj pair."""

        # shape: (j, i, t)
        xis = np.zeros([self.numStates, self.numStates, observation.shape[0] - 1], dtype=np.float128)
        for t in range(observation.shape[0] - 1):
            for i in range(self.numStates):
                for j in range(self.numStates):
                    xis[j, i, t] = alphas[i, t] * self.T[j, i]\
                        * self.M[observation[t+1], j] * betas[j, t+1]
            scale = np.sum(xis[:, :, t], axis=(0,1))
            if scale > 0:
                xis[:, :, t] = np.exp(np.log(xis[:, :, t]) - np.log(scale))
        # pdb.set_trace()
        return xis

    def getAllLogOutputProb(self, observation, states):
        """Return all the log probability of an observation sequence."""

        likelihoods = np.zeros(states.shape[0], dtype=np.float128)
        for t in range(states.shape[0]):
            likelihoods[t] = self.getLogOutputProb(observation[t], states[t])
        # pdb.set_trace()
        return np.sum(likelihoods)

    def getLogStartProb(self, state):
        """Return the log probability of a particular state."""
        return np.log(self.pi[state])

    def getLogTransProb(self, fromState, toState):
        """Return the log probability associated with a state transition."""
        return np.log(self.T[toState, fromState])

    def getLogOutputProb(self, output, state):
        """Return the log probability of a state-dependent observation."""
        return np.log(self.M[output, state])

    def logMult(self, a, b):
        return np.exp(np.log(a) + np.log(b))

    def logDiv(self, a, b):
        return np.exp(np.log(a) - np.log(b))

    def sub2ind(self, i, j):
        """Convert subscript (i,j) to linear index."""
        return self.envShape[1]*i + j

    def sub2ind_all(self, states):
        """return the linear state sequence"""
        states = np.array(states)

        states_linear = np.zeros([states.shape[0], states.shape[1]], dtype=np.float128)
        for i in range(states.shape[0]):
            for t in range(states.shape[1]):
                states_linear[i, t] = self.sub2ind(states[i, t, 0], states[i, t, 1])

        states_linear = np.array(states_linear, dtype=np.int16)

        return states_linear

    def obs2ind(self, obs):
        """Convert observation string to linear index."""
        obsToInt = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
        return obsToInt[obs]

    def obs2ind_all(self, observations):
        """Convert observation string to linear index."""
        obsToInt = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
        k = np.array(list(obsToInt.keys()))
        v = np.array(list(obsToInt.values()), dtype=np.int16)

        observations = np.array(observations)

        for key,val in zip(k,v):
            observations[observations==key] = val

        observations = np.array(observations, dtype=np.int16)

        return observations

    def ind2sub_all(self, states):
        res_states = []
        for state in states:
            res_states.append((int(state // 4), int(state % 4)))

        return res_states

    def plot_likelihood(self, likelihoods):
        # Create the x-axis values
        x = np.arange(len(likelihoods))

        # Plot the curve
        plt.plot(x, likelihoods)

        # Add labels and a title if desired
        plt.xlabel('x-axis')
        plt.ylabel('likelihood')
        plt.title('Likelihood vs Iters')

        # Save the plot to a file
        plt.savefig('likelihoods.png')

        # Display the plot (optional, remove if you only want to save the plot)
        plt.show()