import numpy as np
import matplotlib.pyplot as plt
from Renderer import Renderer
import scipy
import pdb


class EKF(object):
    """A class for implementing EKFs.

    Attributes
    ----------
    mu : numpy.ndarray
        The N-element mean vector
    Sigma : numpy.ndarray
        The N x N covariance matrix
    R : numpy.ndarray
        The 3 x 3 covariance matrix for the additive Gaussian noise
        corresponding to motion uncertainty.
    Q : numpy.ndarray
        The 2 x 2 covariance matrix for the additive Gaussian noise
        corresponding to the measurement uncertainty.
    XYT : numpy.ndarray
        An N x T array of ground-truth poses, where each column corresponds
        to the ground-truth pose at that time step.
    MU : numpy.ndarray
        An N x (T + 1) array, where each column corresponds to mean of the
        posterior distribution at that time step. Note that there are T + 1
        entries because the first entry is the mean at time 0.
    VAR : numpy.ndarray
        An N x (T + 1) array, where each column corresponds to the variances
        of the posterior distribution (i.e., the diagonal of the covariance
        matrix) at that time step. As with MU, the first entry in VAR is
        the variance of the prior (i.e., at time 0)
    renderer : Renderer
        An instance of the Renderer class

    Methods
    -------
    getVariances()
        Return the diagonal of the covariance matrix.
    prediction (u)
        Perform the EKF prediction step with control u.
    update(z)
        Perform the EKF update step with measurement z.
    """

    def __init__(self, mu, Sigma, R, Q, XYT):
        """Initialize the class.

        Attributes
        ----------
        mu : numpy.ndarray
            The initial N-element mean vector for the distribution
        Sigma : numpy.ndarray
            The initial N x N covariance matrix for the distribution
        R : numpy.ndarray
            The N x N covariance matrix for the additive Gaussian noise
            corresponding to motion uncertainty.
        Q : numpy.ndarray
            The M x M covariance matrix for the additive Gaussian noise
            corresponding to the measurement uncertainty.
        XYT : numpy.ndarray
            An N x T array of ground-truth poses, where each column corresponds
            to the ground-truth pose at that time step.
        """
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q
        self.XYT = XYT

        # Keep track of mean and variance over time
        self.MU = mu
        self.VAR = np.diag(self.Sigma).reshape(3, 1)

        xLim = np.array((np.amin(XYT[0, :] - 2), np.amax(XYT[0, :] + 2)))
        yLim = np.array((np.amin(XYT[1, :] - 2), np.amax(XYT[1, :] + 2)))

        self.renderer = Renderer(xLim, yLim, 3, 'red', 'green')

    def angleWrap(self, theta):
        """Ensure that a given angle is in the interval (-pi, pi)."""
        while theta < -np.pi:
            theta = theta + 2*np.pi

        while theta > np.pi:
            theta = theta - 2*np.pi

        return theta

    def prediction(self, u):
        """Perform the EKF prediction step based on control u.

        Parameters
        ----------
        u : numpy.ndarray
            A 2-element vector that includes the forward distance that the
            robot traveled and its change in orientation.
        """
        # Your code goes here

        # computer F and G
        F = np.asarray([[1, 0, - u[0] * np.sin(self.mu[2])],
                        [0, 1, u[0] * np.cos(self.mu[2])],
                        [0, 0, 1]])
        G = np.asarray([[np.cos(self.mu[2]), 0],
                        [np.sin(self.mu[2]), 0],
                        [0, 1]])

        # Update Sigma
        new_Sigma = np.zeros([3,3])
        new_Sigma = F @ self.Sigma @ F.T + G @ self.R @ G.T

        self.Sigma = new_Sigma

        # Update mu
        new_mu = np.zeros(3)
        new_mu[0] = self.mu[0] + u[0] * np.cos(self.mu[2])
        new_mu[1] = self.mu[1] + u[0] * np.sin(self.mu[2])
        new_mu[2] = self.mu[2] + u[1]

        self.mu = new_mu



    def update(self, z):
        """Perform the EKF update step based on observation z.

        Parameters
        ----------
        z : numpy.ndarray
            A 2-element vector that includes the squared distance between
            the robot and the sensor, and the robot's heading.
        """
        # Your code goes here
        H = np.asarray([[2 * self.mu[0], 2 * self.mu[1], 0],
                        [- self.mu[1] / (self.mu[0] ** 2 + self.mu[1] ** 2),
                         self.mu[0] / (self.mu[0] ** 2 + self.mu[1] ** 2), 0]])
        K = self.Sigma @ H.T @ np.linalg.inv(H @ self.Sigma @ H.T + self.Q)

        h_mu = np.zeros(2)
        h_mu[0] = self.mu[0] ** 2 + self.mu[1] ** 2
        h_mu[1] = np.arctan2(self.mu[1], self.mu[0])

        diff = z - h_mu
        diff[1] = self.angleWrap(diff[1])

        self.mu = self.mu + K @ diff

        self.Sigma = self.Sigma - K @ H @ self.Sigma

    def run(self, U, Z):
        """Main EKF loop that iterates over control and measurement data.

        Parameters
        ----------
        U : numpy.ndarray
            A 2 x T array, where each column provides the control input
            at that time step.
        Z : numpy.ndarray
            A 2 x T array, where each column provides the measurement
            at that time step
        """
        for t in range(np.size(U, 1)):
            self.prediction(U[:, t])
            self.update(Z[:, t])

            self.MU = np.column_stack((self.MU, self.mu))
            self.VAR = np.column_stack((self.VAR, np.diag(self.Sigma)))

            self.renderer.render(self.mu, self.Sigma, self.XYT[:, t])

        self.renderer.drawTrajectory(self.MU[0:2, :], self.XYT[0:2, :])
        self.renderer.plotError(self.MU, self.XYT, self.VAR)
        plt.ioff()
        plt.show()
