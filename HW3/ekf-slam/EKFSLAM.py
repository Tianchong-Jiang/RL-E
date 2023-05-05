import numpy as np
import matplotlib.pyplot as plt
from Renderer import Renderer
from Visualization import Visualization

class EKFSLAM(object):
    """A class for implementing EKF-based SLAM

        Attributes
        ----------
        mu :           The mean vector (numpy.array)
        Sigma :        The covariance matrix (numpy.array)
        R :            The process model covariance matrix (numpy.array)
        Q :            The measurement model covariance matrix (numpy.array)
        XGT :          Array of ground-truth poses (optional, may be None) (numpy.array)
        MGT :          Ground-truth map (optional, may be None)

        Methods
        -------
        prediction :   Perform the prediction step
        update :       Perform the measurement update step
        augmentState : Add a new landmark(s) to the state
        run :          Main EKF-SLAM loop
        render :       Render the filter
    """

    def __init__(self, mu, Sigma, R, Q, XGT = None, MGT = None):
        """Initialize the class

            Args
            ----------
            mu :           The initial mean vector (numpy.array)
            Sigma :        The initial covariance matrix (numpy.array)
            R :            The process model covariance matrix (numpy.array)
            Q :            The measurement model covariance matrix (numpy.array)
            XGT :          Array of ground-truth poses (optional, may be None) (numpy.array)
            MGT :          Ground-truth map (optional, may be None)
        """
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q

        self.XGT = XGT
        self.MGT = MGT

        if (self.XGT is not None and self.MGT is not None):
            xmin = min(np.amin(XGT[1, :]) - 2, np.amin(MGT[1, :]) - 2)
            xmax = min(np.amax(XGT[1, :]) + 2, np.amax(MGT[1, :]) + 2)
            ymin = min(np.amin(XGT[2, :]) - 2, np.amin(MGT[2, :]) - 2)
            ymax = min(np.amax(XGT[2, :]) + 2, np.amax(MGT[2, :]) + 2)
            xLim = np.array((xmin, xmax))
            yLim = np.array((ymin, ymax))
        else:
            xLim = np.array((-8.0, 8.0))
            yLim = np.array((-8.0, 8.0))

        self.renderer = Renderer(xLim, yLim, 3, 'red', 'green')

        # Draws the ground-truth map
        if self.MGT is not None:
            self.renderer.drawMap(self.MGT)


        # You may find it useful to keep a dictionary that maps a feature ID
        # to the corresponding index in the mean vector and covariance matrix
        self.mapLUT = {}

    def prediction(self, u):
        """Perform the prediction step to determine the mean and covariance
           of the posterior belief given the current estimate for the mean
           and covariance, the control data, and the process model

            Args
            ----------
            u :  The forward distance and change in heading (numpy.array)
        """

        # TODO: Your code goes here

        # Predict the mean
        self.mu[0] = self.mu[0] + u[0] * np.cos(self.mu[2])
        self.mu[1] = self.mu[1] + u[0] * np.sin(self.mu[2])
        self.mu[2] = self.angleWrap(self.mu[2] + u[1])

        # Compute Jacobian of motion model
        F = np.zeros((2, 3))
        F[0, 0] = 1
        F[0, 2] = -u[0] * np.sin(self.mu[2])
        F[1, 1] = 1
        F[1, 2] = u[0] * np.cos(self.mu[2])

        # Predict the covariance
        import pdb; pdb.set_trace()
        self.Sigma[0:2, 0:2] = F @ self.Sigma[0:3, 0:3] @ F.T + self.R
        self.Sigma[0:2, 2:] = F @ self.Sigma[0:3, 3:]
        self.Sigma[2:, 0:2] = self.Sigma[0:3, 3:] @ F.T

    def update(self, z, id):
        """Perform the measurement update step to compute the posterior
           belief given the predictive posterior (mean and covariance) and
           the measurement data

            Args
            ----------
            z :  The Cartesian coordinates of the landmark
                 in the robot's reference frame (numpy.array)
            id : The ID of the observed landmark (int)
        """
        # TODO: Your code goes here

        # Compute Jacobian of measurement model
        H = np.zeros((2, 3))
        H[0, 0] = -np.cos(self.mu[2])
        H[0, 1] = -np.sin(self.mu[2])
        H[0, 2] = -z[0] * np.sin(self.mu[2]) + z[1] * np.cos(self.mu[2])
        H[1, 0] = np.sin(self.mu[2])
        H[1, 1] = -np.cos(self.mu[2])
        H[1, 2] = z[0] * np.cos(self.mu[2]) + z[1] * np.sin(self.mu[2])

        # Compute Kalman gain
        K = self.Sigma[0:3, 0:3] @ H.T  @ np.linalg.inv(self.H @ self.Sigma[0:3, 0:3] @ H.T + self.Q)

        # Update the mean
        self.mu[0:3] = self.mu[0:3] + K @ (z - self.mu[0:3])

        # Update the covariance
        self.Sigma[0:3, 0:3] = (np.eye(3) - K @ H) @ self.Sigma[0:3, 0:3]


    def augmentState(self, z, id):
        """Augment the state vector to include the new landmark

            Args
            ----------
            z :  The Cartesian coordinates of the landmark
                 in the robot's reference frame (numpy.array)
            id : The ID of the observed landmark
        """

        # TODO: Your code goes here

        # Check if the landmark has already been added
        if id in self.mapLUT.items():
            return

        # Add the landmark to LUT
        self.mapLUT[id] = (self.mu.shape[0] - 1) * 0.5

        # Compute the landmark position in the world frame
        x = self.mu[0] + z[0] * np.cos(self.mu[2]) - z[1] * np.sin(self.mu[2])
        y = self.mu[1] - z[0] * np.sin(self.mu[2]) - z[1] * np.cos(self.mu[2])

        # Augment the mean
        self.mu = np.append(self.mu, np.array((x, y)))

        # Compute G
        G = np.asarray([[1, 0, z[0] * np.sin(self.mu[2]) + z[1] * np.cos(self.mu[2])],
                        [0, 1, z[0] * - np.cos(self.mu[2]) + z[1] * np.sin(self.mu[2])]])

        # Augment the covariance
        self.Sigma[0:3, 3:] = G @ self.Sigma[0:3, 3:]
        self.Sigma[3:, 0:3] = self.Sigma[0:3, 3:] @ G.T
        self.Sigma[3:, 3:] = G @ self.Sigma[3:, 3:] @ G.T + self.Q


    def angleWrap(self, theta):
        """Ensure that a given angle is in the interval (-pi, pi)."""
        while theta < -np.pi:
            theta = theta + 2*np.pi

        while theta > np.pi:
            theta = theta - 2*np.pi

        return theta

    def run(self, U, Z):
        """The main loop of EKF-based SLAM

            Args
            ----------
            U :   Array of control inputs, one column per time step (numpy.array)
            Z :   Array of landmark observations in which each column
                  [t; id; x; y] denotes a separate measurement and is
                  represented by the time step (t), feature id (id),
                  and the observed (x, y) position relative to the robot
        """
        # TODO: Your code goes here

        # You may want to call the visualization function between filter steps where
        #       self.XGT[1:4, t] is the column of XGT containing the pose the current iteration
        #       Zt are the columns in Z for the current iteration
        #       self.mapLUT is a dictionary where the landmark IDs are the keys
        #                   and the index in mu is the value
        #
        # self.renderer.render(self.mu, self.Sigma, self.XGT[1:4, t], Zt, self.mapLUT)

        for i in range(U.shape[1]):
            self.prediction(U[:, i])
            for j in range(Z.shape[1]):
                if Z[0, j] == i:
                    self.update(Z[2:4, j], Z[1, j])
                    self.augmentState(Z[2:4, j], Z[1, j])
            self.renderer.render(self.mu, self.Sigma, self.XGT[1:4, i], Z[:, Z[0, :] == i], self.mapLUT)
