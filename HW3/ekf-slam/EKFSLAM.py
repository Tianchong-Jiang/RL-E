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
        self.R = np.asarray([[R[0,0], 0, 0], [0, R[0, 0], 0], [0, 0, R[1, 1]]])
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
        F = np.eye(3)
        F[0, 2] = -u[0] * np.sin(self.mu[2])
        F[1, 2] = u[0] * np.cos(self.mu[2])

        # Predict the covariance
        self.Sigma[0:3, 0:3] = F @ self.Sigma[0:3, 0:3] @ F.T + self.R
        self.Sigma[0:3, 3:] = F @ self.Sigma[0:3, 3:]
        self.Sigma[3:, 0:3] = self.Sigma[3:, 0:3] @ F.T

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
        id = str(int(id))
        if id not in self.mapLUT.keys():
            return
        pos = self.mapLUT[id]

        # Compute Jacobian of measurement model
        H = np.zeros((2, 3))
        H[0, 0] = -np.cos(self.mu[2])
        H[0, 1] = -np.sin(self.mu[2])
        H[0, 2] = -z[0] * np.sin(self.mu[2]) + z[1] * np.cos(self.mu[2])
        H[1, 0] = np.sin(self.mu[2])
        H[1, 1] = -np.cos(self.mu[2])
        H[1, 2] = z[0] * np.cos(self.mu[2]) + z[1] * np.sin(self.mu[2])

        # Compute Kalman gain
        K = self.Sigma[0:3, 0:3] @ H.T  @ np.linalg.inv(H @ self.Sigma[0:3, 0:3] @ H.T + self.Q)

        # Update the mean
        h_mu = np.zeros(2)
        h_mu[0] = np.cos(self.mu[2]) * (self.mu[pos] - self.mu[0]) + np.sin(self.mu[2]) * (self.mu[pos + 1] - self.mu[1])
        h_mu[1] = -np.sin(self.mu[2]) * (self.mu[pos] - self.mu[0]) + np.cos(self.mu[2]) * (self.mu[pos + 1] - self.mu[1])
        self.mu[0:3] = self.mu[0:3] + K @ (z - h_mu)

        # Update the covariance
        self.Sigma[0:3, 0:3] = (np.eye(3) - K @ H) @ self.Sigma[0:3, 0:3]

        # H = np.zeros((2, self.mu.shape[0]))
        # H[0, 0] = -np.cos(self.mu[2])
        # H[0, 1] = -np.sin(self.mu[2])
        # H[0, 2] = -np.sin(self.mu[2]) * (z[0] - self.mu[0]) + np.cos(self.mu[2])*(z[1] - self.mu[1])
        # H[0, midx] = np.cos(self.mu[2])
        # H[0, midx + 1] = np.sin(self.mu[2])

        # H[1, 0] = np.sin(self.mu[2])
        # H[1, 1] = -np.cos(self.mu[2])
        # H[1, 2] = -np.cos(self.mu[2]) * (z[0] - self.mu[0]) - np.sin(self.mu[2])*(z[1] - self.mu[1])
        # H[1, midx] = -np.sin(self.mu[2])
        # H[1, midx] = np.cos(self.mu[2])

        # K = self.Sigma@H.transpose()@np.linalg.inv(H@self.Sigma@H.transpose() + self.Q)
        # rotation_mat = np.matrix([[np.cos(self.mu[2]), np.sin(self.mu[2])], [-np.sin(self.mu[2]), np.cos(self.mu[2])]])
        # z_pred = np.asarray(rotation_mat@(self.mu[midx:midx+2] - self.mu[0:2])).reshape(-1)
        # self.mu = self.mu + K@(z - z_pred)
        # self.Sigma = (np.eye( self.mu.shape[0]) - K@H)@self.Sigma


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
        id = str(int(id))
        if id in self.mapLUT.keys():
            return

        # # Add the landmark to LUT
        mu_length = int(self.mu.shape[0])
        self.mapLUT[id] = mu_length

        # Compute the landmark position in the world frame
        x = self.mu[0] + z[0] * np.cos(self.mu[2]) - z[1] * np.sin(self.mu[2])
        y = self.mu[1] + z[0] * np.sin(self.mu[2]) + z[1] * np.cos(self.mu[2])

        # Augment the mean
        self.mu = np.append(self.mu, np.array((x, y)))

        # Compute G
        G = np.zeros((2, mu_length))
        G[0, 0] = 1
        G[0, 2] = - z[0] * np.sin(self.mu[2]) - z[1] * np.cos(self.mu[2])
        G[1, 1] = 1
        G[1, 2] = z[0] * - np.cos(self.mu[2]) + z[1] * np.sin(self.mu[2])

        # Augment the covariance
        length = self.Sigma.shape[0]
        newSigma = np.zeros((length + 2, length + 2))
        newSigma[:-2, :-2] = self.Sigma
        newSigma[-2:, -2:] = G @ self.Sigma @ G.T + self.Q
        newSigma[-2:, :-2] = G @ self.Sigma
        newSigma[:-2, -2:] = self.Sigma @ G.T
        self.Sigma = newSigma

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

        for t in range(U.shape[1]):
            self.prediction(U[1:, t])
            self.renderer.render(self.mu, self.Sigma, self.XGT[1:4, t], Z[:, Z[0,:]==t-1], self.mapLUT)
            # import pdb; pdb.set_trace()
            for j in range(Z.shape[1]):
                if Z[0, j] == t:
                    self.update(Z[2:4, j], Z[1, j])
                    self.augmentState(Z[2:4, j], Z[1, j])
            self.renderer.render(self.mu, self.Sigma, self.XGT[1:4, t], Z[:, Z[0,:]==t], self.mapLUT)
            # import pdb; pdb.set_trace()


