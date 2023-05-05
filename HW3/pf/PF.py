import numpy as np
from Visualization import Visualization

#matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PF(object):
    """A class for implementing particle filters

        Attributes
        ----------
        numParticles : The number of particles to use
        particles :    A 3 x numParticles array, where each column represents a
                       particular particle, i.e., particles[:,i] = [x^(i), y^(i), theta^(i)]
        weights :      An array of length numParticles array, where each entry
                       denotes the weight of that particular particle
        Alpha :        Vector of 6 noise coefficients for the motion model
                       (See Table 5.3 in Probabilistic Robotics)
        laser :        Instance of the laser class that defines LIDAR params,
                       observation likelihood, and utils
        gridmap :      An instance of the Gridmap class that specifies
                       an occupancy grid representation of the map
                       where 1: occupied and 0: free
        visualize:     Boolean variable indicating whether to visualize
                       the particle filter


        Methods
        -------
        sampleParticlesUniform : Samples a set of particles according to a
                                 uniform distribution
        sampleParticlesGaussian: Samples a set of particles according to a
                                 Gaussian distribution over (x,y) and a
                                 uniform distribution over theta
        getParticle :            Returns the (x, y, theta) and weight associated
                                 with a particular particle id.
        getNormalizedWeights :   Returns the normalized particle weights (numpy.array)
        getMean :                Queries the sample-based estimate of the mean
        prediction :             Performs the prediction step
        update :                 Performs the update step
        run :                    The main loop of the particle filter

    """

    def __init__(self, numParticles, Alpha, laser, gridmap, visualize=True):
        """Initialize the class

            Args
            ----------
            numParticles : The number of particles to use
            Alpha :        Vector of 6 noise coefficients for the motion model
                           (See Table 5.3 in Probabilistic Robotics)
            laser :        Instance of the laser class that defines LIDAR params,
                           observation likelihood, and utils
            gridmap :      An instance of the Gridmap class that specifies
                           an occupancy grid representation of the map
                           here 1: occupied and 0: free
            visualize:     Boolean variable indicating whether to visualize
                           the particle filter (optional, default: True)
        """
        self.numParticles = numParticles
        self.Alpha = Alpha
        self.laser = laser
        self.gridmap = gridmap
        self.visualize = visualize

        # particles is a numParticles x 3 array, where each column denote a particle_handle
        # weights is a numParticles x 1 array of particle weights
        self.particles = None
        self.weights = None

        if self.visualize:
            self.vis = Visualization()
            self.vis.drawGridmap(self.gridmap)
        else:
            self.vis = None

    def sampleParticlesUniform(self):
        """
            Samples the set of particles according to a uniform distribution and
            sets the weights to 1/numParticles. Particles in collision are rejected
        """

        (m, n) = self.gridmap.getShape()

        self.particles = np.empty([3, self.numParticles])

        for i in range(self.numParticles):
            theta = np.random.uniform(-np.pi, np.pi)
            inCollision = True
            while inCollision:
                x = np.random.uniform(0, (n-1)*self.gridmap.xres)
                y = np.random.uniform(0, (m-1)*self.gridmap.yres)

                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:, i] = np.array([[x, y, theta]])

        self.weights = (1./self.numParticles) * np.ones((1, self.numParticles))

    def sampleParticlesGaussian(self, x0, y0, sigma):
        """
            Samples the set of particles according to a Gaussian distribution
            Orientation are sampled from a uniform distribution

            Args
            ----------
            x0 :           Mean x-position
            y0  :          Mean y-position
                           (See Table 5.3 in Probabilistic Robotics)
            sigma :        Standard deviation of the Gaussian
        """

        (m, n) = self.gridmap.getShape()

        self.particles = np.empty([3, self.numParticles])

        for i in range(self.numParticles):
            inCollision = True
            while inCollision:
                x = np.random.normal(x0, sigma)
                y = np.random.normal(y0, sigma)
                theta = np.random.uniform(-np.pi, np.pi)

                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:, i] = np.array([[x, y, theta]])

        self.weights = (1./self.numParticles) * np.ones((1, self.numParticles))

    def getParticle(self, k):
        """
            Returns desired particle (3 x 1 array) and weight

            Args
            ----------
            k :   Index of desired particle

            Returns
            -------
            particle :  The particle having index k
            weight :    The weight of the particle
        """

        if k < self.particles.shape[1]:
            return self.particles[:, k], self.weights[:, k]
        else:
            print('getParticle: Request for k=%d exceeds number of particles (%d)' % (k, self.particles.shape[1]))
            return None, None

    def getNormalizedWeights(self):
        """
            Returns an array of normalized weights

            Returns
            -------
            weights :  An array of normalized weights (numpy.array)
        """

        return self.weights/np.sum(self.weights)

    def getMean(self):
        """
            Returns the mean of the particle filter distribution

            Returns
            -------
            mean :  The mean of the particle filter distribution (numpy.array)
        """

        weights = self.getNormalizedWeights()
        return np.sum(np.tile(weights, (self.particles.shape[0], 1)) * self.particles, axis=1)

    def render(self, ranges, deltat, XGT):
        """
            Visualize filtering strategies

            Args
            ----------
            ranges :   LIDAR ranges (numpy.array)
            deltat :   Step size
            XGT :      Ground-truth pose (numpy.array)
        """

        self.vis.drawParticles(self.particles)
        if XGT is not None:
            self.vis.drawLidar(ranges, self.laser.Angles, XGT[0], XGT[1], XGT[2])
            self.vis.drawGroundTruthPose(XGT[0], XGT[1], XGT[2])
        mean = self.getMean()
        self.vis.drawMeanPose(mean[0], mean[1])
        plt.pause(deltat)

    def prediction(self, u, deltat):
        """
            Implement the proposal step using the motion model based in inputs
            v (forward velocity) and w (angular velocity) for deltat seconds

            This model corresponds to that in Table 5.3 in Probabilistic Robotics

            Args
            ----------
            u :       Two-vector of control inputs (numpy.array)
            deltat :  Step size
        """

        # TODO: Your code goes here: Implement the algorithm given in Table 5.3
        # Note that the "sample" function in the text assumes zero-mean
        # Gaussian noise. You can use the NumPy random.normal() function
        # Be sure to reject samples that are in collision
        # (see Gridmap.inCollision), and to unwrap orientation so that it
        # it is between -pi and pi.

        # Hint: Repeatedly calling np.random.normal() inside a for loop
        #       can consume a lot of time. You may want to consider drawing
        #       n (e.g., n=10) samples of each noise term at once
        #       (drawing n samples is faster than drawing 1 sample n times)
        #       and if none of the estimated poses are not in collision, assume
        #       that the robot doesn't move from t-1 to t.

        # Getting Gaussian noise for each self.alpha and time step
        vt1 = np.random.normal(0, self.Alpha[0] * (u[0] ** 2) + self.Alpha[1] * (u[1] ** 2), self.numParticles)
        vt2 = np.random.normal(0, self.Alpha[2] * (u[0] ** 2) + self.Alpha[3] * (u[1] ** 2), self.numParticles)
        gammat = np.random.normal(0, self.Alpha[4] * (u[0] ** 2) + self.Alpha[5] * (u[1] ** 2), self.numParticles)

        # Adding noise to movement
        ut1 = u[0] + vt1
        ut2 = u[1] + vt2

        # Calculating new position and orientation of particles
        self.particles[0, :] = self.particles[0, :] + ut1 / ut2 * (np.sin(self.particles[2, :] + ut2 * deltat) - np.sin(self.particles[2, :]))
        self.particles[1, :] = self.particles[1, :] + ut1 / ut2 * (-np.cos(self.particles[2, :] + ut2 * deltat) + np.cos(self.particles[2, :]))
        self.particles[2, :] = self.particles[2, :] + ut2 * deltat + gammat * deltat

        # Rejecting samples that are in collision
        keepindex = np.ones(self.numParticles, dtype=bool)
        for i in range(self.numParticles):
            if self.gridmap.inCollision(self.particles[0, i], self.particles[1, i]):
                keepindex[i] = False
                # self.particles[0, i] = self.particles[0, i] - ut1[i] / ut2[i] * (np.sin(self.particles[2, i] + ut2[i] * deltat) - np.sin(self.particles[2, i]))
                # self.particles[1, i] = self.particles[1, i] - ut1[i] / ut2[i] * (-np.cos(self.particles[2, i] + ut2[i] * deltat) + np.cos(self.particles[2, i]))
                # self.particles[2, i] = self.particles[2, i] - ut2[i] * deltat - gammat[i] * deltat
        self.particles = self.particles[:, keepindex]

        # Unwrapping orientation so that it is between -pi and pi
        self.particles[2, :] = np.unwrap(self.particles[2, :])

    def resample(self):
        """
            Perform resampling with replacement
        """

        # TODO: Your code goes here
        # The np.random.choice function may be useful

        # Resampling
        indices = np.arange(0, self.particles.shape[-1])
        indices = np.random.choice(indices, self.numParticles, p=self.weights / np.sum(self.weights))

        self.particles = self.particles[:, indices]
        self.weights = self.weights[indices]

    def update(self, ranges):
        """
            Implement the measurement update step

            Args
            ----------
            ranges :    Array of LIDAR ranges (numpy.array)
        """
        # TODO: Your code goes here
        self.weights = self.laser.scanProbability(ranges, self.particles, self.gridmap)

    def run(self, U, Ranges, deltat, X0, XGT, filename):
        """
            The main loop that runs the particle filter

            Args
            ----------
            U :      An array of control inputs, one column per time step (numpy.array)
            Ranges : An array of LIDAR ranges (numpy,array)
            deltat : Duration of each time step
            X0 :     The initial pose (may be None) (numpy.array)
            XGT :    An array of ground-truth poses (may be None) (numpy.array)
        """

        # TODO: Try different sampling strategies (including different values for sigma)
        sampleGaussian = False
        NeffThreshold = self.numParticles/10
        if sampleGaussian and (X0 is not None):
            sigma = 0.5
            self.sampleParticlesGaussian(X0[0, 0], X0[1, 0], sigma)
        else:
            self.sampleParticlesUniform()

        # Iterate over the data
        for k in range(U.shape[1]):
            u = U[:, k]
            ranges = Ranges[:, k]

            # TODO: Your code goes here
            self.prediction(u, deltat)
            self.update(ranges)
            self.resample()

            if not self.weights.shape[0] == self.particles.shape[1]:
                import pdb; pdb.set_trace()

            if self.visualize:
                if XGT is None:
                    self.render(ranges, deltat, None)
                else:
                    self.render(ranges, deltat, XGT[:, k])

        plt.savefig(filename, bbox_inches='tight')
