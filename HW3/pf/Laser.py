import numpy as np
from scipy.stats import norm, expon

class Laser(object):
    """A class specifying a LIDAR beam model based on Section 6.3.1 of Probabilistic Robotics,
       which is comprised of a mixture of different components, whose parameters are described below.

       Note that we use pX to denote the weight associated with the X component,
       whereas the text uses zX to denote the weight

        Attributes
        ----------
        pHit :        Likelihood (weight) of getting a valid return (subject to noise)
        pShort :      Likelihood (weight) of getting a short return
        pMax :        Likelihood (weight) of getting a false max range return
        pRand :       Likelihood (weight) of a random range in interval [0, zMax]
        sigmaHit :    Standard deviation of the Gaussian noise that corrupts true range
        lambdaShort : Parameter of model determining likelihood of a short return
                      (e.g., due to an object not in the map)
        zMax:         Maximum sensor range
        zMaxEps :


        Methods
        -------
        scanProbability : Computes the likelihood of a given LIDAR scan from
                          a given pose in a given map
        getXY :           Function that converts the range and bearing to Cartesian
                          coordinates in the LIDAR frame
        rayTracing :      Perform ray tracing from a given pose to predict range and bearing returns
    """

    def __init__(self, numBeams=41, sparsity=1):
        """Initialize the class

            Args
            ----------
            numBeams :    Number of beams in an individual scan (optional, default: 41)
            sparsity :    Downsample beams by taking every sparsity-1 beam (optional, default: 1)
        """
        self.pHit = 0.97
        self.pShort = 0.01
        self.pMax = 0.01
        self.pRand = 0.01
        self.sigmaHit = 0.5
        self.lambdaShort = 1
        self.zMax = 20
        self.zMaxEps = .1
        self.Angles = np.linspace(-np.pi, np.pi, numBeams)# array of angles
        self.Angles = self.Angles[::sparsity]

        # Pre-compute for efficiency
        self.normal = norm(0, self.sigmaHit)
        self.exponential = expon(scale=1/self.lambdaShort)

    def scanProbability(self, z, x, gridmap):
        """The following computes the likelihood of a given LIDAR scan from
           a given pose in the provided map according to the algorithm given
           in Table 6.1 of Probabilistic Robotics

            Args
            -------
            z :           An array of LIDAR ranges, one entry per beam (numpy.array)
            x :           An array specifying the LIDAR (x, y, theta) pose (numpy.array)
            gridmap :     The map of the environment as a gridmap

            Returns
            -------
            likelihood :  The probability of the scan.
        """

        # TODO: Your code goes here
        # Implement the algorithm given in Table 6.1
        # You are provided with an implementation (albeit slow) of ray tracing below

        # Compute zstar using ray tracing
        zstar, coords = self.rayTracing(x[0], x[1], x[2], self.Angles, gridmap)

        # Compute the likelihood of the scan
        import pdb; pdb.set_trace()



    def getXY(self, range, bearing):
        """Function that converts the range and bearing to
           Cartesian coordinates in the LIDAR frame

            Args
            -------
            range :   A 1 x n array of LIDAR ranges (numpy.ndarray)
            bearing : A 1 x n array of LIDAR bearings (numpy.ndarray)

            Returns
            -------
            XY :      A 2 x n array, where each column is an (x, y) pair
        """

        CosSin = np.vstack((np.cos(bearing[:]), np.sin(bearing[:])))
        XY = np.tile(range, (2, 1))*CosSin

        return XY

    def rayTracing(self, xr, yr, thetar, lAngle, gridmap):
        """A vectorized implementation of ray tracing

            Args
            -------
            (xr, yr, thetar):   The robot's pose
            lAngle:             The LIDAR angle (in the LIDAR reference frame)
            gridmap:            An instance of the Gridmap class that specifies
                                an occupancy grid representation of the map
                                where 1: occupied and 0: free
            bearing : A 1 x n array of LIDAR bearings (numpy.ndarray)

            Returns
            -------
            d:                  Range
            coords:             Array of (x,y) coordinates
        """

        angle = np.array(thetar[:, None] + lAngle[None])
        x0 = np.array(xr/gridmap.xres)
        y0 = np.array(yr/gridmap.yres)

        x0 = np.tile(x0[:, None], [1, angle.shape[1]])
        y0 = np.tile(y0[:, None], [1, angle.shape[1]])
        assert angle.shape == x0.shape
        assert angle.shape == y0.shape

        def inCollision(x, y):
            return gridmap.inCollision(np.floor(x).astype(np.int32), np.floor(y).astype(np.int32), True)

        (m,n) = gridmap.getShape()
        in_collision = inCollision(x0, y0)

        x0[x0 == np.floor(x0)] += 0.001
        y0[y0 == np.floor(y0)] += 0.001
        eps = 0.0001

        def inbounds(x, low, high):
            # return x in [low, high)
            return (x < high) * (x >= low)

        # Intersection with horizontal lines
        x = x0.copy()
        y = y0.copy()
        dir = np.tan(angle)
        xh = np.zeros_like(x)
        yh = np.zeros_like(y)
        foundh = np.zeros(x.shape, dtype=np.bool)
        seps = np.sign(np.cos(angle)) * eps
        while np.any(inbounds(x, 1, n)) and not np.all(foundh):
            x = np.where(seps > 0, np.floor(x+1), np.ceil(x-1))
            y = y0 + dir*(x-x0)
            inds = inCollision(x+seps, y) * np.logical_not(foundh) * inbounds(y, 0, m)
            if np.any(inds):
                xh[inds] = x[inds]
                yh[inds] = y[inds]
                foundh[inds] = True

        # Intersection with vertical lines
        x = x0.copy()
        y = y0.copy()
        eps = 1e-6
        dir = 1. / (np.tan(angle) + eps)
        xv = np.zeros_like(x)
        yv = np.zeros_like(y)
        foundv = np.zeros(x.shape, dtype=np.bool)
        seps = np.sign(np.sin(angle)) * eps
        while np.any(inbounds(y, 1, m)) and not np.all(foundv):
            y = np.where(seps > 0, np.floor(y+1), np.ceil(y-1))
            x = x0 + dir*(y-y0)
            inds = inCollision(x,y+seps) * np.logical_not(foundv) * inbounds(x, 0, n)
            if np.any(inds):
                xv[inds] = x[inds]
                yv[inds] = y[inds]
                foundv[inds] = True

        if not np.all(foundh + foundv):
            assert False, 'rayTracing: Error finding return'

        # account for poses in collision
        xh[in_collision] = x0[in_collision]
        yh[in_collision] = y0[in_collision]

        # get dist and coords
        dh = np.square(xh - x0) + np.square(yh - y0) + 1e7 * np.logical_not(foundh)
        dv = np.square(xv - x0) + np.square(yv - y0) + 1e7 * np.logical_not(foundv)
        d = np.where(dh < dv, dh, dv)
        cx = np.where(dh < dv, xh, xv)
        cy = np.where(dh < dv, yh, yv)
        coords = np.stack([cx, cy], axis=-1)
        return np.sqrt(d), coords
