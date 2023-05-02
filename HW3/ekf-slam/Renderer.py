import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import LineCollection


class Renderer(object):
    """A class that provides rendering utilities to visualize EKF-based SLAM.

    Attributes
    ----------
    numSigma : int
        Number of standard deviations to use when rendering uncertainty
    estColor : str
        String that specifies the color to use for visualizing estimates
    gtColor : str
        String that specifies the color to use for visualizing ground truth
    estTriangle : matplotlib.patches.Polygon
        A polygon corresponding to the estimated pose of the robot
    gtTriangle : matplotlib.patches.Polygon
        A polygon corresponding to the ground-truth pose of the robot
    ellipse : matplotlib.patches.Polygon
        A polygon correponding to the uncertainty ellipse

    Methods
    -------
    drawTriangle(xy, theta):
        Create a triangle polygon centered at xy with orientation theta.
    updateTriangle(triangle, xy, theta):
        Update the triangle to be centered at xy with orientation theta.
    drawCircle(xy):
        Create a circle centered at xy.
    updateCircle(point, xy):
        Update the circle to be centered at xy.
    drawEllipse(xy, Sigma):
        Create an ellipse corresponding to the provided covariance
    updateEllipse(ellipse, xy, Sigma):
        Update the given ellipse based on the provided covariance matrix.
    render()
        Render the current estimate and ground-truth pose
    """

    def __init__(self, xLim, yLim, numSigma=3, estColor='red', gtColor='green'):
        """Initialize the class.

        Attributes
        ----------
        xLim : numpy.ndarray
            A 2-element array specifying the minimum (xLim[0]) and maximum
            (xLim[1]) x-axis coordinates of the environment.
        yLim : numpy.ndarray
            A 2-element array specifying the minimum (yLim[0]) and maximum
            (yLim[1]) y-axis coordinates of the environment.
        numSigma : int
            Number of standard deviations to use when rendering uncertainty
        estColor : str, optional
            String that specifies the color to use for visualizing estimates
            Optional (default: 'red')
        gtColor : str
            String that specifies the color to use for visualizing ground truth
            Optional (default: 'green')
        """
        self.numSigma = numSigma
        self.estColor = estColor
        self.gtColor = gtColor

        # Rendering the estimated and ground-truth pose
        self.estTriangle = None
        self.gtTriangle = None
        self.ellipse = None
        self.line_collection = None
        # Dictionaries for map handles
        self.estMapCircles = {}
        self.estMapEllipses = {}
        self.gtMapCircles = {}

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect="equal")

        self.ax.set_xlim(xLim[0], xLim[1])
        self.ax.set_ylim(yLim[0], yLim[1])
        plt.ion()

    def drawEllipse(self, xy, Sigma):
        """Draw the ellipse corresponding to a given covariance matrix.

        Attributes
        ----------
        xy : numpy.ndarray
            2-element array specifying the coordinates of the center.
        Sigma : numpy.ndarray
            A 2 x 2 covariance matrix.

        Returns
        -------
        ellipse : matplotlib.patches.Polygon
            A polygon representation of the ellipse
        """
        if np.shape(Sigma) == (3, 3):
            Sigma = Sigma[0:2, 0:2]

        (W, V) = np.linalg.eig(Sigma)

        a = np.arange(0, 2*np.pi, 0.1).T
        b = self.numSigma * np.array((np.cos(a), np.sin(a)))

        D = np.sqrt(np.diag(W, 0))
        el = V.dot(D.dot(b))

        # Shift the ellipse to be centered at xy
        el = el + np.array([xy, ] * np.shape(el)[1]).T

        ellipse = Polygon(el.T, ec=self.estColor, fill=False)

        return ellipse

    def updateEllipse(self, ellipse, xy, Sigma):
        """Updates the given ellipse based on given covariance matrix.

        Attributes
        ----------
        ellipse : matplotlib.patches.Polygon
            A polygon representation of the ellipse
        xy : numpy.ndarray
            2-element array specifying the coordinates of the center.
        Sigma : numpy.ndarray
            A 2 x 2 covariance array.

        Returns
        -------
        ellipse : matplotlib.patches.Polygon
            A polygon representation of the ellipse
        """
        if np.shape(Sigma) == (3, 3):
            Sigma = Sigma[0:2, 0:2]

        (W, V) = np.linalg.eig(Sigma)

        a = np.arange(0, 2*np.pi, 0.1).T
        b = self.numSigma * np.array((np.cos(a), np.sin(a)))

        D = np.sqrt(np.diag(W, 0))
        el = V.dot(D.dot(b))

        # Shift the ellipse to be centered at xy
        el = el + np.array([xy, ] * np.shape(el)[1]).T

        ellipse.set_xy(el.T)

        return ellipse

    def drawTriangle(self, xy, theta, mycolor='red'):
        """Create a triangle polygon centered at xy with orientation theta.

        Attributes
        ----------
        xy : numpy.ndarray
            2-element array specifying the coordinates of the center.
        theta :
            The orientation in radians.
        mycolor : str
            A string specifying the color to use for rendering.

        Returns
        -------
        triangle : matplotlib.patches.Polygon
            A polygon representation of the triangle
        """
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        length = 0.75
        alpha = np.radians(30)
        a = length * np.cos(alpha/2)
        b = length * np.sin(alpha/2)
        v1 = np.array((a/2, 0))
        v2 = np.array((-a/2, b))
        v3 = np.array((-a/2, -b))

        v1 = R.dot(v1) + xy
        v2 = R.dot(v2) + xy
        v3 = R.dot(v3) + xy
        triangle = Polygon((v1, v2, v3), color=mycolor)

        return triangle

    def updateTriangle(self, triangle, xy, theta):
        """Update the triangle's vertices.

        Attributes
        ----------
        triangle : matplotlib.patches.Polygon
            A polygon representation of the triangle.
        xy : numpy.ndarray
            2-element array specifying the coordinates of the center.
        theta :
            The orientation in radians.

        Returns
        -------
        triangle : matplotlib.patches.Polygon
            A polygon representation of the triangle.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        length = 0.75
        alpha = np.radians(30)
        a = length * np.cos(alpha/2)
        b = length * np.sin(alpha/2)
        v1 = np.array((a/2, 0))
        v2 = np.array((-a/2, b))
        v3 = np.array((-a/2, -b))

        v1 = R.dot(v1) + xy
        v2 = R.dot(v2) + xy
        v3 = R.dot(v3) + xy

        triangle.set_xy((v1, v2, v3))

        return triangle

    def drawCircle(self, xy, mycolor='red'):
        """Create a circle polygon centered at xy with orientation theta.

        Attributes
        ----------
        xy : numpy.ndarray
            2-element array specifying the coordinates of the center.
        mycolor : str
            A string specifying the color to use for rendering.

        Returns
        -------
        circle : matplotlib.patches.Circle
                 A polygon representation of the triangle
        """
        _radius = 0.1
        circle = Circle((xy[0], xy[1]), radius=_radius, color=mycolor)

        return circle

    def updateCircle(self, circle, xy):
        """Update the circle to be centered at xy

        Attributes
        ----------
        circle : matplotlib.patches.Circle
            A polygon representation of the circle.
        xy : numpy.ndarray
            2-element array specifying the coordinates of the center.

        Returns
        -------
        circle : matplotlib.patches.Circle
            A polygon representation of the circle.
        """

        circle.center = xy[0], xy[1]

        return circle

    def drawLineCollection(self, mu, xy, mycolor='red'):
        """Create a line collection, all having one vertex at mu and the second vertex
           being the coordinates of a column of xy

        Attributes
        ----------
        mu : numpy.ndarray
            2-element array specifying the coordinates of the robot
        xy : numpy.ndarray
            2 x m array specifying the coordinates of the landmark
        mycolor : str
            A string specifying the color to use for rendering.

        Returns
        -------
        line_collection :  matplotlib.collections.LineCollection
                           A line collection
        """

        lines = np.array([]).reshape(0, 2)

        for j in range(xy.shape[1]):
            temp = np.array([[mu[0], mu[1]], [xy[0, j], xy[1, j]]])
            lines = np.concatenate((lines, np.array([[mu[0], mu[1]], [xy[0, j], xy[1, j]]])))

        line_collection = LineCollection([lines], color=mycolor, linestyles='dashed')

        return line_collection

    def updateLineCollection(self, line_collection, mu, xy):
        """Update the vertices of the line collection

        Attributes
        ----------
        line_collection :  matplotlib.collections.LineCollection
                           A line collection
        mu : numpy.ndarray
            2-element array specifying the coordinates of the robot
        xy : numpy.ndarray
            2 x m array specifying the coordinates of the landmark

        Returns
        -------
        line_collection :  matplotlib.collections.LineCollection
                           A line collection
        """


        lines = np.array([]).reshape(0, 2)

        for j in range(xy.shape[1]):
            temp = np.array([[mu[0], mu[1]], [xy[0, j], xy[1, j]]])
            lines = np.concatenate((lines, np.array([[mu[0], mu[1]], [xy[0, j], xy[1, j]]])))

        line_collection.set_segments([lines])

        return line_collection

    def drawMap(self, M):
        """Draw the ground-truth map

            Args
            ----------
            M :     A 3 x N array where each column is of the form [id; x; y]
                    denoting the id and coordinates of an individual landmark (numpy.array)
        """

        for i in range(M.shape[1]):
            circle = self.drawCircle(M[1:3, i], self.gtColor)
            self.ax.add_artist(circle)

    def drawTrajectory(self, XYE, XYGT):
        """Draw the estimated and ground-truth trajectories.

        Attributes
        ----------
        XYE : numpy.ndarray
            A 2 x T array, where each column specifies the EKF-based estimate
            of the (x, y) position at that point in time.
        XYGT : numpy.ndarray
            A 2 x T array, where each column specifies the ground-truth
            (x, y) position at that point in time.
        """
        self.ax.plot(XYE[0, :], XYE[1, :], color=self.estColor,
                     linestyle='dashed')
        self.ax.plot(XYGT[0, :], XYGT[1, :], color=self.gtColor,
                     linestyle='dashed')

    def render(self, mu, Sigma, gt, z, mapLUT):
        """Render the current pose estimate.

        Attributes
        ----------
        mu : numpy.ndarray
            The current mean vector.
        Sigma : numpy.ndarray
            The current covariance matrix.
        gt : numpy.ndarray
            The ground-truth pose.
        z :  numpy.ndarray
            A 4 x m array of measurements, where each column is of the form [t; id; zx; zy]
        mapLUT: Dictionary mapping landmark ids to indices
        """
        if not self.estTriangle:
            self.estTriangle = self.drawTriangle(mu[0:2], mu[2], self.estColor)
            self.ax.add_patch(self.estTriangle)
        else:
            self.updateTriangle(self.estTriangle, mu[0:2], mu[2])

        if not self.gtTriangle:
            self.gtTriangle = self.drawTriangle(gt[0:2], gt[2], self.gtColor)
            self.ax.add_patch(self.gtTriangle)
        else:
            self.updateTriangle(self.gtTriangle, gt[0:2], gt[2])


        if not self.ellipse:
            self.ellipse = self.drawEllipse(mu[0:2], Sigma[0:2, 0:2])
            self.ax.add_patch(self.ellipse)
        else:
            self.updateEllipse(self.ellipse, mu[0:2], Sigma[0:2, 0:2])

        if not self.line_collection:

            xy = np.array([]).reshape(2,0)
            for j in range(z.shape[1]):
                idx = mapLUT[str(int(z[1, j]))]
                xy = np.hstack((xy, np.reshape(mu[idx:idx+2],(2,1))))

            self.line_collection = self.drawLineCollection(mu[0:2], xy, self.estColor)
            self.ax.add_collection(self.line_collection)
        else:

            xy = np.array([]).reshape(2,0)
            for j in range(z.shape[1]):
                idx = mapLUT[str(int(z[1, j]))]
                xy = np.hstack((xy, np.reshape(mu[idx:idx+2],(2,1))))

            self.updateLineCollection(self.line_collection, mu[0:2], xy)

        # Now, process the map elements
        for k, v in mapLUT.items():
            idx = v
            if k not in self.estMapCircles:
                circle = self.drawCircle(mu[idx:idx+2], self.estColor)
                self.ax.add_artist(circle)
                self.estMapCircles[k] = circle

                ellipse = self.drawEllipse(mu[idx:idx+2], Sigma[idx:idx+2, idx:idx+2])
                self.ax.add_patch(ellipse)
                self.estMapEllipses[k] = ellipse
            else:
                self.updateCircle(self.estMapCircles[k], mu[idx:idx+2])
                self.updateEllipse(self.estMapEllipses[k], mu[idx:idx+2], Sigma[idx:idx+2, idx:idx+2])

        plt.pause(0.05)

    def angleWrap(self, theta):
        """Ensure that a given angle is in the interval (-pi, pi)."""
        with np.nditer(theta, op_flags=['readwrite']) as it:
            for x in it:
                while x < -np.pi:
                    x[...] = x[...] + 2*np.pi

                while x > np.pi:
                    x[...] = x[...] - 2*np.pi

        return theta

    def plotError(self, XYE, XYGT, Variance, numSigma=3):
        """Plot the estimation error and standard deviations

        Attributes
        ----------
        XYE : numpy.ndarray
            A 3 x T array, where each column specifies the EKF-based estimate
            of the (x, y) position at that point in time.
        XYGT : numpy.ndarray
            A 3 x T array, where each column specifies the ground-truth
            (x, y) position at that point in time.
        Variance : numpy.ndarray
            A 3 x T array, where each column specifies the variance of
            x, y, and theta at that point in time.

        If the
        """

        Error = XYGT - XYE
        Error[2, :] = self.angleWrap(Error[2, :])

        fig, axs = plt.subplots(3)
        axs[0].plot(Error[0, :])
        axs[0].plot(-numSigma * np.sqrt(Variance[0, :]), 'r--')
        axs[0].plot(numSigma * np.sqrt(Variance[0, :]), 'r--')
        axs[0].set(xlabel='Time', ylabel='X Error')

        axs[1].plot(Error[1, :])
        axs[1].plot(-numSigma * np.sqrt(Variance[1, :]), 'r--')
        axs[1].plot(numSigma * np.sqrt(Variance[1, :]), 'r--')
        axs[1].set(xlabel='Time', ylabel='Y Error')

        axs[2].plot(Error[2, :] * 180/np.pi)
        axs[2].plot(-numSigma * np.sqrt(Variance[2, :]) * 180/np.pi, 'r--')
        axs[2].plot(numSigma * np.sqrt(Variance[2, :]) * 180/np.pi, 'r--')
        axs[2].set(xlabel='Time', ylabel='Theta Error (Degrees)')
