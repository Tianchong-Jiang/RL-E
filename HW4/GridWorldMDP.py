import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.path import Path
import matplotlib.patches as patches


class GridWorldMDP(object):
    """A class for implementing a basic MDP solver for a simple gridworld

        20  21  22  23  24\n
        15   x  17  18  19\n
        10   x  12   x  14\n
         5   6   7   8   9\n
         0   1   2   3   4\n

        Attributes
        ----------
        T :      A 24x24x4 array where T[i,j,k] is the likelihood
                 of transitioning from state i to state j when taking
                 action A[k]
        R :      A 24x24x4 array where R[i,j,k] expresses the
                 reward received when going from state i to state j
                 via action A[k]
        A :      A list of actions A = [N=0, E=1, S=2, W=3]
        noise :  The likelihood that the action is incorrect
        gamma :  The discount factor

        Methods
        -------
        drawWorld :      Render the environment, value function, and policy
        valueIteration : Performs value iteration

    """

    def __init__(self, noise=0.2, gamma=0.9):
        """Initialize the class

            Args
            ----------
            noise : The likelihood that the action is incorrect
            gamma : The discount factor

        """

        self.gamma = gamma

        # The actions
        self.A = ['N', 'E', 'S', 'W']

        self.width = 5
        self.height = 5
        self.numstates = self.width*self.height + 1  # +1 for the extra 
                                                     # absorbing state
        self.absorbing_states = [0, 1, 2, 3, 4, 12, 14]
        obstacles = [11, 13, 16]



        # The transition matrix
        self.T = np.zeros([self.numstates, self.numstates, 4])



        for i in self.absorbing_states:
            self.T[i, 25, 0] = 1
            self.T[i, 25, 1] = 1
            self.T[i, 25, 2] = 1
            self.T[i, 25, 3] = 1


        for i in obstacles:
            self.T[i, i, 0] = 1
            self.T[i, i, 1] = 1
            self.T[i, i, 2] = 1
            self.T[i, i, 3] = 1

        for a in range(0,4):
            self.T[25, 25, a] = 1.0

        # Nominally set the transition likelihoods
        for i in range(0, self.width*self.height):

            # We've already taken care of the absorbing and obstacle states
            if i in self.absorbing_states:
                continue

            if i in obstacles:
                continue

            # Are we bounded above, below, left, or right by a
            # boundary or an obstacle
            btop = False
            bbottom = False
            bleft = False
            bright = False

            if (i >= (self.width*(self.height-1))) or \
                (i+self.width in obstacles):
                btop = True

            if (i < self.width) or (i-self.width in obstacles):
                bbottom = True

            if ((i+1) % 5 == 0) or (i+1 in obstacles):
                bright = True

            if (i % 5 == 0) or (i-1 in obstacles):
                bleft = True

            # North
            a = 0

            if btop:
                self.T[i, i, a] = 1 - noise
            else:
                self.T[i, i+self.width, a] = 1 - noise

            if bleft:
                self.T[i, i, a] += noise/2
            else:
                self.T[i, i-1, a] = noise/2

            if bright:
                self.T[i, i, a] += noise/2
            else:
                self.T[i, i+1, a] = noise/2

            # East
            a = 1

            if bright:
                self.T[i, i, a] = 1 - noise
            else:
                self.T[i, i+1, a] = 1 - noise

            if btop:
                self.T[i, i, a] += noise/2
            else:
                self.T[i, i+self.width, a] = noise/2

            if bbottom:
                self.T[i, i, a] += noise/2
            else:
                self.T[i, i-self.width, a] = noise/2

            # South
            a = 2

            if bbottom:
                self.T[i, i, a] = 1 - noise
            else:
                self.T[i, i-self.width, a] = 1 - noise

            if bleft:
                self.T[i, i, a] += noise/2
            else:
                self.T[i, i-1, a] = noise/2

            if bright:
                self.T[i, i, a] += noise/2
            else:
                self.T[i, i+1, a] = noise/2

            # West
            a = 3

            if bleft:
                self.T[i, i, a] = 1 - noise
            else:
                self.T[i, i-1, a] = 1 - noise

            if btop:
                self.T[i, i, a] += noise/2
            else:
                self.T[i, i+self.width, a] = noise/2

            if bbottom:
                self.T[i, i, a] += noise/2
            else:
                self.T[i, i-self.width, a] = noise/2

        # The rewards
        self.R = np.zeros([self.numstates, self.numstates, 4])

        # Rewards are received when taking any action in the absorbing state
        for i in range(0,5):
            for a in range(0,4):
                self.R[i, 25, a] = -10.0

        for a in range(0, 4):
            self.R[12, 25, a] = 1.0
            self.R[14, 25, a] = 10.0

    def drawWorld(self, V, Pi):
        """Visualizes the MDP

            Args
            ----------
            V  : The value function
            Pi : The policy

        """

        fig = pyplot.figure()
        ax = fig.add_subplot(111)

        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            labelbottom=False,
            labelleft=False)

        size = 10
        for i in range(self.height+1):
            ax.plot([0, self.width*size], [i*size, i*size], 'k-')

        for i in range(self.width+1):
            ax.plot([i*size, i*size], [0, self.height*size], 'k-')

        # Draw the obstacles
        verts = [
            (1.*size, 2.*size),
            (1.*size, 4.*size),
            (2.*size, 4.*size),
            (2.*size, 2.*size),
            (1.*size, 2.*size),
        ]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY]

        path = Path(verts, codes)

        patch = patches.PathPatch(path, facecolor='black', lw=1)
        ax.add_patch(patch)


        verts = [
            (1.*size, 3.*size),
            (1.*size, 4.*size),
            (2.*size, 4.*size),
            (2.*size, 3.*size),
            (1.*size, 3.*size),
        ]

        path = Path(verts, codes)

        patch = patches.PathPatch(path, facecolor='black', lw=1)
        ax.add_patch(patch)


        verts = [
            (3.*size, 1.*size),
            (3.*size, 2.*size),
            (4.*size, 2.*size),
            (4.*size, 1.*size),
            (3.*size, 1.*size),
        ]

        path = Path(verts, codes)


        for i in range(0,5):
            verts = [
                (i*size, 0.*size),
                (i*size, 1.*size),
                ((i+1)*size, 1.*size),
                ((i+1)*size, 0.*size),
                (i*size, 0.*size),
            ]

            path = Path(verts, codes)

            patch = patches.PathPatch(path, facecolor='red', lw=1)
            ax.add_patch(patch)


        verts = [
            (3.*size, 2.*size),
            (3.*size, 3.*size),
            (4.*size, 3.*size),
            (4.*size, 2.*size),
            (3.*size, 2.*size),
        ]

        path = Path(verts, codes)

        patch = patches.PathPatch(path, facecolor='black', lw=2)
        ax.add_patch(patch)

        # Draw the goal regions
        verts = [
            (2.*size, 2.*size),
            (2.*size, 3.*size),
            (3.*size, 3.*size),
            (3.*size, 2.*size),
            (2.*size, 2.*size),
        ]

        path = Path(verts, codes)


        patch = patches.PathPatch(path, facecolor='green', lw=2)
        ax.add_patch(patch)


        verts = [
            (4.*size, 2.*size),
            (4.*size, 3.*size),
            (5.*size, 3.*size),
            (5.*size, 2.*size),
            (4.*size, 2.*size),
        ]

        path = Path(verts, codes)

        patch = patches.PathPatch(path, facecolor='green', lw=2)
        ax.add_patch(patch)

        # Draw the value function (-1 because of the auxiliary state)
        for k in range(0,len(V)-1):
            j = np.floor(k/self.width)
            i = k - j*self.width

            v = '%.2f' % V[k]

            ax.text((i+0.4)*size, (j+0.45)*size, v)

            if Pi[k] == 0:
                ax.arrow ((i+0.5)*size, (j+0.6)*size, 0.0, 0.3*size, 
                          head_width=0.1*size, head_length=0.1*size, fc='k', ec='k')

            if Pi[k] == 1:
                ax.arrow ((i+0.7)*size, (j+0.5)*size, 0.15*size, 0.0, 
                          head_width=0.1*size, head_length=0.1*size, fc='k', ec='k')

            if Pi[k] == 2:
                ax.arrow ((i+0.5)*size, (j+0.4)*size, 0.0, -0.3*size, 
                          head_width=0.1*size, head_length=0.1*size, fc='k', ec='k')

            if Pi[k] == 3:
                ax.arrow ((i+0.3)*size, (j+0.5)*size, -0.15*size, 0.0,
                           head_width=0.1*size, head_length=0.1*size, fc='k', ec='k')


        pyplot.show()

    def valueIteration(self, epsilon):
        """Perform value iteration

            Args
            ----------
            epsilon  : The threshold for the stopping criterion 
                       (|Vnew - Vprev|_inf <= epsilon)

            Returns
            -------
            V  :  The value of each state encoded as a 12x1 array
            Pi :  The action associated with each state (i.e., the policy)
                  encoded as a 12x1 array, where |x|_inf is the infinity norm
                  (i.e., max(abs(V[i])) over all i)

        """

        # Your function should populate the following arrays
        V = np.zeros([self.numstates])  # Value function
        Pi = np.zeros([self.numstates]) # Policy where Pi[i] is 0 (N), 1 (E), 2 (S), 3(W)

        n = 0  # Keep track of the number of iterations

        # INSERT YOUR CODE HERE (DON'T FORGET TO INCREMENT THE NUMBER OF ITERATIONS)

        return (V, Pi, n)
