import numpy as np


class Gridmap(object):
    """A class for gridmap representations of maps

        Attributes
        ----------
        occupancy :   An occupancy grid-based representation of the environment (numpy.array)
        xres, yres :  The resolutions of the map in the x- and y-directions

        Methods
        -------
        inCollision : Checks whether a particular pair of Cartesian coordinates or indices
                      is in collision based on the gridmap
        getShape :    Get the shape of the gridmap
        ij2xy :       Converts index pair (i, j) to Cartesian coordinates (x, y) for gridmap
        xy2ij :       Converts Cartesian coordinates (x, y) to index pair for gridmap
    """

    def __init__(self, occupancy, xres=1, yres=1):
        """Initialize the class

            Args
            ----------
            occupancy :   An occupancy grid-based representation of the environment (numpy.array)
            xres, yres :  The resolutions of the map in the x- and y-directions (optional, default: 1)

        """
        # Flip the occupancy grid so that indexing starts in the lower-left
        self.occupancy = occupancy[ ::-1,:].astype(np.bool)
        self.xres = xres
        self.yres = yres
        self.m = self.occupancy.shape[0]
        self.n = self.occupancy.shape[1]

    def inCollision(self, x, y, ij=False):
        """Checks whether a particular (x, y) coordinate is in collision according to the map

            Args
            ----------
            x, y :  Coordinates to check (may be indices, see below)
            ij :    Boolean indicating whether (x, y) are array indices (optional, default: False)

            Returns
            ----------
            collision: Boolean indicating whether coordinate is occupied in map
        """
        if ij == True:
            j = x
            i = y
        else:
            # j = int(np.ceil(x/self.xres))
            # i = int(np.ceil(y/self.yres))
            j = np.int32(np.floor(x/self.xres))
            i = np.int32(np.floor(y/self.yres))
            #i = (self.m-1) - int(np.floor(y/self.yres)) # Since i=0 is upper-left
        i = np.asarray(i)
        j = np.asarray(j)

        inBounds = (i < self.m) * (i >= 0) * (j < self.n) * (j >= 0)
        collision = np.ones(i.shape, dtype=np.bool)
        collision[inBounds] = self.occupancy[i[inBounds], j[inBounds]]
        return collision

    # Returns the height and width of the occupancy
    # grid in terms of the number of cells
    #
    # Returns:
    #   m: Height in number of cells
    #   n: Width in number of cells
    def getShape(self):
        return self.m, self.n

    # Converts an (i,j) integer pair to an (x,y) pair
    #   i:   Row index (zero at bottom)
    #   j:   Column index
    #
    # Returns
    #   x:   x position
    #   y:   y position
    def ij2xy(self, i, j):
        """Converts an (i,j) integer pair to an (x,y) pair

            Args
            ----------
            i, j:  Row (i, zero at bottom) and column (j) indices

            Returns
            ----------
            x, y:  Cartesian coordinates corresponding to indices (i, j)
        """
        x = np.float(j * self.xres)
        y = np.float(i * self.yres)

        return x, y


    # Converts an (i,j) integer pair to an (x,y) pair
    #   x:   x position
    #   y:   y position
    #
    # Returns:
    #   i:   Row index (zero at bottom)
    #   j:   Column index
    def xy2ij(self, x, y):
        """Converts an (x, y) integer pair to an (i, j) pair

            Args
            ----------
            x, y:  Cartesian coordinates

            Returns
            ----------
            i, j:  Row and column indices corresponding to Cartesian coordinates (x, y)
        """
        i = int(np.floor(y/self.yres))
        j = int(np.floor(x/self.yres))

        return i, j
