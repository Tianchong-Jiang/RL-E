import sys
import os
import numpy as np
import pickle
import argparse

from PF import PF
from Laser import Laser
from Gridmap import Gridmap


if __name__ == '__main__':

    # This function should be called with two arguments:
    #    sys.argv[1]: Pickle file defining problem setup
    #    sys.argv[2]: Number of particles (default=100)
    # if (len(sys.argv) == 3):
    #     numParticles = int(sys.argv[2])
    # elif (len(sys.argv) == 2):
    #     numParticles = 100
    # else:
    #     print "usage: RunPF.py Data.pickle numParticles (optional, default=100)"
    #     sys.exit(2)
    parser = argparse.ArgumentParser(description="Particle Filter-based Localization")
    parser.add_argument('pickle_file', help='Path to pickle file')
    parser.add_argument('-n', '--numParticles', type=int, default=1000, help='Number of particles')
    parser.add_argument('--sparsity', type=int, default=5, help='Factor for downsampling LIDAR')
    parser.add_argument('-a', '--animate', action='store_true', help="Generate an animation of the filter")
    args = parser.parse_args()

    # Load data
    numParticles = args.numParticles
    Data = pickle.load (open (args.pickle_file, 'rb'))
    deltat = Data['deltat']
    occupancy = Data['occupancy']
    U = Data['U']
    X0 = Data['X0']
    Ranges = Data['Ranges']
    XGT = Data['XGT']
    Alpha = Data['Alpha']

    numBearings = Ranges.shape[0]
    Ranges = Ranges[::args.sparsity, :]

    # Gridmap class
    gridmap = Gridmap(occupancy)

    # Laser class
    laser = Laser(numBearings, args.sparsity)

    # Instantiate the PF class
    pf = PF(numParticles, Alpha, laser, gridmap, args.animate)

    filename = os.path.basename(sys.argv[1]).split('.')[0] + '_' + str(numParticles) + '_particles'

    pf.run(U, Ranges, deltat, X0, XGT, filename)
