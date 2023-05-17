import sys
import argparse
import math
from numpy import *
import matplotlib.pyplot as plt
from GridWorldMDP import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Simple grid world MDP")
    parser.add_argument('--gamma', help='Discount factor', type=float, 
                        required=True)
    parser.add_argument('--noise', help='Transition noise likelihood', 
                        type=float, required=True)
    parser.add_argument('--epsilon', 
                        help='Value iteration convergence threshold',
                        type=float, required=True)

    args = parser.parse_args()

    mdp = GridWorldMDP(args.noise, args.gamma)

    V, Pi, n = mdp.valueIteration(args.epsilon)

    print(f"Converged in {n} iterations")

    mdp.drawWorld(V, Pi)
