# Useful functions (* denotes files you should expect to modify)

EKFSLAM.py:

    *augmentState: Add new landmarks to state by updating the
                   mean vector and covariance matrix

    *prediction: Implements the EKF prediction step

    *update: Performs the EKF update step

    *run:  The EKF "main" function


Visualization.py:

    drawEstimates: Plots the mean estimates of the robot and map poses
                   along with ellipses that visualize uncertainty as level sets
