Included with the problem set are 8 pickle files, each defining sequences of control inputs and LIDAR observations recorded as the robot navigated through one of 3 environments.


    U: A 2 x T array of control (velocity) inputs

    Ranges: An L x T array of LIDAR ranges (the corresponding bearings are defined in Laser.py)

    Occupancy: An m x n array defining an occupancy grid representation of the environment

    deltat: The time period (in seconds)


The following are provided for some scenarios

    XGT (optional): A 3 x T array specifying the ground-truth pose (x, y, theta)

    X0 (optional): A 3-element vector specifying the initial pose (x, y, theta)
