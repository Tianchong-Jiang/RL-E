# Useful functions (* denotes files you should expect to modify)

PF.py:

    sampleParticlesUniform: Sample particles from a uniform distribution

    sampleParticlesGaussian: Sample particles from a Gaussian with a
                             specified mean position and orientation

    getMean: Returns the particle filter mean

    getNormalizedWeights: Returns an array of normalized weights

    *sampleMotion: Sample a new pose according to the motion model
                   in Table 5.3 of Probabilistic Robotics

    *resample: Resample with replacement

    *prediction: Implements the particle filter prediction step

    *update: Performs the particle filter update step

    *run:  The particle filter "main" function


Laser.py:

    __init__: Specifies the LIDAR parameters (see below for different models)

    *scanProbability: Computes the likelihood of a LIDAR scan according to
                      the algorithm defined in Table 6.1 of Probabilistic Robotics

    rayTracing: Provides a crude implementation of ray tracing


Gridmap.py:

    inCollision: Checks to see whether a particular position
                 falls in an occupied cell





# The following are parameters for different LIDAR models
# You can change the LIDAR model by updating Laser.py

# model 1: "realistic"
self.pHit = 0.95;
self.pShort = 0.02;
self.pMax = 0.02;
self.pRand = 0.01;
self.sigmaHit = 0.05;
self.lambdaShort = 1;
self.zMax = 20;
self.zMaxEps = 0.02;


# model 2: noise free
self.pHit = 1.0;
self.pShort = 0;
self.pMax = 0;
self.pRand = 0;
self.sigmaHit = 0;
self.lambdaShort = 1;
self.zMax = 20;
self.zMaxEps = 0.02;


# model 3: always short
self.pHit = 0.0;
self.pShort = 1.0;
self.pMax = 0;
self.pRand = 0;
self.sigmaHit = 0;
self.lambdaShort = 1;
self.zMax = 20;
self.zMaxEps = 0.02;


# The following models are not very realistic and are
# provided in case you want to try them out

# model 4: always max
self.pHit = 0;
self.pShort = 0;
self.pMax = 1.0;
self.pRand = 0;
self.sigmaHit = 0;
self.lambdaShort = 1;
self.zMax = 20;
self.zMaxEps = 0.02;


# model 5: always random
self.pHit = 0;
self.pShort = 0;
self.pMax = 0;
self.pRand = 1.0;
self.sigmaHit = 0;
self.lambdaShort = 1;
self.zMax = 20;
self.zMaxEps = 0.02;
