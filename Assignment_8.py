# In this file, we create the code necessary to initialize and solve a complete finite-element
# problem. Multiple functions and the stiffness matrix will be needed.


import numpy as np
from Assignment_4 import getBasis
from Assignment_6 import getStiff, getBandScale


####################################################################

# Before the stiffness matrix can be calculated, we need a function that returns the
# energy density at a particular integration point (Bt D B)

# The inputs are:
# 'dims' - number of problem dimensions (1, 2, 3)
# 'intpt' - the specific integration point number (0, 1, ...)
# 'xa' - a vector of the 3D coordinates of the element nodes of size
#       [# element nodes]x[3]
# 'a' - the first elemental matrix subcoordinate (A in the notes) (0, 1, ...)
# 'b' - the second elemental matrix subcoordinate (B in the notes) (0, 1, ...)

# The outputs are:

def getEnergyDensity(dims, intpt, xa, a, b):
    D = getStiff(dims)  # the 'D' matrix
    basis = getBasis(dims)
    # now, get the 'Bmat' for the integration point
    Bmats = getBandScale(dims, basis, intpt, xa)

    # Now, get the matrix product
    k_ab = np.dot(np.transpose(Bmats[a]), np.dot(np.array(D), Bmats[b]))

    return k_ab

##########################################################################

# One essential function that is needed will return the stiffness matrix 'K' for the system.

# The inputs are:

# The outputs are:


def getKmatrix():

    return 0

###################################################################
