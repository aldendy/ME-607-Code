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

def getEnergyDensity(D, Ba, Bb):
    # Now, get the matrix product
    k_ab = np.dot(np.transpose(Ba), np.dot(np.array(D), Bb))

    return k_ab

##########################################################################

# Before we can successfully calculate the stiffness matrix, we need to implement
# an integral of the energy density function in this file.

# The inputs are:

# The outputs are:

def gaussIntKMat(func, dims, xa):
    basis = getBasis(dims)
    numA = dims**2
    D = getStiff(dims)  # the 'D' matrix
    w = 1  # the gauss point integral weight (2 pts)
    ke = np.array([[0.0 for i in range(dims*numA)] for j in range(dims*numA)])

    for i in range(len(basis[0])):  # for every integration point...
        # now, get the 'Bmat' for the integration point
        Bmats, scale = getBandScale(dims, basis, intpt, xa)

        for j in range(numA):  # for the 'a'-th basis function...
            for k in range(numA):  # for the 'b'-th basis function...
                kab = getEnergyDensity(D, Bmats[j], Bmats[k])

                # then, we assemble 'kab' into the appropriate slot in 'ke'
                for m in range(len(kab)):  # for every row...
                    for n in range(len(kab[0])):  # for every column...
                        ke[j*dims + m][k*dims + n] = kab[m][n]
    return ke

############################################################################################

# One essential function that is needed will return the stiffness matrix 'K' for the system.

# The inputs are:
# 'dims' - number of problem dimensions (1, 2, 3)

# The outputs are:


def getElemKmatrix(dims):
    # to get the elemental stiffness matrix, we need to integrate every combination of
    # elemental integrand.
    
    
    return 0

###################################################################
