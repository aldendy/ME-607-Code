# In this file, we implement the various functions needed to make our existing
# code non-linear capable. The needed function alternatives are implemented here.

# We begin by reformulating everthing previous to the stress calculator in
# Assignment 6.


import numpy as np
from Assignment_5 import realN
from Assignment_6_Utils import getEulerStiff, getElemDefs


############################################################################

# First, we write a function that calculates the 'F' matrix (u_{i,j}) at a
# particular integration point.

# The inputs are:
# 'defE' - deformation at all element nodes in format of global 'u' array
# 'pts' - array containing the function and derivative evaluations at a specific
#         integration point indexed by [basis function #]
#         [0 - function, 1 - df/dxi, 2 - df/deta ...]
# 'jac' - the jacobian dx_i/dxi_j

# The outputs are:
# 'F' - the derivatives of the current position 'y' wrt the undeformed 'x'

def getF(defE, pts, jac):
    numD = int(len(pts[0]) - 1)  # number of problem dimensions
    numA = int(len(pts))  # number of basis functions
    F = [[0.0 for i in range(numD)] for j in range(numD)]  # initialize 'F'
    
    for i in range(numA):  # for every basis function...
        dNdX = realN(pts, i, jac)  # get derivatives in the reference config
        for j in range(numD):  # for every real (y) derivative...
            for k in range(numD):  # for every dof in the deformation...
                F[k][j] += dNdX[j]*defE[i][k]

    for i in range(numD):  # Add the identity matrix
        F[i][i] += 1
    
    return F

###########################################################################

# Here, we implement a function that takes a second-order tensor and returns
# the same tensor in Voigt notation.

# The inputs are:
# 'tensor' - an arbitrary tensor of size [3x3], [2x2] or [1x1]

# The outputs are:
# 'vt' - the voigt notation version of the tensor

def getVoigt(tensor):
    if len(tensor) == 1:
        vt = [[tensor[0][0]]]
    if len(tensor) == 2:
        vt = [[tensor[0][0]], [tensor[1][1]], [tensor[0][1]]]
    if len(tensor) == 3:
        vt = [[tensor[0][0]], [tensor[1][1]], [tensor[2][2]],
              [tensor[1][2]], [tensor[0][2]], [tensor[0][1]]]
    return np.array(vt)

###########################################################################

# In this function, we convert from Voigt to standard, square notation

# The inputs are:
# 'tensor' - an arbitrary tensor of size [3x3] or [2x2]

# The outputs are:
# 'vt' - the voigt notation version of the tensor

def getSquareFromVoigt(tensor):
    if len(tensor) == 1:
        sSq = [[tensor[0][0]]]
    if len(tensor) == 3:
        sSq = [[tensor[0][0], tensor[2][0]], [tensor[2][0], tensor[1][0]]]
    if len(tensor) == 6:
        sSq = [[tensor[0][0], tensor[5][0], tensor[4][0]],
              [tensor[5][0], tensor[1][0], tensor[3][0]],
              [tensor[4][0], tensor[3][0], tensor[2][0]]]
    return np.array(sSq)

###########################################################################

# In this function, we get the Green strain at a particular location.

# The inputs are:
# 'defE' - deformation at all element nodes in format of global 'u' array
# 'pts' - array containing the function and derivative evaluations at a specific
#         integration point indexed by [basis function #]
#         [0 - function, 1 - df/dxi, 2 - df/deta ...]
# 'jac' - the jacobian dx_i/dxi_j (not dy/d xi)

# The outputs are:
# 'GSv' - the green strain for the specific location in Voigt notation

def getGstrain(defE, pts, jac):
    numD = int(len(pts[0]) - 1)  # number of problem dimensions
    F = np.array(getF(defE, pts, jac))
    GS = 0.5*(np.dot(np.transpose(F), F) - np.identity(numD))  # Green Strain
    GSv = getVoigt(GS)
    return GSv, F

###########################################################################

# Next, we implement the calculation of the Second Piola-Kirkhoff stress

# The inputs are:
# 'defE' - deformation at all element nodes in format of global 'u' array
# 'pts' - array containing the function and derivative evaluations at a specific
#         integration point indexed by [basis function #]
#         [0 - function, 1 - df/dxi, 2 - df/deta ...]
# 'jac' - the jacobian dx_i/dxi_j
# 'cCons' - an array of the parameters needed to define the constituitive law
#           that contains ['Young's Modulus', 'Poisson's Ratio']

# The outputs are:
# 'S' - the second Piola-Kirkhoff stress (Voigt Notation)

def getPK2(defE, pts, jac, cCons=0):
    numD = int(len(pts[0]) - 1)  # number of problem dimensions
    GSv, F = getGstrain(defE, pts, jac)  # gets Green strain in Voigt notation
    
    if cCons != 0:
        D = np.array(getEulerStiff(F, numD, cCons))
    else:
        D = np.array(getEulerStiff(F, numD))
    
    # Saint Venant-Kirchhoff model
    S = np.dot(D, GSv)

    return S

###########################################################################

# After getting the second Piola-Kirchhoff stress, we can then get the cauchy
# stress.

# The inputs are:
# 'defE' - deformation at all element nodes in format of global 'u' array
# 'pts' - array containing the function and derivative evaluations at a specific
#         integration point indexed by [basis function #]
#         [0 - function, 1 - df/dxi, 2 - df/deta ...]
# 'jac' - the jacobian dx_i/dxi_j (NOT dy/dxi - the current frame jacobian)
# 'cCons' - an array of the parameters needed to define the constituitive law
#           that contains ['Young's Modulus', 'Poisson's Ratio']

# The outputs are:
# 'sigma' - the cauchy stress in Voigt notation

def getCauchy(defE, pts, jac, cCons=0):
    if cCons != 0:
        S = getPK2(defE, pts, jac, cCons)
    else:
        S = getPK2(defE, pts, jac)
    SSq = getSquareFromVoigt(S)
    F = np.array(getF(defE, pts, jac))
    sigmaSq = np.dot(F, np.dot(SSq, np.transpose(F)))/np.linalg.det(F)
    sigmaV = getVoigt(sigmaSq)
    return sigmaV

###########################################################################



