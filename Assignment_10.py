# In this file, we implement the various functions needed to make our existing
# code non-linear capable. The needed function alternatives are implemented here.

# We begin by reformulating everthing previous to the stress calculator in
# Assignment 6.


import numpy as np
from Assignment_5 import realN
from Assignment_6 import getStiff


############################################################################

# Here, we define a function that gets the nodal deformations for a specific
# element.

# The inputs are:
# 'enum' - the element number (0, 1, ...)
# 'deform' - the global deformation array (not missing dof's)
# 'ien' - the ien array

# The outputs are:
# 'defE' - the deformations of the element nodes

def getElemDefs(enum, deform, ien):
    dimMap = {2:1, 4:2, 8:3}  # maps number of element nodes to dimensions
    numD = dimMap[int(len(ien[0]))]  # number of problem dimensions
    defE = []  # stores all the element deformations in the same format
    
    for i in ien[enum]:  # for every node in the element...
        for j in range(numD):  # for every dof at the node...
            defE.append(deform[int(numD*i + j)])
    return defE

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
        for j in range(numD):  # for every real (x) derivative...
            for k in range(numD):  # for every dof in the deformation...
                F[k][j] += dNdX[j]*defE[numD*i + k]

    for i in range(numD):  # Add the identity matrix
        F[i][i] += 1
    
    return F

###########################################################################

# Here, we implement a function that takes a second-order tensor and returns
# the same tensor in Voigt notation.

# The inputs are:
# 'tensor' - an arbitrary tensor of size [3x3] or [2x2]

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
    return vt

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
    F = np.array(getF(defE, pts, jac))
    GS = 0.5*(np.dot(np.transpose(F), F) - np.identity(numD))  # Green Strain
    GSv = getVoigt(GS)
    
    if cCons != 0:
        D = np.array(getStiff(numD, cCons))
    else:
        D = np.array(getStiff(numD))
    
    # Saint Venant-Kirchhoff model
    S = np.dot(D, GSv)

    return S

###########################################################################


