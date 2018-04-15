# In this file, we store various useful functions needed by the internal
# force calculation in Assignment 6.


import numpy as np
from Assignment_5 import posAndJac, scaling, realN


###################################################################

# This function finds the number of elements
# The inputs are
# 'm' - the number of elements in the x-direction
# 'n' - the number of elements in the y-direction (optional)
# 'p' - the number of elements in the z-direction (optional)
# The output is 'num', the number of elements

def numElements(m, n=0, p=0):
    sets = [m, n, p]  # the collection
    num = m  # There will be at least 'm' elements
    for i in range(2):
        if sets[i+1] == 0:
            sets[i+1] = 1
        num *= sets[i+1]
    
    return num

####################################################################

# This function returns the number of problem dimensions
# The inputs are
# 'm' - the number of elements in the x-direction
# 'n' - the number of elements in the y-direction (optional)
# 'p' - the number of elements in the z-direction (optional)

def numDims(m, n=0, p=0):
    sets = [m, n, p]  # defines the collection
    numD = 0  # the number of dimensions in the problem
    for i in range(3):
        if sets[i] > 0:
            numD += 1
    return numD

############################################################################

# Here, we define a function that gets the 3D nodal deformations for a
# specific element and returns a list for all elemental nodes.

# The inputs are:
# 'enum' - the element number (0, 1, ...)
# 'deform' - the global deformation array (not missing dof's)
# 'ien' - the ien array

# The outputs are:
# 'defE' - a list of 3D deformations for all nodes in an element 'enum'

def getElemDefs(enum, deform, ien):
    dimMap = {2:1, 4:2, 8:3}  # maps number of element nodes to dimensions
    numD = dimMap[int(len(ien[0]))]  # number of problem dimensions

    # Here, we store all the deformations in the elemental node array format
    defE = [[0.0, 0.0, 0.0] for i in range(len(ien[0]))]
    
    for i in range(len(ien[enum])):  # for every node in the element...
        for j in range(3):  # for every dimension at the node...
            if j < numD:  # if a deformation was found...
                defE[i][j] = deform[int(numD*ien[enum][i]) + j]
    
    return defE

#########################################################################

# This function gets the nodal positions for all the nodes in a particular
# element 'e'

# The inputs are:

# 'e' - element number
# 'nodes' - the list of problem nodes
# 'ien' - ien array for the problem

# Returns a vector of the 3D coordinates of the element nodes of size
# [# element nodes]x[3]

def getXaArray(e, nodes, ien):
    nodeNum = ien[e]  # get the node locations
    xa = []  # contains the desired nodal coordinates
    for i in range(len(nodeNum)):
        xa.append(nodes[int(nodeNum[i])])
    return xa

##########################################################################

# This function gets the 'B' matrix for the given number of dimensions.
# The inputs are:
# 'numD' - the number of dimensions of the main (problem) domain 'sigma'
# 'basis' - the complete set of basis function evaluations at all element
#           integration points
# 'intpt' - the particular integration (Gauss) point in question
# 'a' - the basis function index (0, 1, 2...) at the element level (not
#       global A)
# 'xa' - a vector of the 3D coordinates of the element nodes of size
#       [# element nodes]x[3]. Can be the deformed coordinates 'ya'.

# The outputs are:
# 'Bmats' - An array of all the B-matrices for every element basis function
#       in the order of the element nodes. All derivatives are in the real
#       domain.

def getBandScale(numD, basis, intpt, xa):
    x, jac = posAndJac(basis[0][intpt], xa)  # get element global position and jacobian
    scale = scaling(0, jac)  # get the element interior integration scaling
    
    Bmats = []  # initialize the array
    for i in range(len(basis[0][intpt])):  # for every basis function...
        dNdxi = realN(basis[0][intpt], i, jac)
        if numD == 1:
            Bmat = dNdxi[0]  # derivative in 'x'
        if numD == 2:
            n1 = dNdxi[0]
            n2 = dNdxi[1]
            # as in notes
            Bmat = np.array([[n1, 0], [0, n2], [n2, n1]])  
        if numD == 3:
            n1 = dNdxi[0]
            n2 = dNdxi[1]
            n3 = dNdxi[2]
            Bmat = np.array([[n1, 0, 0], [0, n2, 0], [0, 0, n3], [0, n3, n2],
                         [n3, 0, n1], [n2, n1, 0]])
        Bmats.append(Bmat)
    
    return Bmats, scale

######################################################################

# This function returns the appropriate 'D' matrix for a material. This
# first iteration assumes an isotropic material

# The inputs are:
# 'n' - the number of problem dimensions
# 'cCons' - an array of the parameters needed to define the constituitive
#           law that contains ['Young's Modulus', 'Poisson's Ratio']

# The output is the 'D' matrix in 1D, 3D or 6D.

def getStiff(n, cCons=0):
    E = 200.0*10**9    # modulus of elasticity (Pa)
    v = 0.3         # Poisson's ratio
    
    if cCons != 0:
        E = cCons[0]
        v = cCons[1]
    
    ld = E*v/((1 + v)*(1 - 2*v))  # Lame parameters
    mu = E/(2*(1 + v))

    aa = E/(1-v**2)  # 2D stiffness parameters
    bb = aa*(1 - v)  # multiplied by 2 since its not engineering strain (green)
    cc = aa*v
    
    if n == 1:
        D = E
    if n == 2:
        D = [[aa, cc, 0],
             [cc, aa, 0],
             [0,   0, bb]]
    if n == 3:
        D = [[ld + 2*mu,    ld,             ld,             0,      0,      0],
            [ ld,           ld + 2*mu,      ld,             0,      0,      0],
            [ ld,           ld,             ld + 2*mu,      0,      0,      0],
            [ 0,            0,              0,              2*mu,   0,      0],
            [ 0,            0,              0,              0,      2*mu,   0],
            [ 0,            0,              0,              0,      0,   2*mu]]
    
    return D

################################################################################

# In this function, we get the elasticity tensor in the current (Eulerian)
# frame.

# The inputs are:
# 'F' - the derivatives of the current position 'y' wrt the undeformed 'x'
# 'n' - the number of problem dimensions (1, 2 or 3)
# 'cCons' - an array of the parameters needed to define the constituitive law
#           that contains ['Young's Modulus', 'Poisson's Ratio']

# The outputs are:
# 'D_y' - the elasticity tensor in the current (Eulerian) frame

def getEulerStiff(F, n, cCons=0):
    E = 200.0*10**9    # modulus of elasticity (Pa)
    v = 0.3         # Poisson's ratio
    
    if cCons != 0:
        E = cCons[0]
        v = cCons[1]
    
    ld0 = E*v/((1 + v)*(1 - 2*v))  # Lame parameters in reference
    mu0 = E/(2*(1 + v))

    J = np.linalg.det(F)
    d1111 = (1/J)*F[0][0]**4*(ld0 + 2*mu0)  # convert first constant
    
    if len(F) == 1:  # for 1D...
        d1122 = ld0
    else:
        d1122 = (1/J)*F[0][0]**2*F[1][1]**2*(ld0)

    ld = ld0  #d1122  # get the Lame parameters in the current configuration
    mu = mu0  #0.5*(d1111 - d1122)
    
    if n == 1:
        D = [[mu*(3*ld + 2*mu)/(ld + mu)]]
    if n == 2:
        aa = ld + 2*mu - ld**2/(ld + 2*mu)  # 2D stiffness parameters
        bb = 2*mu
        cc = ld - ld**2/(ld + 2*mu)
    
        D = [[aa, cc, 0],
             [cc, aa, 0],
             [0,   0, bb]]
    if n == 3:
        D = [[ld + 2*mu,    ld,             ld,             0,      0,      0],
            [ ld,           ld + 2*mu,      ld,             0,      0,      0],
            [ ld,           ld,             ld + 2*mu,      0,      0,      0],
            [ 0,            0,              0,              2*mu,   0,      0],
            [ 0,            0,              0,              0,      2*mu,   0],
            [ 0,            0,              0,              0,      0,   2*mu]]
    
    return D

###########################################################################





