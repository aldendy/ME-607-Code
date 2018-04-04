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
#       [# element nodes]x[3]

# The outputs are:
# 'Bmats' - An array of all the B-matrices for every element basis function
#       in the order of the element nodes. All derivatives are in the real
#       domain.

def getBandScale(numD, basis, intpt, xa):
    x, jac = posAndJac(basis[0][intpt], xa)  # get the element global position and jacobian
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
    bb = aa*(1 - v)/2
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
            [ 0,            0,              0,              mu,     0,      0],
            [ 0,            0,              0,              0,      mu,     0],
            [ 0,            0,              0,              0,      0,      mu]]
    
    return D

################################################################################







