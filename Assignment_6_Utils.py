# In this file, we store various useful functions needed by the internal
# force calculation in Assignment 6.


import numpy as np
from Assignment_5 import posAndJac, scaling, realN


def numElements(m, n=0, p=0):
    """This function finds the number of elements
    The inputs are
    'm' - the number of elements in the x-direction
    'n' - the number of elements in the y-direction (optional)
    'p' - the number of elements in the z-direction (optional)
    The output is 'num', the number of elements
    """
    sets = [m, n, p]  # the collection
    num = m  # There will be at least 'm' elements
    for i in range(2):
        if sets[i+1] == 0:
            sets[i+1] = 1
        num *= sets[i+1]

    return num


def numDims(m, n=0, p=0):
    """This function returns the number of problem dimensions
    The inputs are
    'm' - the number of elements in the x-direction
    'n' - the number of elements in the y-direction (optional)
    'p' - the number of elements in the z-direction (optional)
    """
    sets = [m, n, p]  # defines the collection
    numD = 0  # the number of dimensions in the problem
    for i in range(3):
        if sets[i] > 0:
            numD += 1
    return numD


def getElemDefs(enum, deform, ien):
    """Here, we define a function that gets the 3D nodal deformations for a
    specific element and returns a list for all elemental nodes.

    The inputs are:
    'enum' - the element number (0, 1, ...)
    'deform' - the global deformation array (not missing dof's)
    'ien' - the ien array

    The outputs are:
    'defE' - a list of 3D deformations for all nodes in an element 'enum'
    """
    dimMap = {2: 1, 4: 2, 8: 3}  # maps number of element nodes to dimensions
    numD = dimMap[int(len(ien[0]))]  # number of problem dimensions

    # Here, we store all the deformations in the elemental node array format
    defE = [[0.0, 0.0, 0.0] for i in range(len(ien[0]))]

    for i in range(len(ien[enum])):  # for every node in the element...
        for j in range(3):  # for every dimension at the node...
            if j < numD:  # if a deformation was found...
                defE[i][j] = deform[int(numD*ien[enum][i]) + j]

    return defE


def getXaArray(e, nodes, ien):
    """This function gets the nodal positions for all the nodes in a particular
    element 'e'

    The inputs are:
    'e' - element number
    'nodes' - the list of problem nodes
    'ien' - ien array for the problem

    Returns a vector of the 3D coordinates of the element nodes of size
    [# element nodes]x[3]
    """
    nodeNum = ien[e]  # get the node locations
    xa = []  # contains the desired nodal coordinates
    for i in range(len(nodeNum)):
        xa.append(nodes[int(nodeNum[i])])
    return xa


def getBandScale(numD, basis, intpt, xa):
    """This function gets the 'B' matrix for the given number of dimensions.
    The inputs are:
    'numD' - the number of dimensions of the main (problem) domain 'sigma'
    'basis' - the complete set of basis function evaluations at all element
              integration points
    'intpt' - the particular integration (Gauss) point in question
    'a' - the basis function index (0, 1, 2...) at the element level (not
          global A)
    'xa' - a vector of the 3D coordinates of the element nodes of size
          [# element nodes]x[3]. Can be the deformed coordinates 'ya'.

    The outputs are:
    'Bmats' - An array of all the B-matrices for every element basis function
          in the order of the element nodes. All derivatives are in the real
          domain.
    """
    # get element global position and jacobian
    x, jac = posAndJac(basis[0][intpt], xa)
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


def getStiff(n, cCons=0, es=0):
    """This function returns the appropriate 'D' matrix for a material. This
    first iteration assumes an isotropic material

    The inputs are:
    'n' - the number of problem dimensions
    'cCons' - an array of the parameters needed to define the constituitive
              law that contains ['Young's Modulus', 'Poisson's Ratio']
              Not used if given 'n' or '0'
    'es' - if no argument, assume 'D' for engineering

    The output is the 'D' matrix in 1D, 3D or 6D.
    """
    E = 200.0*10**9    # modulus of elasticity (Pa)
    v = 0.3         # Poisson's ratio

    if (cCons != 0) and (cCons != 'n') and (cCons != 'empty'):
        E = cCons[0]
        v = cCons[1]

    ld = E*v/((1 + v)*(1 - 2*v))  # Lame parameters
    mu = E/(2*(1 + v))

    c = 2
    if es != 0:  # if 'D' will not be used with engineering strain...
        c = 1

    aa = E/(1-v**2)  # 2D stiffness parameters
    bb = aa*(1 - v)*c/2  # doubled since its not engineering strain (green)
    cc = aa*v

    if n == 1:
        D = E
    if n == 2:
        D = [[aa, cc, 0],
             [cc, aa, 0],
             [0,   0, bb]]
    if n == 3:
        D = [[ld + 2*mu,    ld,             ld,             0,      0,      0],
             [ld,           ld + 2*mu,      ld,             0,      0,      0],
             [ld,           ld,             ld + 2*mu,      0,      0,      0],
             [0,            0,              0,              c*mu,   0,      0],
             [0,            0,              0,              0,      c*mu,   0],
             [0,            0,              0,              0,      0,   c*mu]]

    return D


def getCijkl(i, j, k, m, F, C):
    """Here, we find any elasticity constant pushed forward to the current
    frame. The assumption of this method is that the material is isotropic.

    The inputs are:
    'i, j, k, l' - indices (0, 1, 2) of the desired tensor entry (current)
    'F' - the deformation gradient
    'ld, mu' - Lame parameters

    The outputs are:
    'c_ijkl' - the desired ijkl-th entry of the elasticity tensor
    """
    Cijkl = 0.0  # initialize the result
    Jac = np.linalg.det(F)  # determinant of the deformation gradient

    for I in range(3):  # for first capital index...
        for J in range(3):  # for the second...
            for K in range(3):  # for the third...
                for M in range(3):  # for the fourth...
                    diff = (1.0/Jac)*F[i][I]*F[j][J]*F[k][K]*F[m][M]*C
                    Cijkl += diff
    return Cijkl


def get3x3F(F):
    """In this function, we expand 'F' to fill a 3x3 array. The additional
    entries, if needed are taken from an identity matrix.

    The inputs are:
    'F' - the deformation gradient

    The outputs are:
    'F' - a 3x3 deformation gradient
    """
    newF = np.eye(3)

    for i in range(len(F)):  # for every row in 'F'...
        for j in range(len(F[0])):  # for every column in 'F'...
            newF[i][j] = F[i][j]

    return newF


def getEulerStiff(F, n, cCons, es):
    """In this function, we get the elasticity tensor in the current (Eulerian)
    frame.

    The inputs are:
    'F' - the derivatives of the current position 'y' wrt the undeformed 'x'
    'n' - the number of problem dimensions (1, 2 or 3)
    'cCons' - an array of the parameters needed to define the constituitive
              law that contains ['Young's Modulus', 'Poisson's Ratio']
              Not used if given 'n' or '0'
    'es' - if no argument, assume 'D' for Green strain (not engineering)

    The outputs are:
    'dd' - the elasticity tensor in the current (Eulerian) frame
    """
    D = getStiff(n, cCons, es)
    newF = get3x3F(F)  # get full-size 'F'

    if n == 1:  # make the matrix 6x6
        ym = D
        D = [[0.0 for i in range(6)] for j in range(6)]
        D[0][0] = ym

    if n == 2:  # make the matrix 6x6
        for i in range(3):
            D[i].extend([0, 0, 0])
            D.append([0, 0, 0, 0, 0, 0])

    p = [[0, 1, 2, 1, 0, 0],
         [0, 1, 2, 2, 2, 1]]
    d = [[0.0 for i in range(6)] for j in range(6)]  # 'D' pushed forward

    for i in range(6):  # for every row of 'D'...
        for j in range(6):  # for every column of 'D'...
            d[i][j] = getCijkl(p[0][i], p[1][i], p[0][j],
                               p[1][j], newF, D[i][j])

    if n == 1:
        dd = d[0][0]
    elif n == 2:
        dd = [d[i][0:3] for i in range(3)]
    elif n == 3:
        dd = d

    return dd
