# In this file, we create the code necessary to initialize and solve a complete finite-element
# problem. Multiple functions and the stiffness matrix will be needed.


import numpy as np
from Assignment_4 import getBasis
from Assignment_5 import posAndJac, realN
from Assignment_6_Utils import getEulerStiff, getBandScale, getXaArray
from Assignment_6_Utils import getElemDefs
from Assignment_6 import intForceVec
from Assignment_7 import getExtForceVec
from Assignment_10 import getCauchy, getSquareFromVoigt


####################################################################

# Before the stiffness matrix can be calculated, we need a function that returns the
# energy density at a particular integration point (Bt D B)

# The inputs are:
# 'dims' - number of problem dimensions (1, 2, 3)
# 'intpt' - the specific integration point number (0, 1, ...)
# 'xa' - a vector of the 3D coordinates of the element nodes of size
#               [# element nodes]x[3]
# 'a' - the first elemental matrix subcoordinate (A in the notes) (0, 1, ...)
# 'b' - the second elemental matrix subcoordinate (B in the notes) (0, 1, ...)

# The outputs are:

def getEnergyDensity(D, Ba, Bb):
    # Now, get the matrix product
    k_ab = np.dot(np.transpose(Ba), np.dot(np.array(D), Bb))
    
    if type(k_ab).__name__ != 'ndarray':
        k_ab = [[k_ab]]

    return k_ab

##########################################################################

# Before we can successfully calculate the stiffness matrix, we need to implement
# an integral of the energy density function in this file. This integral takes
# place over all the basis functions in the element and the return is the element
# stiffness matrix.

# The inputs are:
# 'dims' - the number of problem dimensions
# 'xa' - the 3D real coordinates of the element nodes arranged in a list
# 'defE' - a list of the 3D deformations at every elemental node
# 'cCons' - an array of the parameters needed to define the constituitive law
#           that contains ['Young's Modulus', 'Poisson's Ratio']

# The outputs are:
# 'ke' - the element stiffness matrix of dimensions
#               [(# dims)x(# element basis funcs.)] x [(# dims)x(# element basis funcs.)]

def gaussIntKMat(dims, xa, defE, cCons=0):
    basis = getBasis(dims)
    numA = 2**dims
    ya = np.array(xa) + np.array(defE)
    
    if cCons != 0:
        D = getEulerStiff(np.identity(3), dims, cCons, 'es')  # the 'D' matrix
    else:
        D = getEulerStiff(np.identity(3), dims, 'n', 'es')
    
    w = 1  # the gauss point integral weight (2 pts)
    ke = np.array([[0.0 for i in range(dims*numA)] for j in range(dims*numA)])
    
    for i in range(len(basis[0])):  # for every integration point...
        # now, get the 'Bmat' for the integration point
        Bmats, scaleYxi = getBandScale(dims, basis, i, ya)
        y, jacYxi = posAndJac(basis[0][i], ya)
        x, jacXxi = posAndJac(basis[0][i], xa)
        
        for j in range(numA):  # for the 'a'-th basis function...
            for k in range(numA):  # for the 'b'-th basis function...
                kab = getEnergyDensity(D, Bmats[j], Bmats[k])

                # Next, we get the geometric stiffness
                gradNa = realN(basis[0][i], j, jacYxi)
                gradNb = realN(basis[0][i], k, jacYxi)

                if cCons != 0:
                    sigmaV = getCauchy(defE, basis[0][i], jacXxi, 'nl', cCons)
                else:
                    sigmaV = getCauchy(defE, basis[0][i], jacXxi, 'nl')

                sigmaSq = getSquareFromVoigt(sigmaV)
                kabG = np.eye(dims)*np.dot(gradNa, np.dot(sigmaSq, gradNb))
                kab += kabG  # add the geometric stiffness
                
                # then, we assemble 'kab' into the appropriate slot in 'ke'
                for m in range(len(kab)):  # for every row...
                    for n in range(len(kab[0])):  # for every column...
                        ke[j*dims + m][k*dims + n] += kab[m][n]*scaleYxi*w
    return ke

############################################################################################

# The essential task of this function is to assemble the element stiffness
# matrices into a global stiffness matrix, excluding the constrained degrees
# of freedom.

# The inputs are:
# 'nodes' - a list of the 3D locations of all the problem nodes
# 'ien' - the ien array for the problem implementing the map
#                 [global eqn. #] = ien[elem. #][local basis func. #]
# 'ida' - an array mapping (Eqn. #) = ID[Eqn. # including restrained dof's]
# 'deform0' - deformation array with the appropriate constrained dof's added
# 'ncons' - the number of dof constraints
# 'cCons' - an array of the parameters needed to define the constituitive law
#           that contains ['Young's Modulus', 'Poisson's Ratio']

# The outputs are:
# 'kmat' - the global stiffness matrix of size
#               [(# dims)x(# global basis funcs.)] x [(# dims)x(# global basis funcs.)]

def getStiffMatrix(nodes, ien, ida, deform0, ncons, cCons=0):
    dims = len(ida)/len(nodes)  # 'ida' has (# dims) as many
    numA = 2**dims  # the number of element basis functions
    totA = len(nodes)*dims - ncons  # the number of stiffness matrix equations
    # the return global stiffness matrix
    kmat = np.array([[0.0 for i in range(totA)] for j in range(totA)])
    
    for i in range(len(ien)):  # for every element...
        xa = getXaArray(i, nodes, ien)  # get element nodal locations (global)
        defE = getElemDefs(i, deform0, ien)
        
        if cCons != 0:  # if we get a parameter definition...
            ke = gaussIntKMat(dims, xa, defE, cCons)
        else:
            ke = gaussIntKMat(dims, xa, defE)
        
        for j in range(len(ke)):  # for every row in 'ke'...
            for k in range(len(ke[0])):      # for every column in 'ke'...
                P = int(ien[i][j/dims])  # the global node row number
                Q = int(ien[i][k/dims])  # the global node column number
                pp = j%dims      # the dof num. for the row number
                qq = k%dims      # the dof num. for the column number
                
                if (ida[P*dims + pp] != 'n') and (ida[Q*dims + qq] != 'n'):
                    kmat[ida[P*dims + pp]][ida[Q*dims + qq]] += ke[j][k]
    return kmat

#############################################################################################

# This function updates the 'd' vector using the partial 'd' vector found in the Newton-
# Raphson method.

# The inputs are:
# 'ida' - the id array mapping (unconstrained eqn. #) = ID[Global Eqn. #]
# 'deform' - the deformation array missing constrained degrees of freedom
# 'cons' - the constraint vector indexed by [dimension #][node #]

# The outputs are:
# 'deform0' - the deformation array with the appropriate constrained dof's added

def getFullDVec(ida, deform, cons):
    deform0 = []  # initializes the array
    numD = len(cons)  # this should correspond to the number of dimensions
    
    for i in range(len(ida)):  # for every part of the id array...
        a = i/numD  # the equation number (node #)
        dof = i%numD  # the degree of freedom number
        
        if ida[i] != 'n':  # if 'i' maps to an unconstrained dof...
            deform0.append(deform[ida[i]])
        else:
            deform0.append(cons[dof][a])
    
    return deform0

############################################################################################

# After getting the stiffness matrix, the next task is to implement the Newton-Raphson method

# The inputs are:
# 'numD' - the number of problem dimensions
# 'loads' - an array containing either traction vectors, scalar pressures or chars indicating no load
#           indexed by [element region (0 - interior, 1,2.. bounds)][element #]
# 'nodes' - the list of the 3D coordinates of the problem nodes
# 'ien' - ien array for the problem coding mapping (Global Eqn. #) = ien[Element #][Local Eqn. # 'a']
# 'ida' - an array mapping (Eqn. #) = ID[Eqn. # including restrained dof's]
# 'ncons' - the number of dof constraints
# 'cons' - the constraint vector indexed by [dimension #][node #]
# 'cCons' - an array of the parameters needed to define the constituitive law
#           that contains ['Young's Modulus', 'Poisson's Ratio']

# The outputs are:
# 'dFinal' - final deformation vector giving the nodal deformations in all dof

def solver(numD, loads, nodes, ien, ida, ncons, cons, cCons=0):
    basis = getBasis(numD)
    imax = 20  # the maximum number of iterations tolerable
    extFV = getExtForceVec(loads, basis, nodes, ien, ida, ncons)
    deform = np.array((numD*len(nodes) - ncons)*[0.0])  # def array missing dof's
    deform0 =  getFullDVec(ida, deform, cons) # the complete deformation array
    i = 0  # starting iteration
    
    while i < imax:
        if cCons != 0:
            stiff = getStiffMatrix(nodes, ien, ida, deform0, ncons, cCons)
            intFV = intForceVec(nodes, ien, ida, ncons, numD,
                    len(ien), deform0, cCons)
        else:
            stiff = getStiffMatrix(nodes, ien, ida, deform0, ncons)
            intFV = intForceVec(nodes, ien, ida, ncons, numD,
                    len(ien), deform0)
        
        residual = np.array(extFV) - np.array(intFV)
        print(i, np.linalg.norm(residual))
        # if the error is small...
        if abs(np.linalg.norm(residual)) < 10.0**(-4):
            deform0 = getFullDVec(ida, deform, cons)
            return deform0, i
        
        Kinv = np.linalg.inv(np.array(stiff))
        b = np.transpose(np.array(residual))
        du = np.dot(Kinv, b)
        
        deform += du
        deform0 = getFullDVec(ida, deform, cons)
        i += 1
    
    return deform0, i


###############################################################################



