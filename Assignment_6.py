# In this file, we develop a routine for calculating the internal
# force vector for the Galerkin approximation of the weak form. 


import numpy as np
from Assignment_1 import nodeList, get_ien, getIDArray
from Assignment_4 import *
from Assignment_5 import posAndJac, scaling, realN
from Assignment_6_Utils import *
from Assignment_10 import getCauchy


#################################################################################

# This function gets the strain at a given integration point. 
# The inputs are:
# 'numD' - the number of problem dimensions (1, 2, or 3)
# 'e' - the element number (0, 1, 2...)
# 'deform' - the global deformation array indexed by [c_ax, c_ay, c_(a+1)x, ...]
# 'ien' - the global ien array indexed by [element #][3D coordinates]
# 'Bmats' - a collection of 'B' matrices for all 'a'
#           (of length = # basis funcs)

# The output is:
# 'strain' - the strain vector (Voigt notation) of size
#           [1 - 1D, 3 - 2D, 6 - 3D]x[1]

def strainVec(numD, e, deform, ien, Bmats):
    sDim = [1, 3, 6]  # the number of rows in the stress or strain vector
    strain = np.array(sDim[numD-1]*[[0.0]])  # initialize the strain vector
  
    for k in range(len(Bmats)):  # for every element basis function...
        first = int(numD*ien[e][k])  # location of c_kx in global 'deform'
        second = first + numD  # location of c_ky (2D) in global 'deform'
        cmat = np.transpose(np.array([deform[first:second]]))
    
        strain += np.dot(np.array(Bmats[k]), cmat)
  
    return strain 

########################################################################

# This function returns the stress vector in Voigt notation 

# The inputs are:
# 'numD' - the number of problem dimensions
# 'strain' - the strain vector in Voigt notation
# 'cCons' - an array of the parameters needed to define the constituitive
#           law that contains ['Young's Modulus', 'Poisson's Ratio']

# The output is the stress vector in Voigt notation 

def stressVec(numD, strain, cCons=0):
    if cCons != 0:
        return np.dot(np.array(getStiff(numD, cCons)), strain)
    else:
        return np.dot(np.array(getStiff(numD)), strain)

################################################################################

# This function calculates the internal force for a particular 'k' and 'm' 

# The inputs are:
# 'deform' - the array of all deformation cooeficients for the entire mesh
#            [c11, c12, c21, c22...] for 2D where c is indexed by
#            [global node #][degree of freedom]
# 'basis' - the basis function evaluations at the integration points
# 'ien' - the ien array for the global mesh
# 'e' - the element number
# 'i' - the particular integration point number (0, 1, 2...)
# 'ya' - a vector of the current vector positions of the element nodes
# 'cCons' - an array of the parameters needed to define the constituitive
#           law that contains ['Young's Modulus', 'Poisson's Ratio']

# The output is the nodal evaluation of the force vector element at an
# integration point 

def func(deform, basis, ien, e, i, ya, cCons=0):
    numD = len(basis[0][0][0]) - 1  # the number of dimensions
    Bmats, scaleYxi = getBandScale(numD, basis, i, ya)
    defE = getElemDefs(e, deform, ien)
    xa = np.array(ya) - np.array(defE)
    
    y, jacYxi = posAndJac(basis[0][i], ya)
    x, jacXxi = posAndJac(basis[0][i], xa)
    
    # strain = strainVec(numD, e, deform, ien, Bmats)
    if cCons != 0:
        stress = getCauchy(defE, basis[0][i], jacXxi, cCons)
        #stress = stressVec(numD, strain, cCons)
    else:
        stress = getCauchy(defE, basis[0][i], jacXxi)
        #stress = stressVec(numD, strain)
    
    # the force vector for all nodes 'a' over element 'e'
    f = np.array(numD*len(basis[0][0])*[[0.0]])  
    
    for k in range(len(basis[0][0])):  # for every element basis function...
        ff = np.dot(np.transpose(np.array(Bmats[k])), stress)*scaleYxi
        for j in range(numD):  # for every dimension
            f[numD*k + j] = ff[j]
    
    return f

################################################################################

# Here, we implement an updated version of Gaussian integration that relies
# on the 'basis' calculated in the 'getBasis' function. This eliminates
# the need to evaluate the functional as in the previous version.

# The inputs are:
# 'func' - the function values obtained based on the 'basis' for the given
#           'sigma' domain
# 'deform' - a vector of the displacements (not positions) of all nodes
# 'basis' - the basis for the given 'sigma' domain
# 'e' - the element number
# 'ien' - the mesh ien array (global node # = ien(element #, element basis
#           func #))
# 'ya' - a vector of the current vector positions of the element nodes
# 'cCons' - an array of the parameters needed to define the constituitive
#           law that contains ['Young's Modulus', 'Poisson's Ratio']

# The output is the integral of 'func' over the domain given in 'basis'

def GaussInt(func, deform, basis, e, ien, ya, cCons=0):
    numD = len(basis[0][0][0]) - 1  # the number of dimensions
    # initialize the integral value to an array of size [(# basis funcs) x
    # (# dimensions)] x [1]
    area = np.array(numD*len(basis[0][0])*[[0.0]])   
    weights = len(basis[0])*[1]  # the weights for the integration process
    
    for i in range(len(basis[0])):  # for every integration point...
        if cCons != 0:
            area += func(deform, basis, ien, e, i, ya, cCons)*weights[i]
        else:
            area += func(deform, basis, ien, e, i, ya)*weights[i]
  
    return area

################################################################################

# This function combines the current element node locations with the
# deformation array to obtain the current locations of the element nodes.

# The inputs are:
# 'i' - the element number
# 'nodes' - the global 3D coordinates of all nodes in the mesh
# 'deform' - a vector of the displacements of all nodes with size
#           [# nodes x 3]x[1]
# 'ien' - the mesh ien array (global node # = ien(element #,
#           element basis func #))

# The outputs are:
# 'ya' - a list of current, 3D positions of the element nodes (with
#       deformation)

def getYa(i, nodes, deform, ien):
    dims = int(len(deform)/len(nodes))  # gets number of problem dimensions
    ya = getXaArray(i, nodes, ien)  # adds the initial coordinates
    
    for j in range(len(ien[0])):  # for every element node...
        for k in range(dims):  # for every result dof...
            ya[j][k] += deform[int(dims*ien[i][j]) + k]
    return ya

################################################################################

# This function iterates over every integration point in the problem and
# calculates the internal energy integrals over the 2 or 3D domain
# describing the interior of all the elements (sigma).

# The inputs are:
# 'nodes' - the global 3D coordinates of all nodes in the mesh
# 'ien' - the mesh ien array (global node # = ien(element #, element basis
#           func #))
# 'numD' - the number of degrees of freedom in 'sigma' domain
# 'numE' - the number of elements in the mesh
# 'deform' - a vector of the displacements (not positions) of all nodes
#           with size [# nodes x 3]x[1]
# 'cCons' - an array of the parameters needed to define the constituitive
#           law that contains ['Young's Modulus', 'Poisson's Ratio']

# The output of this function is the internal force vector (global) of
# size = '[# nodes x 3]x[1]

def intForceVec(nodes, ien, ida, ncons, numD, numE, deform, cCons=0):
    basis = getBasis(numD)  # get the integration point and function data
    Fint = np.array((numD*len(nodes) - ncons)*[0.0])  # initialize the internal force vector 
    
    for i in range(numE):  # for every element...
        ya = getYa(i, nodes, deform, ien)
        
        # the internal force for a basis function 'a' over element 'i'
        if cCons != 0:
            fai = GaussInt(func, deform, basis, i, ien, ya, cCons)
        else:
            fai = GaussInt(func, deform, basis, i, ien, ya)
        
        # for this particular element vector, we then assemble it into the
        # global vector 
        for j in range(len(basis[0][0])):  # for each basis function...
            for k in range(numD):  # for each dimension...
                # assemble the correct values
                if ida[numD*int(ien[i][j]) + k] != 'n':  # if there is an available dof...
                    Fint[ida[numD*int(ien[i][j]) + k]] += fai[numD*j + k][0] 
        
    return Fint 
    
#################################################################################

# Here, we define the problem, initial deformation and mesh and generate
# the internal force array

# The inputs to this funcion are:
# 'M' - the scalar width in the x-direction
# 'N' - the scalar width in the y-direction (optional)
# 'P' - the scalar width in the z-direction (optional)
# 'm' - the number of elements in the x-direction
# 'n' - the number of elements in the y-direction (optional)
# 'p' - the number of elements in the z-direction (optional)

def initialize(M, N, P, m, n=0, p=0):
    nodes = nodeList(M, N, P, m, n, p)  # get the nodes for the problem
    ien = get_ien(m, n, p)  # get the ien array
    numD = numDims(m, n, p)  # get the number of problem dimensions
    numE = numElements(m, n, p)
    deformation = numD*len(nodes)*[0]  # initialize the deformation array
                                       # of size [# nodes]x[# dimensions]
    
    Fint = intForceVec(nodes, ien, numD, numE, deformation)

##########################################################

#initialize(1, 1, 1, 3, 2, 1)
