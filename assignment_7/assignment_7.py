# In this file, we implement the routines necessary to calculate the external
# force vector for a tensor-product mesh

import numpy as np
from assignment_1.assignment_1 import *
from assignment_2.assignment_2 import *
from assignment_5.assignment_5 import *
from assignment_6.assignment_6 import getXaArray

###############################################

# Here, we create a function that represents the integrand 'N_a f_i det J',
# which represents the effects of the body forces

# The inputs are:
# 'data' - the basis function evaluations at a specific integration point indexed
#           by [basis func. #][0 - value, 1, 2,... - dN/dxi_i]
# 'j' - the degree of freedom in question (j = 0, 1, 2)
# 'fj' - the j-th component of the force (f_j, p n_j, h_j) at the specific
#        integration pt.
# 'scale' - the integral scaling factor needed to change to the parent domain
# 'w' - the integration point weight

# The ouputs are:
# 'faj' - a vector of size [(# dimensions)x(# element basis funcs)][1]

def integrand(data, j, fj, scale, w):
    numD = len(data[0]) - 1  # the number of problem dimensions
    numB = len(data)  # the number of basis functions
    faj = numD*numB*[0]  # the return value is initialized as a column vector
    
    for a in range(len(data)):  # for every basis function...
        faj[numD*a + j] = data[a][0]*fj*scale*w
    
    return faj

#############################################################

# Here, we write a function that takes 'integrand' and integrates over the
# appropriate domain.

# The inputs are:
# 'data' - the basis function evaluations for an element region indexed by
#           [int. pt. #][basis func. #][0 - value, 1, 2,... - dN/dxi_i]
# 's' - the element region (s - 0, interior, s - 1, 2, 3... boundaries)
# 'j' - the degree of freedom number (j = 0, 1, 2)
# 'fj' - the j-th component of the force (f_j, p n_j, h_j) at the specific
#        integration pt.
# 'xa0' - the real coordinates of the element nodes [node #][3]

# The outputs are:
# 'area' - an array of size [(# dimensions)x(# element basis funcs)][1]

def gaussInt(data, s, j, fj, xa0):
    numD = len(data[0][0]) - 1  # the number of problem dimensions
    numB = len(data[0])  # the number of basis functions
    area = np.array(numD*numB*[0.0])  # initialize the particular integral
    w = 1  # the Gauss-point integration weight (2 points)
    
    for i in range(len(data)):  # for every integration point...
        x, jac = posAndJac(data[i], xa0)
        scale = scaling(s, jac)
        
        area += np.array(integrand(data[i], j, fj, scale, w))
    
    return area

#######################################################

# This function performs the correct integrations and assembles the external force vector

# The inputs are:
# 'loads' - an array containing either traction vectors, scalar pressures or chars indicating no load
#           indexed by [element region (0 - interior, 1,2.. bounds)][element #]
# 'basis' - the appropriate basis for the problem dimensionality (contains the basis function evals)
#           indexed by [element region][int. pt. #][basis func. #][0 - value, 1, 2,... - dN/dxi_i]
# 'nodes' - the list of problem nodes
# 'ien' - ien array for the problem coding mapping (Global Eqn. #) = ien[Element #][Local Eqn. # 'a']
# 'ida' - an array mapping (Eqn. #) = ID[Eqn. # including restrained dof's]
# 'ncons' - the number of dof constraints

# The outputs are:
# 'forceVec' - a vector of the external forces for every dof of every node

def getExtForceVec(loads, basis, nodes, ien, ida, ncons):
    numD = {3:1, 5:2, 7:3}  # array mapping (# dimensions) = array[# element regions]
    dims = numD[len(loads)]  # the number of dimensions
    numA = len(nodes)  # the number of global function 'A' that must be filled.
    forceVec = np.array((dims*numA - ncons)*[0.0])  # initializes the force vector array to appropriate size
    
    for i in range(len(loads[0])):  # for every element...
        xai = getXaArray(i, nodes, ien)  # get the coordinates of the element nodes
        
        for j in range(len(loads)):  # for every element region...
            
            integral = np.array(dims*len(basis[j][0])*[0.0])
            
            for k in range(dims):  # for every degree of freedom...
                dtype = type(loads[j][i]).__name__  # the data type
                
                if dtype == 'list':  # if the load is a traction...
                    integral = gaussInt(basis[j], j, k, loads[j][i][k], xai)
                    
                elif dtype == 'float' or dtype == 'int':  # if the load is a pressure...
                    xi, jaci = posAndJac(basis[j][0], xai)
                    normal = boundNormal(j, jaci)
                    p_vec = (loads[j][i])*np.array(normal)
                    integral = gaussInt(basis[j], j, k, p_vec[k], xai)
                    
                # Next, we assembly the proper components
                for m in range(len(integral)):  # for every item in the integral...
                    a = int(m/dims)  # local number of the element basis function
                    jj = int(m%dims)  # the degree of freedom for the element of 'integral'
                    bigA = int(ien[i][a])  # global equation number

                    if ida[bigA*dims + jj] != 'n':  # if there is an available dof...
                        forceVec[ida[bigA*dims + jj]] += integral[m]
                    integral[m] = 0
                
    return forceVec

###########################################################################################


                    

