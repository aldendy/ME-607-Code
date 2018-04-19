# In this file, we create the code necessary to initialize and solve a
# complete finite-element problem. Multiple functions and the stiffness
# matrix will be needed.


import numpy as np
from Assignment_4 import getBasis
from Assignment_5 import posAndJac, realN
from Assignment_6 import  getYa, intForceVec
from Assignment_6_Utils import getBandScale, getElemDefs, getEulerStiff
from Assignment_7 import getExtForceVec
from Assignment_10 import getF, getSquareFromVoigt, getCauchy


####################################################################

# Before the stiffness matrix can be calculated, we need a function that
# returns the energy density at a particular integration point (Bt D B)

# The inputs are:
# 'pts' - the integration point array containing the function and
#       derivative evaluations at a point indexed by [basis function #]
#       [0 - function, 1 - df/dxi, 2 - df/deta ...] - a component of the
#       'basis' for the element
# 'ya' - a vector of the 3D coordinates of the element nodes of size
#       [# element nodes]x[3]\
# 'D' - the elasticity tensor for a specific number of problem dims.
# 'sigmaSq' - the square Cauchy stress tensor
# 'a' - the first elemental eq. num. (A in the notes)(0, 1, ...)
# 'b' - the second elemental eq. num. (B in the notes)(0, 1, ...)
# 'Ba' - the B-matrix associated with the eq. num. 'a'
# 'Bb' - the B-matrix associated with the eq. num. 'b'

# The outputs are:
# 'k_ab' - elemental stiffness matrix [# prob. dims]x[# prob. dims]

def getEnergyDensity(pts, ya, D, sigmaSq, a, b, Ba, Bb):
    dims = int(len(pts[0]) - 1)  # number of problem dimensions
    
    # Now, get the material stiffness
    k_abM = np.dot(np.transpose(Ba), np.dot(np.array(D), Bb))
    
    y, jacYxi = posAndJac(pts, ya)
    gradNa = realN(pts, a, jacYxi)  # gradient of 'Na' in 'y'
    gradNb = realN(pts, b, jacYxi)  # gradient of 'Nb' in 'y'

    # Here, get the geometric stiffness
    k_abG = np.dot(gradNa, np.dot(sigmaSq, np.transpose(gradNb)))

    diff = k_abG/(4.7e10)
    #if (a == b) and (abs(diff) > 0.0001):
    #    print(diff)
        
    k_ab = (k_abM + k_abG*np.eye(dims))  # combine both
    
    if type(k_ab).__name__ != 'ndarray':  # if it is a scalar...
        k_ab = [[k_ab]]

    return k_ab

##########################################################################

# Before we can successfully calculate the stiffness matrix, we need to
# implement an integral of the energy density function in this file. This
# integral takes place over all the basis functions in the element and the
# return is the element stiffness matrix.

# The inputs are:
# 'dims' - the number of problem dimensions
# 'ya' - current 3D coordinates of the element nodes arranged in a list
# 'defE' - deformation at all element nodes in format of global 'u' array
# 'cCons' - an array of the parameters needed to define the constituitive
#           law that contains ['Young's Modulus', 'Poisson's Ratio']

# The outputs are:
# 'ke' - the element stiffness matrix of dimensions
#       [(# dims)x(# element basis funcs.)] x [(# dims)x(# element basis
#       funcs.)]

def gaussIntKMat(dims, ya, defE, cCons=0):
    basis = getBasis(dims)
    numA = 2**dims  # number of element basis functions
    xa = np.array(ya) - np.array(defE)  # reference 3D elemental node positions
    w = 1  # the gauss point integral weight (2 pts)
    ke = np.array([[0.0 for i in range(dims*numA)] for j in range(dims*numA)])
    
    for i in range(len(basis[0])):  # for every integration point...
        x, jacXxi = posAndJac(basis[0][i], xa)
        F = getF(defE, basis[0][i], jacXxi)
        
        if cCons != 0:
            D = getEulerStiff(F, dims, cCons)  # the 'D' matrix
            sigma = getCauchy(defE, basis[0][i], jacXxi, cCons)
        else:
            D = getEulerStiff(F, dims)
            sigma = getCauchy(defE, basis[0][i], jacXxi)
        
        sigmaSq = getSquareFromVoigt(sigma)  # get square Cauchy
	
	# now, get the 'Bmat' for the integration point
	Bmats, scaleYxi = getBandScale(dims, basis, i, ya)
	
	for j in range(numA):  # for the 'a'-th basis function...
	    for k in range(numA):  # for the 'b'-th basis function...
		kab = getEnergyDensity(basis[0][i], ya, D, sigmaSq, j, k,
                                       Bmats[j], Bmats[k])
		
		# then, we assemble 'kab' into the appropriate slot in 'ke'
		for m in range(len(kab)):  # for every row...
		    for n in range(len(kab[0])):  # for every column...
			ke[j*dims + m][k*dims + n] += kab[m][n]*scaleYxi*w
    return ke

############################################################################################

# The essential task of this function is to assemble the element stiffness
# matrices into a global stiffness matrix, excluding the constrained
# degrees of freedom.

# The inputs are:
# 'nodes' - a list of the 3D locations of all the problem nodes
# 'ien' - the ien array for the problem implementing the map
#         [global eqn. #] = ien[elem. #][local basis func. #]
# 'deform' - a vector of the displacements (not positions) of all nodes
#           with size [# nodes x 3]x[1]
# 'ida' - an array mapping (Eqn. #) = ID[Eqn. # including restrained dof's]
# 'ncons' - the number of dof constraints
# 'cCons' - an array of the parameters needed to define the constituitive
#           law that contains ['Young's Modulus', 'Poisson's Ratio']

# The outputs are:
# 'kmat' - the global stiffness matrix of size
#       [(# dims)x(# global basis funcs.)] x [(# dims)x(# global basis
#       funcs.)]

def getStiffMatrix(nodes, ien, deform, ida, ncons, cCons=0):
    dims = len(ida)/len(nodes)  # 'ida' has (# dims) as many
    numA = 2**dims  # the number of element basis functions
    totA = len(nodes)*dims - ncons  # number of stiffness matrix equations
    # the return global stiffness matrix
    kmat = np.array([[0.0 for i in range(totA)] for j in range(totA)])

    for i in range(len(ien)):  # for every element...
        # get the element nodal locations (global)
	ya = getYa(i, nodes, deform, ien)
        defE = getElemDefs(i, deform, ien)
        
	if cCons != 0:  # if we get a parameter definition...
	    ke = gaussIntKMat(dims, ya, defE, cCons)
	else:
	    ke = gaussIntKMat(dims, ya, defE)
	
	for j in range(len(ke)):  # for every row in 'ke'...
	    for k in range(len(ke[0])):  # for every column in 'ke'...
		P = int(ien[i][j/dims])  # the global node row number
		Q = int(ien[i][k/dims])  # the global node column number
		pp = j%dims  # the dof num. for the row number
		qq = k%dims  # the dof num. for the column number
		
		if (ida[P*dims + pp] != 'n') and (ida[Q*dims + qq] != 'n'):
		    kmat[ida[P*dims + pp]][ida[Q*dims + qq]] += ke[j][k]
    return kmat

#############################################################################################

# This function updates the 'd' vector using the partial 'd' vector found
# in the Newton-Raphson method.

# The inputs are:
# 'ida' - the id array mapping (unconstrained eqn. #) = ID[Global Eqn. #]
# 'deform' - the deformation array missing constrained degrees of freedom
# 'cons' - the constraint vector indexed by [dimension #][node #]

# The outputs are:
# 'deform0' - the deformation array with the appropriate constrained dof's
#               added

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

# After getting the stiffness matrix, the next task is to implement the
# Newton-Raphson method

# The inputs are:
# 'numD' - the number of problem dimensions
# 'loads' - an array containing either traction vectors, scalar pressures
#           or chars indicating no load indexed by
#           [element region (0 - interior, 1,2.. bounds)][element #]
# 'nodes' - the list of the 3D coordinates of the problem nodes
# 'ien' - ien array for the problem coding mapping (Global Eqn. #) =
#           ien[Element #][Local Eqn. # 'a']
# 'ida' - an array mapping (Eqn. #) = ID[Eqn. # including restrained dof's]
# 'ncons' - the number of dof constraints
# 'cons' - the constraint vector indexed by [dimension #][node #]
# 'cCons' - array of the parameters needed to define the constituitive law
#           that contains ['Young's Modulus', 'Poisson's Ratio']

# The outputs are:
# 'dFinal' - final deformation vector giving the nodal deformations in all
#           dof

def solver(numD, loads, nodes, ien, ida, ncons, cons, cCons=0):
    basis = getBasis(numD)
    imax = 10  # the maximum number of iterations tolerable

    # deformation array missing dof's
    deform = np.array((numD*len(nodes) - ncons)*[0.0])
    deform0 =  getFullDVec(ida, deform, cons) # complete deformation array
    print('')
    i = 0  # starting iteration
    
    while i < imax:
        extFV = getExtForceVec(loads, basis, nodes, deform0, ien, ida,
                               ncons)
	if cCons != 0:
            stiff = getStiffMatrix(nodes, ien, deform0, ida, ncons, cCons)
	    intFV = intForceVec(nodes, ien, ida, ncons, numD,
		    len(ien), deform0, cCons)
	else:
	    stiff = getStiffMatrix(nodes, ien, deform0, ida, ncons)
	    intFV = intForceVec(nodes, ien, ida, ncons, numD,
		    len(ien), deform0)
	
	residual = np.array(extFV) - np.array(intFV)

	#print(deform0[9:12])
	msg = 'res {0:1.2E} intFV {1:1.2E} extFV {2:1.2E}'
	print(msg.format(np.linalg.norm(residual), np.linalg.norm(intFV),
                         np.linalg.norm(extFV)))
        print(stiff[0][0])
	# if the error is small...
	if abs(np.linalg.norm(residual)) < 10.0**(-7):
	    deform0 = getFullDVec(ida, deform, cons)
	    return deform0, i
	
	Kinv = np.linalg.inv(np.array(stiff))
	b = np.transpose(np.array(residual))
	du = np.dot(Kinv, b)
	
	deform += du
	deform0 = getFullDVec(ida, deform, cons)
	i += 1

    return deform0, i


###############################################################################################




