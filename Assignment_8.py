# In this file, we create the code necessary to initialize and solve a complete finite-element
# problem. Multiple functions and the stiffness matrix will be needed.


import numpy as np
from Assignment_4 import getBasis
from Assignment_6 import getStiff, getBandScale, getXaArray


####################################################################

# Constructing the stiffness matrix requires knowledge of the 'ID' array mapping the 
# problem degrees of freedom onto the available degrees of freedom.

# The inputs are:
# 'cons' - the constraint array indexed by [dim #][node #] used as 'id' array

# The output is:
# 'id' - an array implementing the map (Global Eqn. #) = ID[(# dims)(Node #) + (DOF #)]
# 'ncons' - the number of constraints applied

def getIDArray(cons):
	ida = []	 # initialize the array
	count = 0  # counter needed to assign equation map
	
	for i in range(len(cons[0])):  # for every global node...
		for j in range(len(cons)):	# for every problem dimension...
			if cons[j][i] == 'n':  # if no constraint...
				ida.append(count)
				count += 1	# increment the counter
			else:
				ida.append('n')	# indicates dof shouldn't be included
	
	ncons = len(cons)*len(cons[0]) - count  # number of constraints
	
	return ida, ncons

####################################################################

# Before the stiffness matrix can be calculated, we need a function that returns the
# energy density at a particular integration point (Bt D B)

# The inputs are:
# 'dims' - number of problem dimensions (1, 2, 3)
# 'intpt' - the specific integration point number (0, 1, ...)
# 'xa' - a vector of the 3D coordinates of the element nodes of size
#		[# element nodes]x[3]
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

# Test function
def getotherstiff(a, b, c):
	return a + b + c

# Before we can successfully calculate the stiffness matrix, we need to implement
# an integral of the energy density function in this file. This integral takes
# place over all the basis functions in the element and the return is the element
# stiffness matrix.

# The inputs are:
# 'dims' - the number of problem dimensions
# 'xa' - the 3D real coordinates of the element nodes arranged in a list

# The outputs are:
# 'ke' - the element stiffness matrix of dimensions
#		[(# dims)x(# element basis funcs.)] x [(# dims)x(# element basis funcs.)]

def gaussIntKMat(dims, xa):
	basis = getBasis(dims)
	numA = 2**dims
	D = getStiff(dims)	# the 'D' matrix
	w = 1  # the gauss point integral weight (2 pts)
	ke = np.array([[0.0 for i in range(dims*numA)] for j in range(dims*numA)])
	
	for i in range(len(basis[0])):	# for every integration point...
		# now, get the 'Bmat' for the integration point
		Bmats, scale = getBandScale(dims, basis, i, xa)

		for j in range(numA):  # for the 'a'-th basis function...
			for k in range(numA):  # for the 'b'-th basis function...
				kab = getEnergyDensity(D, Bmats[j], Bmats[k])
				
				# then, we assemble 'kab' into the appropriate slot in 'ke'
				for m in range(len(kab)):  # for every row...
					for n in range(len(kab[0])):  # for every column...
						ke[j*dims + m][k*dims + n] += kab[m][n]*scale*w
	return ke

############################################################################################

# The essential task of this function is to assemble the element stiffness
# matrices into a global stiffness matrix, excluding the constrained degrees
# of freedom.

# The inputs are:
# 'nodes' - a list of the 3D locations of all the problem nodes
# 'ien' - the ien array for the problem implementing the map
#		  [global eqn. #] = ien[elem. #][local basis func. #]
# 'cons' - the constraint array indexed by [dim #][node #] used as 'id' array

# The outputs are:
# 'kmat' - the global stiffness matrix of size
#		[(# dims)x(# global basis funcs.)] x [(# dims)x(# global basis funcs.)]

def getStiffMatrix(nodes, ien, cons):
	dims = len(cons)  # 'cons' should always contain the problem dimensionality
	numA = 2**dims	# the number of element basis functions
	ida, ncons = getIDArray(cons)
	totA = len(nodes)*dims - ncons	# the number of stiffness matrix equations
	# the return global stiffness matrix
	kmat = np.array([[0.0 for i in range(totA)] for j in range(totA)])
	
	for i in range(len(ien)):  # for every element...
		xa = getXaArray(i, nodes, ien)	# get the element nodal locations (global)
		ke = gaussIntKMat(dims, xa)

		for j in range(len(ke)):  # for every row in 'ke'...
			for k in range(len(ke[0])):	 # for every column in 'ke'...
				P = int(ien[i][j/dims])	 # the global node row number
				Q = int(ien[i][k/dims])	 # the global node column number
				pp = j%dims	 # the dof num. for the row number
				qq = j%dims	 # the dof num. for the column number
				
				if (cons[pp][P] == 'n') and (cons[qq][Q] == 'n'):
					kmat[ida[P*dims + pp]][ida[Q*dims + qq]] += ke[i][j]
	
	return kmat

###################################################################


