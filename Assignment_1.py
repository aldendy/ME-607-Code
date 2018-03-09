import numpy as np


# In this program, we begin to implement some of the functions needed for a multi-dimensional finite-
# element code.

###########################################################################

# This function takes parameters describing the essential characteristics of a tensor-product mesh
# and generates the corresponding 1D, 2D or 3D mesh of nodal positions. 

# The inputs are:
# 'M' - the scalar width in the x-direction
# 'N' - the scalar width in the y-direction (optional)
# 'P' - the scalar width in the z-direction (optional)
# 'm' - the number of elements in the x-direction
# 'n' - the number of elements in the y-direction (optional)
# 'p' - the number of elements in the z-direction (optional)

# The function returns and array of the 3D coordinates of all the nodes

def nodeList(M, N, P, m, n=0, p=0):  # dimensions other than 'x' default to zero dimensions
	nodes = []  # an array of size [num. of nodes]x[num. dimensions]
	stepi = 1.0*M/m
	
	if (n > 0) and (p == 0):
		stepj = 1.0*N/n
		nodes = np.zeros([(m+1)*(n+1), 3])  # initialize the nodal array
		
		for j in range(n+1):
			for i in range(m+1):
				nodes[j*(m+1) + i] = [i*stepi, j*stepj, 0]
				
	if (n > 0) and (p > 0):
		stepj = 1.0*N/n
		stepk = 1.0*P/p
		nodes = np.zeros([(m+1)*(n+1)*(p+1), 3])  # initialize the nodal array
		
		for k in range(p+1):
			for j in range(n+1):
				for i in range(m+1):
					nodes[k*(m+1)*(n+1) + j*(m+1) + i] = [i*stepi, j*stepj, k*stepk]
					
	if (n == 0) and (p > 0):
		stepk = 1.0*P/p
		nodes = np.zeros([(m+1)*(p+1), 3])  # initialize the nodal array
		
		for k in range(p+1):
			for i in range(m+1):
				nodes[k*(m+1) + i] = [i*stepi, 0, k*stepk]
	
	if (n == 0) and (p == 0):
		stepi = 1.0*M/m
		nodes = np.zeros([(m+1), 3])  # initialize the nodal array
		
		for i in range(m+1):
			nodes[i] = [i*stepi, 0, 0]
	
	return nodes

##################################################################################
	
# In this function, we assemble the connectivity array. The inputs to this function are the number of elements in 
# each dimension of the tensor mesh. Only the number of elements in the first dimension is obligatory.

# 'm' - the number of elements in the x-direction
# 'n' - the number of elements in the y-direction (optional)
# 'p' - the number of elements in the z-direction (optional)

# This function returns and array
# 'global node #' = ien('e = element #', 'a = local node number')

def get_ien(m, n=0, p=0):
	size = 2  # the number of nodes per edge
	
	if (n == 0) and (p == 0):
		ien = np.zeros([m, size])  # the connectivity array 
		
		for e in range(m):
			for i in range(size):
				ien[e][i] = e*(size-1) + i
			
	if (n > 0) and (p == 0):
		ien = np.zeros([m*n, size**2])  # the connectivity array with two edges worth of nodes
		
		for e in range(m*n):
			ill = e%m 
			jll = e/m 
			
			for j in range(size):
				jc = jll + j 
				for i in range(size):
					ic = ill + i 
					A = jc*(m+1) + ic 
					a = j*size + i 
					ien[e][a] = A
	
	if (n == 0) and (p > 0):
		ien = np.zeros([m*p, size**2])  # the connectivity array with two edges worth of nodes
		
		for e in range(m*p):
			ill = e%m 
			kll = e/m 
			
			for k in range(size):
				kc = kll + k 
				for i in range(size):
					ic = ill + i 
					A = kc*(m+1) + ic 
					a = k*size + i 
					ien[e][a] = A
					
	if (n > 0) and (p > 0):
		ien = np.zeros([m*n*p, size**3])  # the connectivity array with three edges worth of nodes
		
		for e in range(m*n*p):
			ill = (e%(m*n))%m 
			jll = (e%(m*n))/m 
			kll = e/(m*n)
			
			for k in range(size):
				kc = kll + k 
				for j in range(size):
					jc = jll + j 
					for i in range(size):
						ic = ill + i 
						A = kc*(m+1)*(n+1) + jc*(m+1) + ic 
						a = k*size**2 + j*size + i 
						ien[e][a] = A
		
	return ien 

#########################################################################

# Constructing the stiffness matrix requires knowledge of the 'ID' array mapping the 
# problem degrees of freedom onto the available degrees of freedom.

# The inputs are:
# 'cons' - the constraint array indexed by [dim #][node #] used as 'id' array

# The output is:
# 'id' - an array coding map (Global Eqn. #) = ID[(# dims)(Node #) + (DOF #)]
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



