# In this file, we implement the functions described in Lecture 4. These functions rely on functions
# written in 'Assignment_4.txt'. 

import math, numpy


# In this first function, we find the real 3D coordinate, x, and the Jacobian matrix for a given 
# integration point with coordinates given in the parent domain. This is done for the 2D and 3D cases.

# The inputs are:
# 'pts' - the integration point array containing the function and derivative evaluations at a point indexed by 
# [basis function #][0 - function, 1 - df/dxi, 2 - df/deta ...] - a component of the 'basis' for the element
# 'xa' - an array of the real coordinates of each node (ordered like the corresponding basis funcs) with size
# [# nodes]x[3 dimensions]
# The returns are:
# 'x' - a vector position of the physical coordinate (3x1) that corresponds to the parent domain int. pt.
# 'jac' - an array of all d(x_i)/d(xi_j) for all 'i (0, 1, 2)' and 'j (0 or 0,1 or 0,1,2)'

def posAndJac(pts, xa):
	# because 'pts' could have differing numbers of basis funcs, go through a loop.
	x = [0, 0, 0]  # the physical coordinate initialized to the origin
	
	for i in range(len(pts)):  # for each basis function...
		x[0] += xa[i][0]*pts[i][0]  # add the basis func times the correct component of 'xa'
		x[1] += xa[i][1]*pts[i][0]
		x[2] += xa[i][2]*pts[i][0]
	
	# for the Jacobian...
	jac = []  # initialize the jacobian array
	for i in range(3):  # for every dimension 'i' in 'x_i'...
		jac.append([])
		for j in range(len(pts[0]) - 1):  # for every parent dimension...
			dxdxi = 0
			for k in range(len(pts)):  # for every basis function...
				dxdxi += pts[k][j+1]*xa[k][i]
			jac[i].append(dxdxi)
	
	return x, jac

###################################################################

# The second function we write takes the element region, 's', and the Jacobian and produces the appropriate
# integral scaling. 

# The inputs are:
# 's' - the element region (0 - interior, 1, 2 ... - faces or walls)
# 'jac' - the jacobian dx_i/dxi_j

def scaling(s, jac):
	scale = 0  # initialize the result value
	
	if len(jac[0]) == 1:  # if the integral is 1D...
		scale = jac[0][0]
	
	if len(jac[0]) == 2:  # if the integral is 2D...
		v1 = [jac[0][0], jac[1][0], jac[2][0]]  # dx/dxi
		v2 = [jac[0][1], jac[1][1], jac[2][1]]  # dx/deta
	
		if s == 0:  # element interior
			scale = numpy.linalg.norm(numpy.cross(v1, v2))
		if s == 1:  # left wall
			scale = abs(v2[0]**2 + v2[1]**2)**0.5
		if s == 2:  # right wall
			scale = abs(v2[0]**2 + v2[1]**2)**0.5
		if s == 3:  # bottom wall
			scale = abs(v1[0]**2 + v1[1]**2)**0.5
		if s == 4:  # top wall
			scale = abs(v1[0]**2 + v1[1]**2)**0.5
	
	if len(jac[0]) == 3:  # if the integral is 3D...
		v1 = [jac[0][0], jac[1][0], jac[2][0]]  # dx/dxi
		v2 = [jac[0][1], jac[1][1], jac[2][1]]  # dx/deta
		v3 = [jac[0][2], jac[1][2], jac[2][2]]  # dx/dzeta
		
		if s == 0:  # element interior
			scale = numpy.linalg.det(numpy.array(jac))
		if s == 1:  # negative x face
			scale = numpy.linalg.norm(numpy.cross(v3, v2))
		if s == 2:  # positive x face
			scale = numpy.linalg.norm(numpy.cross(v2, v3))
		if s == 3:  # negative y face
			scale = numpy.linalg.norm(numpy.cross(v1, v3))
		if s == 4:  # positive y face
			scale = numpy.linalg.norm(numpy.cross(v3, v1))
		if s == 5:  # negative z face
			scale = numpy.linalg.norm(numpy.cross(v2, v1))
		if s == 6:  # positive z face
			scale = numpy.linalg.norm(numpy.cross(v1, v2))

	return scale

#######################################################

# Next, we write a function that can find the face normals (3D vector). 

# The inputs are:
# 's' - the element region (1, 2 ... - faces or walls)
# 'jac' - the jacobian dx_i/dxi_j

def boundNormal(s, jac):
	norm = [0, 0, 0]  # initialize the vector normal
	
	if len(jac[0]) == 2:  # if the integral is 2D...
		v1 = [jac[0][0], jac[1][0], jac[2][0]]  # dx/dxi
		v2 = [jac[0][1], jac[1][1], jac[2][1]]  # dx/deta
		
		if s == 1:  # left wall
			v = numpy.cross([0, 0, 1], v2)
			norm = v/numpy.linalg.norm(v)
		if s == 2:  # right wall
			v = numpy.cross(v2, [0, 0, 1])
			norm = v/numpy.linalg.norm(v)
		if s == 3:  # bottom wall
			v = numpy.cross(v1, [0, 0, 1])
			norm = v/numpy.linalg.norm(v)
		if s == 4:  # top wall
			v = numpy.cross([0, 0, 1], v1)
			norm = v/numpy.linalg.norm(v)
	
	if len(jac[0]) == 3:  # if the integral is 3D...
		v1 = [jac[0][0], jac[1][0], jac[2][0]]  # dx/dxi
		v2 = [jac[0][1], jac[1][1], jac[2][1]]  # dx/deta
		v3 = [jac[0][2], jac[1][2], jac[2][2]]  # dx/dzeta
		
		if s == 1:  # negative x face
			v = numpy.cross(v3, v2)
			norm = v/numpy.linalg.norm(v)
		if s == 2:  # positive x face
			v = numpy.cross(v2, v3)
			norm = v/numpy.linalg.norm(v)
		if s == 3:  # negative y face
			v = numpy.cross(v1, v3)
			norm = v/numpy.linalg.norm(v)
		if s == 4:  # positive y face
			v = numpy.cross(v3, v1)
			norm = v/numpy.linalg.norm(v)
		if s == 5:  # negative z face
			v = numpy.cross(v2, v1)
			norm = v/numpy.linalg.norm(v)
		if s == 6:  # positive z face
			v = numpy.cross(v1, v2)
			norm = v/numpy.linalg.norm(v)
	
	return norm

#########################################################

# Finally, we implement a function that performs a change of coordinates on the derivatives of N, transferring
# it from the parent domain to the physical domain. 

# The inputs are:
# 'pts' - the integration point array containing the function and derivative evaluations at a point indexed by 
# [basis function #][0 - function, 1 - df/dxi, 2 - df/deta ...] - a component of the 'basis' for the element
# 'a' - the basis function number (0, 1, 2 ...)
# 'jac' - the jacobian dx_i/dxi_j

# The function returns a vector, 'dNa/dxi', of all the 'xi' derivatives of a-th 'N' (real-dimensional derivatives)

def realN(pts, a, jac):
	dNdxi = pts[a][1:len(pts[a])]  # a vector of derivatives of N_a
	dim = len(jac[0])  # number of problem dimensions
	
	i = 0  # current jacobian row
	while i < len(jac):  # for every row in the jacobian...
		if numpy.linalg.norm(jac[i]) < 10**(-10):
			del jac[i]  # eliminate the odd, zero rows for the 2D and 1D cases
		else:
			i += 1
	
	dNdreal = numpy.dot(dNdxi, numpy.linalg.inv(jac))
	
	return dNdreal

#########################################################

# This function tests the operation of 'posAndJac'

def routineTest():
	e = 2  # for and element 'e'...
	esize = [1, 1, 1]  # tensor side length
	edim2d = [2, 2]  # the element dimensions of the 2D tensor mesh
	edim3d = [2, 2, 2]  # element dimensions of the 3D tensor mesh
	s = 4  # the element region...
	intpt = 0  # the integration point number
	
	ien2 = get_ien(edim2d[0], edim2d[1])  # get the ien array
	ien3 = get_ien(edim3d[0], edim3d[1], edim3d[2])  # get the ien array
	nodes2 = nodeList(esize[0], esize[1], esize[2], edim2d[0], edim2d[1])
	nodes3 = nodeList(esize[0], esize[1], esize[2], edim3d[0], edim3d[1], edim3d[2])
	
	xa2 = []  # will contain the positions of all nodes in the element
	for i in range(len(ien2[e])):
		xa2.append(nodes2[int(ien2[e][i])])
	
	xa3 = []  # will contain the positions of all nodes in the element
	for i in range(len(ien3[e])):
		xa3.append(nodes3[int(ien3[e][i])])
	
	basis2d = twoDBasis()
	basis3d = threeDBasis()
	
	x2, jac2 = posAndJac(basis2d[s][intpt], xa2)
	x3, jac3 = posAndJac(basis3d[s][intpt], xa3)
	
	scale = scaling(s, jac2)
	normal = boundNormal(s, jac2)
	dNdreal2 = realN(basis2d[s][intpt], 0, jac2)
	dNdreal3 = realN(basis3d[s][intpt], 0, jac3)
	
	print(scale)
	print(normal)
	print(dNdreal2)
	print(dNdreal3)

#######################################################

#routineTest()


# The last function to implement is the element loop described in the assignment. The input is:
# 's' - the element region (0 - interior, 1, 2 ... - faces or walls)
def eLoadData(s):
	esize = [1, 1, 1]  # tensor product mesh global (total) side length
	edim = [2, 2, 2]  # the element dimensions of the tensor mesh
	eNum = 1  # the number of elements
	
	for i in range(len(edim)):  # change all 0's to 1's
		if edim[i] != 0:
			eNum *= edim[i]
	
	for i in range(eNum):  # for every element in the mesh...
		ien = get_ien(edim[0], edim[1], edim[2])  # get the ien array
		nodes = nodeList(esize[0], esize[1], esize[2], edim[0], edim[1], edim[2])
	
		xa = []  # will contain the positions of all nodes in the element
		
		for j in range(len(ien[i])):  # gather the element nodal positions
			xa.append(nodes[int(ien[i][j])])
		
		if edim[2] == 0:  # get the correct basis depending on the dimension of the problem
			basis = twoDBasis()
		else:
			basis = threeDBasis()
		
		for j in range(len(basis[s])):  # for every integration point in the region 's'...
			x, jac = posAndJac(basis[s][j], xa)
			scale = scaling(s, jac)
			normal = boundNormal(s, jac)
			dNdreal = realN(basis[s][j], 0, jac)
			print(dNdreal)
			
		
#eLoadData(1)


