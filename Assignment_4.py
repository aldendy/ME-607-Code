# In this file, we implement the 1, 2 and 3D basis functions. Each basis function returns an array
# containing evaluations of each basis function and its derivatives at every integration point.

############################################################

# This function gets the appropriate values and assembles them into the
# integration point array. These points must be ordered similarly to the nodes
# for any given element because later methods make this assumption.

# The inputs are:
# 'numD' - the number of dimensions in the element region (1, 2, 3)

# The outputs are:
# 'intpts' - an array indexed by [element location (0 - interior, 1, 2.. bounds)]
#            [integration point #] storing (coordinate (of dim. 1, 2, 3)),(weight)

def getIntPts(numD, d=1.0/3.0**0.5):
        sets = [-d, d]
        side = [-1, 1]
        
        if numD == 1:
                # an array indexed by [point #] where the data is [xi value, weight]
                return [[[[-d], 1], [[d], 1]], [[[-1], 1]], [[[1], 1]]]
        
        if numD == 2:
                intpts = [[], [], [], [], []]
                
                for i in range(2):
                        for j in range(2):
                                intpts[0].append([[sets[j], sets[i]], 1])
                for i in range(2):  # for the left and right sides...
                        for j in range(2):
                                intpts[i+1].append([[side[i], sets[j]], 1])
                for i in range(2):  # for the top and bottom sides...
                        for j in range(2):
                                intpts[i+3].append([[sets[j], side[i]], 1])
                return intpts
        
        if numD == 3:
                intpts = [[], [], [], [], [], [], []]
                # first, we initialize the interior (s = 0)
                for i in range(2):
                        for j in range(2):
                                for k in range(2):
                                        p = [[sets[k], sets[j], sets[i]], 1]
                                        intpts[0].append(p)
                
                # for the sides normal to 'x'...
                for i in range(2):
                        for j in range(2):
                                for k in range(2):
                                        p = [[side[i], sets[k], sets[j]], 1]
                                        intpts[i+1].append(p)
                # for the sides normal to 'y'...
                for i in range(2):
                        for j in range(2):
                                for k in range(2):
                                        p = [[sets[k], side[i], sets[j]], 1]
                                        intpts[i+3].append(p)
                # for the sides normal to 'z'...
                for i in range(2):
                        for j in range(2):
                                for k in range(2):
                                        p = [[sets[k], sets[j], side[i]], 1]
                                        intpts[i+5].append(p)
                
                return intpts
        

##############################################################

# First, we write functions defining the two linear basis functions and their derivatives for a given 
# dimension. The input is:
# 'xi' - independent variable
# 'a' - the basis number (0 - with negative slope, 1 - with positive slope)
def b(xi, a):
	return 0.5*(1 + (-1)**(a+1)*xi), 0.5*(-1)**(a+1)  # returns the function evaluation and its derivative

###########################################################

# This function returns evaluations of the basis functions and their derivatives in the 1D case
# over domain xi (-1, 1).

def oneDBasis(d):
	intpts = getIntPts(1, d)
	
	# Next, we intialize and fill and array with basis function evaluations at quadrature poins where the
	# array is indexed by [element location (0 - interior, 1,2,... walls)][int. point #][basis function #]
	# [0 - function, 1 - df/dxi, 2 - df/deta ...]
	basis = []  # stores the basis data at each quadrature point
	for i in range(len(intpts)):  # for each element region...
                basis.append([])
                for j in range(len(intpts[i])):  # for each integration point...
        		f1, f1d = b(intpts[i][j][0][0], 0)  # get the function evaluations and derivatives
                	f2, f2d = b(intpts[i][j][0][0], 1)
                        basis[i].append([[f1, f1d], [f2, f2d]])  # assemble
	
	return basis

#########################################################

# Here, we write functions defining the four linear basis functions and their derivatives for a given 
# dimension. The input is:
# 'xi' - independent variable
# 'eta' - the second independent variable
# The returns are the f(xi, eta), df/dxi and df/deta
def bb1(xi, eta):
	return (b(xi, 0))[0]*(b(eta, 0))[0], (b(xi, 0))[1]*(b(eta, 0))[0], (b(xi, 0))[0]*(b(eta, 0))[1]

def bb2(xi, eta):
	return (b(xi, 1))[0]*(b(eta, 0))[0], (b(xi, 1))[1]*(b(eta, 0))[0], (b(xi, 1))[0]*(b(eta, 0))[1]

def bb3(xi, eta):
	return (b(xi, 0))[0]*(b(eta, 1))[0], (b(xi, 0))[1]*(b(eta, 1))[0], (b(xi, 0))[0]*(b(eta, 1))[1]

def bb4(xi, eta):
	return (b(xi, 1))[0]*(b(eta, 1))[0], (b(xi, 1))[1]*(b(eta, 1))[0], (b(xi, 1))[0]*(b(eta, 1))[1]

#############################################################

# This function returns evaluations of the basis functions and their derivatives in the 2D case
# over domain (-1, 1) x (-1, 1) and variables 'xi' and 'eta'. The input variables are:
# 'xi' - the first independent variable
# 'eta' - the second independent variable
def twoDBasis(d):
	# A table of integration points and weights where the indexes are
	# [interior (0) or one of four surfaces, integration point #] and 
	# the data stores is a [point coordinate in 'xi' and 'eta', weight]
	intpts = getIntPts(2, d)
	
	# Next, we intialize and fill and array with basis function evaluations at quadrature poins where the
	# array is indexed by [element location (0 - interior, 1,2,... walls)][int. point #][basis function #]
	# [0 - function, 1 - df/dxi, 2 - df/deta ...]
	basis = []  # stores the basis data at each quadrature point
	for i in range(len(intpts)):  # for each element region s...
		basis.append([])  # add a slot for the element region
		for j in range(len(intpts[i])):  # for each quadrature point
			f1, f1d, f1dd = bb1(intpts[i][j][0][0], intpts[i][j][0][1]) 
			f2, f2d, f2dd = bb2(intpts[i][j][0][0], intpts[i][j][0][1])
			f3, f3d, f3dd = bb3(intpts[i][j][0][0], intpts[i][j][0][1])
			f4, f4d, f4dd = bb4(intpts[i][j][0][0], intpts[i][j][0][1])
			
			# Now, assemble the data
			basis[i].append([[f1, f1d, f1dd], [f2, f2d, f2dd], [f3, f3d, f3dd], [f4, f4d, f4dd]])  
	
	return basis

############################################################################

# Here, we write functions defining the eight linear basis functions and their derivatives for 3D. The input is:
# 'xi' - independent variable
# 'eta' - the second independent variable
# 'zeta' - the third independent variable
# 'a' - the number of the basis function (a = 0, 1, 2 ...)
# The returns are the f(xi, eta, zeta), df/dxi, df/deta and df/dzeta
def threeDBasisFunc(xi, eta, zeta, a):
	k = int(a/4)  # the first index
	j = int(a%4/2)  # the second index
	i = a%2
	
	f = (b(xi, i))[0]*(b(eta, j))[0]*(b(zeta, k))[0]
	fd1 = (b(xi, i))[1]*(b(eta, j))[0]*(b(zeta, k))[0]
	fd2 = (b(xi, i))[0]*(b(eta, j))[1]*(b(zeta, k))[0]
	fd3 = (b(xi, i))[0]*(b(eta, j))[0]*(b(zeta, k))[1]
	return f, fd1, fd2, fd3

##############################################################################

# This function returns evaluations of the basis functions and their derivatives in the 3D case
# over domain (-1, 1) x (-1, 1) x (-1, 1) and variables 'xi', 'eta' and 'zeta'. The input variables are:
# 'xi' - the first independent variable
# 'eta' - the second independent variable
# 'zeta' - the third independent variable
def threeDBasis(d):
	# A table of integration points and weights where the indexes are
	# [interior (0) or one of four surfaces, integration point #] and 
	# the data stores is a [point coordinate in 'xi' and 'eta', weight]
	intpts = getIntPts(3, d)
	
	# Next, we intialize and fill and array with basis function evaluations at quadrature poins where the
	# array is indexed by [element location (0 - interior, 1,2,... walls)][int. point #][basis function #]
	# [0 - function, 1 - df/dxi, 2 - df/deta ...]
	basis = []  # stores the basis data at each quadrature point
	for i in range(len(intpts)):  # for each element region s...
		basis.append([])  # add a slot for the element region
		for j in range(len(intpts[i])):  # for each quadrature point...
			basis[i].append([])  # add a slot for each quadrature point
			for k in range(8):  # for each basis function...
				f, fd1, fd2, fd3 = threeDBasisFunc(intpts[i][j][0][0], intpts[i][j][0][1], intpts[i][j][0][2], k)
				basis[i][j].append([f, fd1, fd2, fd3])
	
	return basis

#########################################################################

# Here, we define a function to test the output of oneDBasis()
def oneDBasisTest():
	basis = oneDBasis()
	d = 1.0/3**0.5  # the distance of a point from the origin
	h1 = 0.5*(1-d)  # b1's output at 'd'
	h2 = 0.5*(1+d)  # b1's output at '-d'
	s1 = -0.5  # b1's slope
	s2 = 0.5  # b2's slope
	test = [[[[h2, s1], [h1, s2]], [[h1, s1], [h2, s2]]]]  # expected output
	print('Is 1D broken?:', not(basis == test))  # is expected?

########################################################################

# Next, we write a test for the 2D basis output
def twoDBasisTest():
	basis = twoDBasis()  # get the basis
	d = 1.0/3**0.5  # the distance of a point from the origin
	intpts = getIntPts(2)
	
	problem = False  # indicates whether a value is incorrect
	for i in range(len(basis)):  # for every element region...
		for j in range(len(basis[i])):  # for every integration point...
			k = 3  # the basis number
			# get the appropriate values form one of the bases
			f, fd, fdd = bb4(intpts[i][j][0][0], intpts[i][j][0][1])  # get real values
			if((f != basis[i][j][k][0]) or (fd != basis[i][j][k][1]) or (fdd != basis[i][j][k][2])):
				problem = True  #  the stored value is incorrect
	
	print('Is 2D broken?:', problem)

################################################################

# Next, we write a test for the 3D basis output
def threeDBasisTest():
	basis = threeDBasis()  # get the basis
	d = 1.0/3**0.5  # the distance of a point from the origin
	# A table of integration points and weights where the indexes are
	# [interior (0) or one of four surfaces, integration point #] and 
	# the data stores is a [point coordinate in 'xi' and 'eta', weight]
	intpts = getIntPts(3)
	
	problem = False  # is the code broken?
	
	for i in range(len(basis)):  # for every element region...
		for j in range(len(basis[i])):  # for every integration point...
			for k in range(8):  # for every basis function...
				# get the appropriate values form one of the bases
				f, fd1, fd2, fd3 = threeDBasisFunc(intpts[i][j][0][0], intpts[i][j][0][1], intpts[i][j][0][2], k)
				if((f != basis[i][j][k][0]) or (fd1 != basis[i][j][k][1]) or (fd2 != basis[i][j][k][2])):
					problem = True  #  the stored value is incorrect
	
	print('Is 3D broken?:', problem)

######################################################

# This function returns the appropriate basis array for the given element dimension (a 2D basis for 2D problem, 
# 3D basis for a 3D problem). 

# The inputs are
# 'numD' - the number of problem dimensions
# The returned array is 'basis' and is indexed as [element location (0 - interior, 1,2,... walls)][int. point #]
# [basis function #][0 - function, 1 - df/dxi, 2 - df/deta ...]

def getBasis(numD, d=1.0/3**0.5):
        # the distance of a point from the origin
	basis = []  # initializes 'basis'
	if numD == 1:
                basis = oneDBasis(d)
	if numD == 2:
		basis = twoDBasis(d)
	if numD == 3:
        	basis = threeDBasis(d)
	return basis

#########################################################################




