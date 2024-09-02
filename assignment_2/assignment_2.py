"""In this file, we implement the data structures needed to store the 
constraints and loads applied to a given mesh.

This function generates the constraint and load arrays.

Inputs are:
'e' - the number of elements
'n' - the number of nodes
'nd' - the number of problem dimensions

Outputs are:
'cons' -  a blank array of size [# dimensions]x[# nodes] containing 'n' when no
          constraint exists and a number when there is (initialized to all 'n')
'loads' - an array of size [# element regions (s)]x[# elements] storing 'n'
          when no constraint exists, a number for a pressure and a vector for
          a traction or body force (initialized to all 'n')
"""
def load_and_cons(e, n, nd):	
	# first, we generate the constraint array (initialized with 0's)
	# this array is of size 'nd x n' and will contain 'n' when no
	# constraint exists and a scalar for the constraint value.
	cons = [['n' for j in range(n)] for i in range(nd)]
	
	# next, we generate the load array. Here, we assume that all 
	# elements are lines, quads or bricks. To do that, we must find
	# how many s's. This array relates (# element regions (s)) =
	# array(# of dimensions)
	numS = [3, 5, 7]  
	
	# This array will contain an 'n' when no load or body force is
	# applied and either a vector for a load/body force or a scalar for
	# a pressure applied at the boundary.
	loads = [['n' for j in range(e)] for i in range(numS[nd-1])]
        
	return cons, loads
