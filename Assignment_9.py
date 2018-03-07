# In this file, we implement routines necessary to solve more general problems.
# We also implement functions to simplify the process of generating a mesh and
# establishing boundary conditions.


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Assignment_1 import getIDArray
from Assignment_2 import load_and_cons
from Assignment_8 import solver


#######################################################################

# Once we have a mesh, we will have the 'ien' array and the 'nodes' array.
# This will give use the number of elements, the nodes for each element and the
# nodal locations. We then need to constrain them and get the 'id' array, number
# of constraints and the 'cons' array.

# This will be done using the 'nsel' function which gets all the node numbers
# using specific location criteria

# The inputs are:
# 'nodes' - the 3D coordinates of all nodes in the mesh
# 'nodeNums' - a 'list' of the numbers (0, 1, 2...) of currently selected nodes
# 'direction' - 'x', 'y' or 'z' for the different directions
# 'selType' - the selection type ('n' - new set, 'a' - additional set,
#               's' - subset)
# 'loc' - the selection location in distance
# 'tol' - the selection tolerance in units of distance

# The return is:
# 'nodeSet' - the node numbers (0, 1, 2...) locating each member of 'nodeSet' in
#               'nodes', the global nodal array. The members are the selected
#               nodes in the mesh (a subset)

def nsel(nodes, nodeNums, direction, selType, loc, tol):
    axis = {'x':0, 'y':1, 'z':2}  # associates each direction with a number
    nodeSet = []  # contains all the locations of all selected nodes.

    if selType == 'n':  # if we are making a new selection...
        for i in range(len(nodes)):  # for every node in the mesh...
            # particular node coordinate in 'direction'
            p = nodes[i][axis[direction]]  
            if (p > (loc - tol)) and (p < (loc + tol)):  # if in bounds...
                nodeSet.append(i)  # add the node number
    
    elif selType == 'a':  # if we are making an additional selection...
        nodeSet += nodeNums  # add the already selected nodes
        for i in range(len(nodes)):  # for every node in the mesh...
            # particular node coordinate in 'direction'
            p = nodes[i][axis[direction]]  
            if (p > (loc - tol)) and (p < (loc + tol)):  # if in bounds...
                if i not in nodeNums:  # if the node is not currently selected...
                    nodeSet.append(i)
                    nodeSet.sort()
                    
    elif selType == 's':  # selecting a subset of the current set...
        for i in range(len(nodeNums)):  # for every selected node in the mesh...
            # particular node coordinate in 'direction'
            p = nodes[nodeNums[i]][axis[direction]]
            if (p > (loc - tol)) and (p < (loc + tol)):  # if in bounds...
                nodeSet.append(nodeNums[i])  # add the node number

    return nodeSet

##########################################################################

# After getting the nodes, we now need to constrain all the appropriate nodes

# The inputs are:
# 'nodes' - an array of the 3D locations of every node in the mesh
# 'selSet' - selected node set (location of node in the 'nodes' array (0, 1...))
# 'ien' - a map (global node #) = ien(element #, local eqn. #)
# 'dof' - the degree of freedom constrained ('x', 'y', 'z')
# 'd0'- the value of deformation applied (scalar)

# The outputs are:
# 'ida' - an array mapping (Eqn. #) = ID[Eqn. # including restrained dof's]
# 'ncons' - the number of dof constraints
# 'cons' - the constraint vector indexed by [dimension #][node #]

def constrain(nodes, selSet, ien, dof, d0):
    dims = {2:1, 4:2, 8:3}  # dictionary maping the number of element nodes to
                            # the number of dimensions
    dofMap = {'x':0, 'y':1, 'z':2}  # maps the degree of freedom of interest to
                                    # an appropriate dof number.
    cons, loads = load_and_cons(len(ien), len(nodes), dims[len(ien[0])])

    for i in range(len(selSet)):  # for every selected node...
        cons[dofMap[dof]][selSet[i]] = d0  # apply the constraint

    ida, ncons = getIDArray(cons)

    return ida, ncons, cons

###########################################################################

# Another essential component to more advanced analyses is the capability to
# plot results from simulations. This function is written to pring 2D fields
# of data extracted from simulation results

# The inputs are:
# 'deform' - the deformations for each degree of freedom of each node indexed by
#           [d1x, d1y, d2x, d2y...] for each node (1, 2...) and dof(x, y..)
# 'nodes' - an array of the 3D locations of all nodes in the mesh
# 'selSet' - a subset of selected nodes (positions in 'nodes') to plot
# 'viewNormal' - the direction from which to view the results (3D vector)

# The outputs are:

def plotResults(deform, nodes, selSet, viewNormal):
    plotName = {0:'Deformation', 1:'Stress', 2:'Strain'}  # contains plot titles
    uvn = np.array(viewNormal)/np.linalg.norm(viewNormal)  # normalize view vec

    
    # for every node, find its position perpendicular to the view
    #for i in range(len(selSet)):  # for every selected node...
        
    
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    a = [0, 0.5, 1]
    b = [0, 0.3, 0.6, 1]
    X, Y = np.meshgrid(a, b)
    
    Z1 = X + Y

    plt.figure()
    CS = plt.contour([0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 2])
    plt.title('Simplest default with labels')
    plt.clabel(CS, inline=1, fontsize=10, manual=1)
    plt.show()
    return 0

################################################################################




