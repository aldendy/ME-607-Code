# In this file, we implement routines necessary to solve more general problems.
# We also implement functions to simplify the process of generating a mesh and
# establishing boundary conditions.

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
#               'nodes', the global nodal array

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
# 'ien' - a map (global node #) = ien(element #, local eqn. #)



###########################################################################


