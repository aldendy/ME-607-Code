# In this file, we implement routines necessary to solve more general problems.
# We also implement functions to simplify the process of generating a mesh and
# establishing boundary conditions.


import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from Assignment_1 import getIDArray
from Assignment_2 import load_and_cons
from Assignment_4 import getBasis
from Assignment_6 import getXaArray, getBandScale, strainVec, stressVec
from Assignment_8 import solver


#######################################################################

# In order to properly select points, we must be able to compute the correct
# 'distance' in order to see if the node meets the selection criteria.

# The inputs are:
# 'node' - an individual, 3D coordinate for a node
# 'axis' - the axis on which the node position is measured ('x', 'y', 'z', 'r')
# 'origin' - the origin from which the criteria will be measured

# The returns are:
# 'dist' - the scalar distance needed for comparison

def position(node, axis, origin):
    axisL = {'x':0, 'y':1, 'z':2}  # associates each direction with a number
    
    if axis in ['x', 'y', 'z']:  # if the selection is planar...
        dist = (np.array(node) - np.array(origin))[axisL[axis]]
    if axis == 'r':  # if the selection is spherical...
        dist = np.linalg.norm((np.array(node) - np.array(origin)))

    return dist

##########################################################################

# Once we have a mesh, we will have the 'ien' array and the 'nodes' array.
# This will give use the number of elements, the nodes for each element and the
# nodal locations. We then need to constrain them and get the 'id' array, number
# of constraints and the 'cons' array.

# This will be done using the 'nsel' function which gets all the node numbers
# using specific location criteria

# The inputs are:
# 'direction' - 'x', 'y', 'z' or 'r' for the different directions
# 'selType' - the selection type ('n' - new set, 'a' - additional set,
#               's' - subset)
# 'loc' - the selection location in distance
# 'tol' - the selection tolerance in units of distance
# 'nodes' - the 3D coordinates of all nodes in the mesh
# 'nodeNums' - a 'list' of the numbers (0, 1, 2...) of currently selected nodes

# The return is:
# 'nodeSet' - the node numbers (0, 1, 2...) locating each member of 'nodeSet' in
#               'nodes', the global nodal array. The members are the selected
#               nodes in the mesh (a subset)

def nsel(direction, selType, loc, tol, nodes, nodeNums=0):
    nodeSet = []  # contains all the locations of all selected nodes.

    if selType == 'n':  # if we are making a new selection...
        for i in range(len(nodes)):  # for every node in the mesh...
            # particular node coordinate in 'direction'
            p = position(nodes[i], direction, [0, 0, 0])
            if (p > (loc - tol)) and (p < (loc + tol)):  # if in bounds...
                nodeSet.append(i)  # add the node number
    
    elif selType == 'a':  # if we are making an additional selection...
        nodeSet += nodeNums  # add the already selected nodes
        for i in range(len(nodes)):  # for every node in the mesh...
            # particular node coordinate in 'direction'
            p = position(nodes[i], direction, [0, 0, 0])
            if (p > (loc - tol)) and (p < (loc + tol)):  # if in bounds...
                if i not in nodeNums:  # if the node is not currently selected...
                    nodeSet.append(i)
                    nodeSet.sort()
                    
    elif selType == 's':  # selecting a subset of the current set...
        for i in range(len(nodeNums)):  # for every selected node in the mesh...
            # particular node coordinate in 'direction'
            p = position(nodes[nodeNums[i]], direction, [0, 0, 0])
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
# 'cons0' - the current set of constraints (defaults to 0 if not provided)

# The outputs are:
# 'ida' - an array mapping (Eqn. #) = ID[Eqn. # including restrained dof's]
# 'ncons' - the number of dof constraints
# 'cons' - the constraint vector indexed by [dimension #][node #]
# 'loads' - the load array indexed by [region # (s)][element #]

def constrain(nodes, selSet, ien, dof, d0, cons0='n'):
    dims = {2:1, 4:2, 8:3}  # dictionary maping the number of element nodes to
                            # the number of dimensions
    dofMap = {'x':0, 'y':1, 'z':2}  # maps the degree of freedom of interest to
                                    # an appropriate dof number.
    cons, loads = load_and_cons(len(ien), len(nodes), dims[len(ien[0])])
    if cons0 != 'n':  # if a previous constraint set is provided...
        cons = cons0

    for i in range(len(selSet)):  # for every selected node...
        cons[dofMap[dof]][selSet[i]] = d0  # apply the constraint

    ida, ncons = getIDArray(cons)

    return ida, ncons, cons, loads

###########################################################################

# We need a function that can compute the von Mises stress

# The inputs are:
# 's' - a stress vector in Voigt notation

# The outputs are:
# 'vM' - von Mises stress

def getMises(s):
    if len(s) == 6:  # in 3D...
        a = (s[0][0] - s[1][0])**2 + (s[1][0] - s[2][0])**2 + (s[2][0] - s[0][0])**2
        b = 6.0*(s[5][0]**2 + s[3][0]**2 + s[4][0]**2)
        vM = (0.5*(a + b))**0.5
    elif len(s) == 3:  # in 2D...
        vM = (s[0][0]**2 - s[0][0]*s[1][0] + s[1][0]**2 + 3*s[2][0]**2)**0.5
    else:  # in 1D...
        vM = s[0][0]

    return vM

#############################################################################

# This function returns the cylindrical stress at a given point given the node
# location and the stress vector (Voigt notation) at the point. It assumes that
# the problem has dimension of at least 2.

# The inputs are:
# 'pt' - the 3D coordinate of the point
# 's' - the stress vector in the appropriate number of problem dimensions (1-3)
# 'p' - the direction of the stress ('t' - tangential, 'r' - radial)
# 'axis' - the z-axis of the cylindrical coordinate system as a 3D vector

# The returns are:
# 'sigma_rt' - the radial or tangential stress at point 'pt'

def getSigmaR(pt, s, p, axis):
    cauchy = np.array(s[0])  # initialize cauchy stress matrix for the 1D case
    z = np.array(axis)/np.linalg.norm(np.array(axis))  #  unit vector on axis
    r = np.array(pt) - np.dot(np.array(pt), z)*z  # point radial position from z
    ru = r/np.linalg.norm(r)  # normalized radial position
    t = np.cross(z, ru)  # tangential direction
    
    if len(s) == 3:  # for the 2D case...
        cauchy = np.array([[s[0][0],    s[2][0],    0.0],
                           [s[2][0],    s[1][0],    0.0],
                           [0.0,        0.0,        0.0]])
    elif len(s) == 6:  # for the 3D case...
        cauchy = np.array([[s[0][0], s[5][0], s[4][0]],
                           [s[5][0], s[1][0], s[3][0]],
                           [s[4][0], s[3][0], s[2][0]]])
    if p == 't':  # for tangential stress
        return (np.dot(np.dot(cauchy, t), t))
    elif p == 'r':  # for the radial stress...
        sr = np.dot(np.dot(cauchy, ru), ru)
        return sr

#############################################################################

# An important function that will be needed for plotting is a method for finding
# the stress at the nodes.

# The inputs are:
# 'deform' - the deformations for each degree of freedom of each node indexed by
#           [d1x, d1y, d2x, d2y...] for each node (1, 2...) and dof(x, y..)
# 'ien' - an array mapping (Global node #) = ien[element #][element node #]
# 'nodes' - an array of the 3D locations of all nodes in the mesh
# 'stype' - the type of stress ('sigma_x', 'sigma_y', 'sigma_z', 'tau_xy',
#           'tau_yz', 'tau_zx', 'von Mises', 'sigma_r', 'sigma_t', 'd_abs')
# 'cCons' - an array of the parameters needed to define the constituitive law
#           that contains ['Young's Modulus', 'Poisson's Ratio']

# The outputs are:
# 'data' - an array of results for every node in the mesh

def get_stress_sol(deform, ien, nodes, stype, cCons=0):
    stypeMap3 = {'sigma_x':0, 'sigma_y':1, 'sigma_z':2, 'tau_xy':5, 'tau_yz':3,
                'tau_zx':4}  # maps types to numbers
    stypeMap2 = {'sigma_x':0, 'sigma_y':1, 'tau_xy':2}
    dMap = {'d_x':0, 'd_y':1, 'd_z':2}  # maps deformation dof's to numbers
    data = len(nodes)*[0]  # initializes the stress
    flags = len(nodes)*[0]  # records whether assignment has been made
    dims = int(len(deform)/len(nodes))  # number of problem dimensions
    
    for i in range(len(ien)):  # for every element...
        for j in range(len(ien[0])):  # for every node in the element...
            if flags[int(ien[i][j])] == 0:  # if assignment hasn't occurred...
                basis = getBasis(dims, 1.0)  # get basis evals at corners
                xa = getXaArray(i, nodes, ien)  # get the element nodes
                Bmats, scale = getBandScale(dims, basis, j, xa)
                strain = strainVec(dims, i, deform, ien, Bmats)
                if cCons != 0:
                    s = stressVec(dims, strain, cCons)
                else:
                    s = stressVec(dims, strain)
                
                if stype == 'von Mises':
                    data[int(ien[i][j])] = getMises(s)
                elif stype == 'sigma_r':  # if we want the radial stress...
                    data[int(ien[i][j])] = getSigmaR(nodes[int(ien[i][j])], s,
                                                     'r', [0, 0, 1])
                elif stype == 'sigma_t':  # if we want the tangential stress...
                    data[int(ien[i][j])] = getSigmaR(nodes[int(ien[i][j])], s,
                                                     't', [0, 0, 1])
                elif stype == 'd_abs':  # the magnitude of the deformation
                    n = int(ien[i][j])
                    first = dims*n
                    second = first + dims
                    data[int(ien[i][j])] = np.linalg.norm(deform[first:second])
                elif stype in dMap:  # for a component of deformation...
                    n = dims*int(ien[i][j]) + dMap[stype]
                    data[int(ien[i][j])] = deform[n]
                elif dims == 3:  # for three dimensions...
                    data[int(ien[i][j])] = s[stypeMap3[stype]][0]
                elif dims == 2:  # for two dimensions...
                    data[int(ien[i][j])] = s[stypeMap2[stype]][0]
                elif dims == 1:  # for one dimension...
                    data[int(ien[i][j])] = s[0][0]

                flags[int(ien[i][j])] = 1  # mark as having been assigned
    return data

###########################################################################

# Another essential component to more advanced analyses is the capability to
# plot results from simulations. This function is written to pring 2D fields
# of data extracted from simulation results

# The inputs are:
# 'deform' - the deformations for each degree of freedom of each node indexed by
#           [d1x, d1y, d2x, d2y...] for each node (1, 2...) and dof(x, y..)
# 'nodes' - an array of the 3D locations of all nodes in the mesh
# 'selSet' - a subset of selected nodes (positions in 'nodes') to plot
# 'plotDir' - a 3D vector following the direction of the selected nodes
# 'dof' - the degree of freedom of results desired ('x', 'y', 'z' or 'r')
# 's' - if not 'no', show the plot

# The outputs are:

def plotResults(deform, nodes, selSet, plotDir, dof, s='no'):
    dims = int(len(deform)/len(nodes))  # the numer of problem dimensions
    # contains plot titles
    plotName = {0:'Deformation', 1:'Stress', 2:'Strain'}
    plotAxes = {0:'Displacement in ', 1:'Stress in ', 2:'Strain in '}
    dofN = {'x':0, 'y':1, 'z':2}  # maps degree of freedom to number
    uvn = np.array(plotDir)/np.linalg.norm(plotDir)  # normalize view vec
    
    x = []  # distances of the nods from the origin along 'viewDir'
    y = []  # stores the values that will be plotted
    
    for i in range(len(selSet)):  # for every selected node...
        xi = np.dot(uvn, np.array(nodes[selSet[i]]))
        x.append(xi)  # add the appropriate distance
        y.append(deform[dims*selSet[i] + dofN[dof]])

    plt.plot(x, y)
    plt.title(plotName[0])
    plt.xlabel('Location along view vector')
    plt.ylabel(plotAxes[0] + dof)

    if s != 'no':
        plt.show()
    plt.close()
    
    return 0

################################################################################

# Here, we experiment with plotting contour plots of result fields

# The inputs are:
# 'deform' - a list of deformation data for each dof of each node indexed by
#           [d1x, d1y, d2x, d2y...] for the 2D case
# 'ien' - array mapping (global node #) = ien[element #][element node #]
# 'nodes' - a list of all the 3D locations of each node
# 'stype' - the type of stress ('sigma_x', 'sigma_y', 'sigma_z', 'tau_xy',
#           'tau_yz', 'tau_zx', 'von Mises')
# 'view' - picks on of the cardinal directions ('x', 'y', 'z') as the viewpoint
# 'cCons' - an array of the parameters needed to define the constituitive law
#           that contains ['Young's Modulus', 'Poisson's Ratio']
# 's' - if not 'no', show the plot
# 'size' - figure size in inches

# The outputs are:
# '0' - indicating it ran successfully

def contourPlot(deform, ien, nodes, stype, view, cCons=0, s='no', size=None):
    viewMap = {'x':0, 'y':1, 'z':2}
    titleMap = {'tau_xy':'tau_{xy}', 'tau_yz':'tau_{yz}', 'tau_zx':'tau_{zx}',
                'd_abs':'delta_{abs}', 'd_x':'delta_x', 'd_y':'delta_y',
                'd_z':'delta_z'}
    dimMap = {2:1, 4:2, 8:3}  # number of problem dimensions
    dims = dimMap[len(ien[0])]
    if cCons != 0:
        data = get_stress_sol(deform, ien, nodes, stype, cCons)
    else:
        data = get_stress_sol(deform, ien, nodes, stype)
    
    x = []  # stores the x-values
    y = []  # stores the y-values
    z = []  # stores the desired simulation value
    triang = []  # stores the particular trianglulation used to plot results

    for i in range(len(nodes)):  # for each node...
        x.append(nodes[i][0])
        y.append(nodes[i][1])
        z.append(data[i])

    for i in range(len(ien)):  # for each element...
        a = [ien[i][0], ien[i][1], ien[i][3]]  # first triangle
        b = [ien[i][0], ien[i][3], ien[i][2]]  # second triangle
        triang.append(a)  # assemble triangulation for ith element
        triang.append(b)
    
    mesh = tri.Triangulation(x, y, triang)
    lvs = 0
    span = abs(max(z, key=abs) - min(z, key=abs))  # range of plot
    
    if min(z, key=abs)/max(z, key=abs) > 0.99:  # if difference is small...
        lvs = list(min(z)*np.linspace(0, 1.0, 11))
        lvs.sort()
    
    # if difference is small...
    if span < 10**-12 and max(z, key=abs) < 2*span:  # for plots of small nums...
        lvs = list(np.linspace(-1, min(z), 11))
    
    # pcolor plot.
    fig = plt.figure(figsize=size)
    plt.gca().set_aspect('equal')
    
    if lvs != 0:  # set contour plot levels
        img = plt.tricontourf(mesh, z, extend='both', levels=lvs)
    else:
        img = plt.tricontourf(mesh, z, extend='both')
    
    if stype in titleMap:
        plt.title('$\\' + titleMap[stype] + '$')  # set the title
    else:
        plt.title('$\\' + stype + '$')  # set the title
    
    axesTitles = ['x', 'y', 'z']  # organize the other axis labels
    axesTitles.remove(view)

    if s != 'no':  # if it should be plotted...
        plt.xlabel(axesTitles[0])
        plt.ylabel(axesTitles[1])
        plt.colorbar(img, ax=plt.gca(), pad=0.1)
        plt.tight_layout()
        plt.show()

    plt.close()
    
    return 0

################################################################################






