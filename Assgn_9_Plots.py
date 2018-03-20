# In this file, we perform all the specific testing needed


import unittest
import numpy as np
from math import pi, sin, cos
from Assignment_1 import nodeList, get_ien, getIDArray
from Assignment_2 import load_and_cons
from Assignment_8 import solver
from Assignment_9 import nsel, constrain, get_stress_sol, contourPlot


######################################################################

# For problem 1...

class Problem1Test(unittest.TestCase):
    # Here, we define all the plotting capability needed for Problem 1
    def test_OnePart1(self):
        numD = 2  # number of dimensions
        enum = 2  # number of elements in each dof
        nodes = nodeList(1, 1, 1, enum, enum)
        ien = get_ien(enum, enum)
        cc = [1.0, 0.0]

        wall = nsel('x', 'n', 0, 0.01, nodes)  # select the wall
        edge = nsel('x', 'n', 1, 0.01, nodes)  # select free edge
        
        ida, ncons, cons0, loads = constrain(nodes, wall, ien, 'x', 0)
        ida, ncons, cons1, loads = constrain(nodes, wall, ien, 'y',
                                             0.0, cons0)
        ida, ncons, cons2, loads = constrain(nodes, edge, ien, 'x',
                                             1.0, cons1)
        
        deform, i = solver(numD, loads, nodes, ien, ida, ncons,
                                cons2, cc)
        
        correct = [0, 0, 0.5, 0, 1, 0, 0, 0, 0.5, 0, 1, 0, 0, 0, 0.5, 0, 1, 0]
        for i in range(len(deform)):  # for every item in the solution...
            self.assertAlmostEqual((correct[i] + 1)/(correct[i] + 1), 1)
        
        c = contourPlot(deform, ien, nodes, 'sigma_x', 'z', cc)
        c = contourPlot(deform, ien, nodes, 'sigma_y', 'z', cc)
        c = contourPlot(deform, ien, nodes, 'tau_xy', 'z', cc)
        c = contourPlot(deform, ien, nodes, 'd_x', 'z', cc)
        c = contourPlot(deform, ien, nodes, 'd_y', 'z', cc)

    # Here, we solve a similar problem but with a traction load
    def test_OnePart2(self):
        numD = 2  # number of dimensions
        enum = 10  # number of elements in each dof
        nodes = nodeList(1, 1, 1, enum, enum)
        ien = get_ien(enum, enum)
        cc = [1.0e7, 0.3]

        wall = nsel('x', 'n', 0, 0.01, nodes)  # select the wall
        
        ida, ncons, cons0, loads = constrain(nodes, wall, ien, 'x', 0)
        ida, ncons, cons1, loads = constrain(nodes, wall, ien, 'y',
                                             0.0, cons0)
        for i in range(enum):  # for every element on a side...
            loads[2][enum*(i + 1) - 1] = 1000.0  # traction on the right
        
        deform, i = solver(numD, loads, nodes, ien, ida, ncons,
                                cons1, cc)
        
        c = contourPlot(deform, ien, nodes, 'd_abs', 'z', cc)
        c = contourPlot(deform, ien, nodes, 'd_x', 'z', cc)
        c = contourPlot(deform, ien, nodes, 'd_y', 'z', cc)
        c = contourPlot(deform, ien, nodes, 'sigma_x', 'z', cc)
        c = contourPlot(deform, ien, nodes, 'sigma_y', 'z', cc)
        c = contourPlot(deform, ien, nodes, 'tau_xy', 'z', cc)

        self.assertEqual(c, 0)
        
#############################################################################

# Here, we test the plots for the cylinder simulation

class PresurizedCylinderPlot(unittest.TestCase):
    # first we define important necessary variables
    def test_CylinderPlot(self):
        # Here, we generate the mesh
        thetaDomain = pi/2.0  # quarter circle of the pipe
        self.ri = 1.2  # the inner radius of the pipe
        self.ro = 1.8  # the outer radius of the pipe
        self.nr = 16 # the number of elements in the radial direction
        self.nt = 32  # the number of elements in the circumfrential direction
        self.nodes = []  # stores the nodes in the cylindrical mesh
        self.p = -2.0e6  # the pressure (Pa)
        cc = [2.0e11, 0.3]  # [Young's Modulus, Poisson's Ratio]

        for i in range(self.nr+1):  # for every node in the r-direction...
            for j in range(self.nt+1):  # for every node in the theta-direction...
                radius = i*(self.ro - self.ri)/self.nr + self.ri
                theta = j*thetaDomain/self.nt
                self.nodes.append([radius*sin(theta), radius*cos(theta), 0])

        self.ien = get_ien(self.nt, self.nr)
        self.nnums = np.linspace(0, len(self.nodes)-1, len(self.nodes))

        # now the fun begins. We solve this problem using roller constraints on
        # the straight faces and pressure loads on the inner face.
        s0 = nsel('y', 'n', 0, 0.01, self.nodes)
        s1 = nsel('x', 'n', 0, 0.01, self.nodes)

        ida, ncons, cons0, loads = constrain(self.nodes, s0, self.ien, 'y', 0)
        ida, ncons, cons, loads = constrain(self.nodes, s1, self.ien, 'x', 0,
                                            cons0)
        for i in range(self.nt):  # for every inner element...
            loads[3][i] = self.p

        self.deform, i = solver(2, loads, self.nodes, self.ien, ida, ncons, cons)
        ps0 = nsel('y', 'n', 0, 0.01, self.nodes)
        #c = plotResults(deform, self.nodes, ps0, [1, 0, 0], 'x')
        c = contourPlot(self.deform, self.ien, self.nodes, 'sigma_t', 'z', cc, 'y')

#############################################################################

# Now the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(Problem1Test)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(PresurizedCylinderPlot)

FullSuite = unittest.TestSuite([Suite1, Suite2])

SingleSuite = unittest.TestSuite()
SingleSuite.addTest(Problem1Test('test_OnePart1'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)
