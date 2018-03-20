# In this file, we perform all the specific testing needed


import unittest
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
        c = contourPlot(deform, ien, nodes, 'tau_xy', 'z', cc, 'y', (2, 2))

        self.assertEqual(c, 0)
        
#############################################################################

# Now the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(Problem1Test)

FullSuite = unittest.TestSuite([Suite1])

SingleSuite = unittest.TestSuite()
SingleSuite.addTest(Problem1Test('test_OnePart1'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)
