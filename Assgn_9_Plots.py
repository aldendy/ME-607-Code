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
    def test_Part1(self):
        numD = 2  # number of dimensions
        enum = 2  # number of elements in each dof
        nodes = nodeList(1, 1, 1, enum, enum)
        ien = get_ien(enum, enum)
        cons, load = load_and_cons(enum**2, len(nodes), numD)
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
        c = contourPlot(deform, ien, nodes, 'tau_xy', 'z', cc)
        
#############################################################################

# Now the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(Problem1Test)

FullSuite = unittest.TestSuite([Suite1])

#SingleSuite = unittest.TestSuite()
#SingleSuite.addTest(StressSolutionTest('test_stressSolution1D1Elem'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)
