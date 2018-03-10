# In this file, we test the code needed to guarantee full plot flexibility


import unittest
from Assignment_1 import nodeList, get_ien, getIDArray
from Assignment_2 import load_and_cons
from Assignment_8 import solver
from Assignment_9 import get_stress_sol


##########################################################################

# This first class tests the ability of the stress solution function to operate

class StressSolutionTest(unittest.TestCase):
    # Here, we initialize solution dat needed for the tests
    def setUp(self):
        self.numD = [1, 2, 3]  # the number of problem dimensions
        enum = [1, 2]  # number of elements
        self.nodes1 = nodeList(1, 1, 1, enum[0])
        self.ien1 = get_ien(enum[0])
        self.cons1, self.load1 = load_and_cons(enum[0], len(self.nodes1),
                                               self.numD[0])
        self.cons1[0][0] = 0.0
        self.load1[2][0] = [2.0e6, 0, 0]  # load to right end
        self.ida1, self.ncons1 = getIDArray(self.cons1)

        # now the 3D problem setup
        self.nodes3 = nodeList(1, 1, 1, enum[0], enum[0], enum[0])
        self.ien3 = get_ien(enum[0], enum[0], enum[0])
        self.cons3, self.load3 = load_and_cons(enum[0], len(self.nodes3),
                                               self.numD[2])
        self.cons3[1][0] = 0.0  # constrain node 0 in all dof
        self.cons3[2][0] = 0.0
        self.cons3[2][2] = 0.0  # prevent rotation about x
        for i in [0, 2, 4, 6]:  # for every constrained node
            self.cons3[0][i] = 0.0
        
        self.load3[2][0] = [2.0e8, 0, 0]  # load to right end
        self.ida3, self.ncons3 = getIDArray(self.cons3)

    # Now we test the solution process for a 1D problem
    def test_stressSolution1D1Elem(self):
        result, steps = solver(self.numD[0], self.load1, self.nodes1, self.ien1,
                               self.ida1, self.ncons1, self.cons1)

        stress = get_stress_sol(result, self.ien1, self.nodes1)
        correct = [[[2.0e6]], [[2.0e6]]]

        for i in range(len(correct)):  # for each node...
            for j in range(len(correct[0])):  # for each degree of freedom...
                self.assertAlmostEqual(correct[i][j][0], stress[i][j][0])

    # Here, we test the solution process for 3D
    def test_stressSolution3D1Elem(self):
        result, steps = solver(self.numD[2], self.load3, self.nodes3, self.ien3,
                               self.ida3, self.ncons3, self.cons3)

        stress = get_stress_sol(result, self.ien3, self.nodes3)
        correct = [[[2.0e8], [0], [0]], [[2.0e8], [0], [0]], [[2.0e8], [0], [0]],
                   [[2.0e8], [0], [0]], [[2.0e8], [0], [0]], [[2.0e8], [0], [0]],
                   [[2.0e8], [0], [0]], [[2.0e8], [0], [0]]]
        
        for i in range(len(correct)):  # for each node...
            for j in range(len(correct[0])):  # for each degree of freedom...
                self.assertAlmostEqual(correct[i][j][0], stress[i][j][0],3)

#############################################################################

# Now the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(StressSolutionTest)

FullSuite = unittest.TestSuite([Suite1])

SingleSuite = unittest.TestSuite()
SingleSuite.addTest(StressSolutionTest('test_stressSolution1D1Elem'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)




