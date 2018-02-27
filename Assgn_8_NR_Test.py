# This file tests the netwton-raphson algorithm discussed in the notes


import unittest
from Assignment_1 import nodeList, get_ien, getIDArray
from Assignment_2 import load_and_cons
from Assignment_8 import solver


######################################################################

# This class tests the operation of the solver using a single element
# problem as described in Assignment 8.

class BasicTest(unittest.TestCase):
    # In this first function, we test the solver written in Assignment_8
    def setUp(self):
        self.numD = [1, 2, 3]  # the number of problem dimensions
        enum = 1  # number of elements
        self.nodes1 = nodeList(1, 1, 1, enum)
        self.ien1 = get_ien(enum)
        self.cons1, self.load1 = load_and_cons(enum, len(self.nodes1), self.numD[0])
        self.load1[2][0] = [1, 0, 0]  # load to right end
        self.ida1, self.ncons1 = getIDArray(self.cons1)

    # Here, we test the 1D solution process
    def test_solver(self):
        result = solver(self.numD[0], self.load1, self.nodes1, self.ien1, self.ida1,
                        self.ncons1)

        correct = [0.0, 0.0]
        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual(correct[i]/result[i], 1)
        
    
###############################################################################

# now, the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(BasicTest)

FullSuite = unittest.TestSuite([Suite1])

unittest.TextTestRunner(verbosity=2).run(FullSuite)
