# This file implements all necessary tests for the code in Assignment_8


import unittest
from Assignment_8 import *


###################################################################

# The first test to implement will determine if the stiffnes matrix
# integrand in 'getEnergyDensity' is working properly.

class BasicTest(unittest.TestCase):
    # Here, we perform a simple test on the integrand.
    def test_getEnDen1D(self):
        dims = 1
        intpt = 0
        xa = [[0, 0, 0], [1, 0, 0]]
        # Next, we test for all combinations of 'a' and 'b'
        for i in range(2):  # for every 'a' or local element #...
            for j in range(2):  # for every 'b'...
                kab = getEnergyDensity(dims, intpt, xa, i, j)
                self.assertAlmostEqual(kab, (-1)**(i+j)*200e9)

    # next, we test in 2D
    def test_getEnDen2D(self):
        dims = 2
        intpt = 0
        xa = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        a = 0
        b = 0

        kab = getEnergyDensity(dims, intpt, xa, a, b)
        # come back to this answer and check
        correct = [[1.84551963e+11, 8.88583526e+10], [8.88583526e+10, 1.84551963e+11]]
        for i in range(len(kab)):  # for every row...
            for j in range(len(kab)):  # for every column...
                self.assertAlmostEqual(kab[i][j]/correct[i][j], 1)

###################################################################

# Now the testing.
        
Suite1 = unittest.TestLoader().loadTestsFromTestCase(BasicTest)

FullSuite = unittest.TestSuite([Suite1])

unittest.TextTestRunner(verbosity=2).run(FullSuite)
