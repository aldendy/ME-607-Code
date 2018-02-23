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
        a = 0
        b = 0

        kab = getEnergyDensity(dims, intpt, xa, a, b)
        self.assertAlmostEqual(kab, 200e9)

###################################################################

# Now the testing.
        
Suite1 = unittest.TestLoader().loadTestsFromTestCase(BasicTest)

FullSuite = unittest.TestSuite([Suite1])

unittest.TextTestRunner(verbosity=2).run(FullSuite)
