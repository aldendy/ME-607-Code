# In this file, we test non-linear behavior in the internal force calculation
# implemented in Assignment_6


import unittest
import numpy as np
from Assignment_1 import get_ien
from Assignment_6 import getElemDefs


############################################################################

# Here, we test important auxilliary functions needed for smooth operation
class TestProcessUtils(unittest.TestCase):
    # First, we test the basics of the getElemDefs function
    def test_getElemDefs_1D(self):
        enum = 1
        deform = np.linspace(1, 1.9, 4)
        ien = get_ien(3)
        result = getElemDefs(enum, deform, ien)
        self.assertEqual(result[0], 1.3)
        self.assertEqual(result[1], 1.6)

    # Next, in 2D
    def test_getElemDefs_2D(self):
        enum = 1
        deform = np.linspace(1, 3.3, 24)
        ien = get_ien(3, 2)
        result = getElemDefs(enum, deform, ien)
        correct = [1.2, 1.3, 1.4, 1.5, 2, 2.1, 2.2, 2.3]
        
        for i in range(len(correct)):
            self.assertAlmostEqual(result[i], correct[i])
