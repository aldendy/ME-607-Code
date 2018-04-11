# In this file, we test the additional functions and some of the changes made
# to make our code non-linear

import unittest
import numpy as np
from Assignment_1 import nodeList, get_ien
from Assignment_4 import getBasis
from Assignment_5 import posAndJac
from Assignment_6 import getElemDefs
from Assignment_10 import getVoigt, getSquareFromVoigt

#############################################################################

class VogtConversionTest(unittest.TestCase):
    # First we test the function for getting Voigt notation
    def test_getVoigt1D(self):  # testing in 1D
        mat = [[5]]  # test tensor
        result = getVoigt(mat)
        correct = [[5]]
        self.assertEqual(mat[0][0], correct[0][0])

    # Now, test the 2D case...
    def test_getVoigt2D(self):
        mat = [[2, 3], [4, 5]]  # test tensor
        result = getVoigt(mat)
        correct = [[2], [5], [3]]
        
        for i in range(len(result)):  # for every row (entry in Voigt)...
            self.assertEqual(correct[i][0], result[i][0])

    # Here, we test the 3D case...
    def test_getVoigt3D(self):
        mat = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]  # test tensor
        result = getVoigt(mat)
        correct = [[2], [6], [10], [7], [4], [3]]
        
        for i in range(len(result)):  # for every row (entry in Voigt)...
            self.assertEqual(correct[i][0], result[i][0])

    # Now, we test the reverse process, going from Voigt to Square notation
    # In 1D...
    def test_getSquare1D(self):
        mat = [[4]]  # test tensor
        result = getSquareFromVoigt(mat)
        correct = [[4]]
        self.assertEqual(result[0][0], correct[0][0])

    # For the 2D case...
    def test_getSquare2D(self):
        mat = [[3], [4], [5]]  # test tensor
        result = getSquareFromVoigt(mat)
        correct = [[3, 5], [5, 4]]
        
        for i in range(len(result)):  # for every row...
            for j in range(len(result[0])):  # for every column...
                self.assertEqual(correct[i][j], result[i][j])

    # Finally, we test the 3D case...
    def test_getSquare3D(self):
        mat = [[2], [6], [10], [7], [4], [3]]  # test tensor
        result = getSquareFromVoigt(mat)
        correct = [[2, 3, 4], [3, 6, 7], [4, 7, 10]]
        
        for i in range(len(result)):  # for every row (entry in Voigt)...
            self.assertEqual(correct[i][0], result[i][0])

#############################################################################

# Now the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(VogtConversionTest)

FullSuite = unittest.TestSuite([Suite1])

#SingleSuite = unittest.TestSuite()
#SingleSuite.addTest(PressurizedCylinderTest('test_accuracyPressCylinSol'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)
