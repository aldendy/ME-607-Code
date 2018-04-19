# In this file, we test non-linear behavior in the internal force calculation
# implemented in Assignment_6


import unittest
import numpy as np
from Assignment_1 import nodeList, get_ien
from Assignment_6_Utils import getElemDefs, getStiff, getEulerStiff
from Assignment_6 import getYa


############################################################################

# Here, we test important auxilliary functions needed for smooth operation
class TestProcessUtils(unittest.TestCase):
    # First, we test the basics of the getElemDefs function
    def test_getElemDefs_1D(self):
        enum = 1
        deform = np.linspace(1, 1.9, 4)
        ien = get_ien(3)
        result = getElemDefs(enum, deform, ien)
        correct = [[1.3, 0, 0], [1.6, 0, 0]]
        
        for i in range(len(result)):  # for every node...
            for j in range(len(result[0])):  # for every dof...
                self.assertEqual(result[i][j], correct[i][j])

    # Next, in 2D
    def test_getElemDefs_2D(self):
        enum = 1
        deform = np.linspace(1, 3.3, 24)
        ien = get_ien(3, 2)
        result = getElemDefs(enum, deform, ien)
        correct = [[1.2, 1.3, 0], [1.4, 1.5, 0], [2, 2.1, 0], [2.2, 2.3, 0]]
        
        for i in range(len(correct)):  # for every node...
            for j in range(len(correct[0])):  # for every dof...
                self.assertAlmostEqual(result[i][j], correct[i][j])

############################################################################

# Here, we test the getYa function and ensure that it properly merges the
# element node array and the deformation array.
class TestGetCurrentElemNodeLoc(unittest.TestCase):
    # First, test 1D merging
    def test_getYa1D(self):
        elemNum = 1   # element number
        nodes = nodeList(3, 3, 3, 2)
        deform = [0.1, 0.2, 0.3]
        ien = get_ien(2)
        ya = getYa(elemNum, nodes, deform, ien)
        correct = [[1.7, 0, 0], [3.3, 0, 0]]
        
        for i in range(len(ya)):  # for every node...
            for j in range(len(ya[0])):  # for every dimension...
                self.assertAlmostEqual(ya[i][j], correct[i][j])

    # Next, we test merging in 2D
    def test_getYa2D(self):
        elemNum = 0  # element
        nodes = nodeList(3, 2, 2, 2, 1)
        deform = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]
        ien = get_ien(2, 1)
        ya = getYa(elemNum, nodes, deform, ien)
        correct = [[0, 0.1, 0], [1.7, 0.3, 0], [0.6, 2.7, 0], [2.3, 2.9, 0]]
        
        for i in range(len(ya)):  # for every node...
            for j in range(len(ya[0])):  # for every dimension...
                self.assertAlmostEqual(ya[i][j], correct[i][j])

    # Finally, we test merging in 3D
    def test_getYa3D(self):
        elemNum = 0  # element
        nodes = nodeList(3, 2, 2, 2, 1, 1)
        deform = np.linspace(0, 3.5, 36)
        ien = get_ien(2, 1, 1)
        ya = getYa(elemNum, nodes, deform, ien)
        correct = [[0, 0.1, 0.2], [1.8, 0.4, 0.5], [0.9, 3, 1.1], [2.7, 3.3, 1.4],
                   [1.8, 1.9, 4], [3.6, 2.2, 4.3], [2.7, 4.8, 4.9], [4.5, 5.1, 5.2]]
        
        for i in range(len(ya)):  # for every node...
            for j in range(len(ya[0])):  # for every dimension...
                self.assertAlmostEqual(ya[i][j], correct[i][j])

###############################################################################

# Next, we perform testing on the elasticity tensor methods and pushing forward
class Elasticity_Tensor_Test(unittest.TestCase):
    # for the Eulerian tensor (pushed forward)...
    def test_EulerTensor1D_linear(self):
        E = 1.5  # Young's Modulus
        v = 0.3  # Poisson's ratio
        a = 0.0/2  # strain
        F = [[a + 1, 0, 0], [0, 1, 0], [0, 0, 1]]
        C = getEulerStiff(F, 1, [E, v])
        correct = [[E]]
        
        for i in range(len(C)):  # for every row...
            for j in range(len(C[0])):  # for every column...
                self.assertAlmostEqual(correct[i][j], C[i][j])

    def test_EulerTensor1D_large(self):
        E = 1.5  # Young's Modulus
        v = 0.3  # Poisson's ratio
        a = 0.2/2  # strain
        F = [[a + 1, 0, 0], [0, 1 - v*a, 0], [0, 0, 1 - v*a]]
        C = getEulerStiff(F, 1, [E, v])
        correct = [[2.38052852609754]]
        
        for i in range(len(C)):  # for every row...
            for j in range(len(C[0])):  # for every column...
                self.assertAlmostEqual(correct[i][j], C[i][j])

    def test_EulerTensor3D_linear(self):
        E = 2.0  # Young's Modulus
        v = 0.25  # Poisson's ratio
        a = 0.0/2  # strain
        F = [[a + 1, 0, 0], [0, 1 - v*a, 0], [0, 0, 1 - v*a]]
        C = getEulerStiff(F, 3, [E, v])
        
        correct = getStiff(3, [E, v])
        
        for i in range(len(C)):  # for every row...
            for j in range(len(C[0])):  # for every column...
                if (i > 2) or (j > 2):  # shear terms are not the same 
                    self.assertAlmostEqual(correct[i][j]/2, C[i][j])
                else:
                    self.assertAlmostEqual(correct[i][j], C[i][j])

    def EulerTensor3D_large(self):
        E = 2.0  # Young's Modulus
        v = 0.25  # Poisson's ratio
        a = 0.2/2  # strain
        F = [[a + 1, 0, 0], [0, 1 - v*a, 0], [0, 0, 1 - v*a]]
        C = getEulerStiff(F, 3, [E, v])
        print(C)
        correct = getStiff(3, [E, v])
        
        for i in range(len(C)):  # for every row...
            for j in range(len(C[0])):  # for every column...
                if (i > 2) or (j > 2):  # shear terms are not the same 
                    self.assertAlmostEqual(correct[i][j]/2, C[i][j])
                else:
                    self.assertAlmostEqual(correct[i][j], C[i][j])

###############################################################################

# Here, we perform the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(TestProcessUtils)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(TestGetCurrentElemNodeLoc)
Suite3 = unittest.TestLoader().loadTestsFromTestCase(Elasticity_Tensor_Test)

FullSuite = unittest.TestSuite([Suite1, Suite2, Suite3])

singleTestSuite = unittest.TestSuite()
singleTestSuite.addTest(Elasticity_Tensor_Test('test_EulerTensor3D_linear'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)
        
