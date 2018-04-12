# In this file, we test the additional functions and some of the changes made
# to make our code non-linear

import unittest
import numpy as np
from Assignment_1 import nodeList, get_ien
from Assignment_4 import getBasis
from Assignment_5 import posAndJac
from Assignment_6_Utils import getElemDefs
from Assignment_10 import getF, getVoigt, getSquareFromVoigt, getGstrain

############################################################################

# Here, we test the calculation process for the 'F' matrix
class TestFGeneration(unittest.TestCase):
    # First, we test the 'F' matrix in multiple dimensions. First 1D...
    def test_F_1D_1Elem(self):
        deform = [0.0, 0.1]
        ien = get_ien(1)
        basis = getBasis(1)
        xa = nodeList(2, 2, 2, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        F = getF(defE, basis[0][0], jac)
        self.assertEqual(F[0][0], 1.05)

    # Next, in 2D
    def test_F_2D_1Elem(self):
        deform = [0.0, 0.0, 0.1, 0.0, 0.0, -0.03, 0.1, -0.03]
        ien = get_ien(1, 1)
        basis = getBasis(2)
        xa = nodeList(1.0, 0.5, 0.5, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        F = getF(defE, basis[0][0], jac)
        correct = [[1.1, 0], [0, 0.94]]
        
        for i in range(len(F)):  # for every row...
            for j in range(len(F[0])):  # for every column...
                self.assertAlmostEqual(correct[i][j], F[i][j])

    # Finally, we test in 3D
    def test_F_3D_1Elem(self):
        deform = [0, 0, 0, 0.1, 0, 0, 0, -0.03, 0, 0.1, -0.03, 0,
                  0, 0, -0.03, 0.1, 0, -0.03, 0, -0.03, -0.03, 0.1, -0.03, -0.03]
        ien = get_ien(1, 1, 1)
        basis = getBasis(3)
        xa = nodeList(2, 2, 2, 1, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        F = getF(defE, basis[0][0], jac)
        correct = [[1.05, 0, 0], [0, 0.985, 0], [0, 0, 0.985]]
        
        for i in range(len(F)):  # for every row...
            for j in range(len(F[0])):  # for every column...
                self.assertAlmostEqual(correct[i][j], F[i][j])

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

# Here, we test the proper functioning of the Green strain calculation
class GreenStrainTests(unittest.TestCase):
    # First, we test the calculation in 1D
    def test_GS_1D_1Elem_Normal(self):
        d = 2.0e-1  # deformation amount
        elemL = 2.0  # element length
        deform = [0.0, d]
        ien = get_ien(1)
        basis = getBasis(1)
        xa = nodeList(elemL, elemL, elemL, 1)
        x, jacXxi = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        GSv, F = getGstrain(defE, basis[0][0], jacXxi)
        
        correct = [[d/elemL + 0.5*(d/elemL)**2]]
        self.assertAlmostEqual(GSv[0][0], correct[0][0])

    # Next, we test in 2D
    def test_GS_2D_1Elem_Normal(self):
        d = 2.0e-5  # deformation amount
        eLx = 2.0  # x-direction element size
        eLy = 1.0  # y-direction element size
        nu = 0.3  # Poisson's ratio
        deform = [0, 0, d, 0, 0, -nu*d, d, -nu*d]
        ien = get_ien(1, 1)
        basis = getBasis(2)
        xa = nodeList(eLx, eLy, 2, 1, 1)
        x, jacXxi = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        GSv, F = getGstrain(defE, basis[0][1], jacXxi)

        ex = d/eLx
        ey = d/eLy
        correct = [[ex + 0.5*ex**2], [(ey*nu - 1)**2/2 - 0.5], [0]]
        
        for i in range(len(GSv)):  # for every row...
            self.assertAlmostEqual(GSv[i][0], correct[i][0])

    # Here, we test shear behavior in 2D in the xy plane
    def test_GS_2D_1Elem_Shear(self):
        d = 2.0e-1  # deformation amount
        eLx = 2.0  # x-direction element size
        eLy = 1.0  # y-direction element size
        deform = [0, 0, 0, 0, d, 0, d, 0]
        ien = get_ien(1, 1)
        basis = getBasis(2)
        xa = nodeList(eLx, eLy, 2, 1, 1)
        x, jacXxi = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        GSv, F = getGstrain(defE, basis[0][1], jacXxi)
        
        ey = d/eLy  # deformation ratio
        correct = [[0], [ey**2/2], [ey/2]]
        
        for i in range(len(GSv)):  # for every row...
            self.assertAlmostEqual(GSv[i][0], correct[i][0])

    # Here, we test shear behavior in 3D in the yz plane
    def test_GS_3D_1Elem_Shear(self):
        d = 2.0e-1  # deformation amount
        eLx = 2.0  # x-direction element size
        eLy = 1.0  # y-direction element size
        eLz = 2.0  # z-direction element size
        deform = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, d, 0, 0, d, 0, 0, d, 0, 0, d, 0]
        ien = get_ien(1, 1, 1)
        basis = getBasis(3)
        xa = nodeList(eLx, eLy, eLz, 1, 1, 1)
        x, jacXxi = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        GSv, F = getGstrain(defE, basis[0][2], jacXxi)
        
        ey = d/eLz  # deformation ratio
        correct = [[0], [0], [ey**2/2], [ey/2], [0], [0]]
        
        for i in range(len(GSv)):  # for every row...
            self.assertAlmostEqual(GSv[i][0], correct[i][0])

#############################################################################

# Now the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(TestFGeneration)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(VogtConversionTest)
Suite3 = unittest.TestLoader().loadTestsFromTestCase(GreenStrainTests)

FullSuite = unittest.TestSuite([Suite1, Suite2, Suite3])

#SingleSuite = unittest.TestSuite()
#SingleSuite.addTest(PressurizedCylinderTest('test_accuracyPressCylinSol'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)
