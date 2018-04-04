# In this file, we test all the code written for the non-linear code


import unittest
import numpy as np
from Assignment_1 import nodeList, get_ien
from Assignment_4 import getBasis
from Assignment_5 import posAndJac
from Assignment_10 import getElemDefs, getF, getPK2, getCauchy


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

############################################################################

# Here, we test the second Piola-Kirchhoff stress tensor calculation
class TestPK2Stress(unittest.TestCase):
    # First, we test the stress in 1D...
    def test_S_1D_1Elem(self):
        deform = [0.0, 2.0e-5]
        ien = get_ien(1)
        basis = getBasis(1)
        xa = nodeList(2, 2, 2, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getPK2(defE, basis[0][0], jac)
        correct = [[2000010.00001393]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual(correct[i], S[i])

    # For large deformation
    def test_S_1D_1Elem_Large(self):
        deform = [0.0, 0.2]
        ien = get_ien(1)
        basis = getBasis(1)
        xa = nodeList(2, 2, 2, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getPK2(defE, basis[0][0], jac)
        correct = [[2.1e10]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual(correct[i], S[i], 3)

    # For 2D small deformation...
    def test_S_2D_1Elem(self):
        deform = [0, 0, 2.0e-5, 0, 0, -0.6e-5, 2.0e-5, -0.6e-5]
        ien = get_ien(1, 1)
        basis = getBasis(2)
        xa = nodeList(2, 2, 2, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getPK2(defE, basis[0][0], jac)
        correct = [[2.00001129e+06], [4.28572684e+00], [0]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual(correct[i][0], S[i][0], 1)

    # For 2D large deformation...
    def test_S_2D_1Elem_Large(self):
        deform = [0, 0, 0.2, 0, 0, -0.06, 0.2, -0.06]
        ien = get_ien(1, 1)
        basis = getBasis(2)
        xa = nodeList(2, 2, 2, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getPK2(defE, basis[0][0], jac)
        correct = [[2.11285714e+10], [4.28571429e+08], [0.00000000e+00]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1)

    # For 3D small deformation...
    def test_S_3D_1Elem(self):
        deform = [0, 0, 0, 2.0e-5, 0, 0, 0, -0.6e-5, 0, 2.0e-5, -0.6e-5, 0,
                  0, 0, -0.6e-5, 2.0e-5, 0, -0.6e-5,
                  0, -0.6e-5, -0.6e-5, 2.0e-5, -0.6e-5, -0.6e-5]
        ien = get_ien(1, 1, 1)
        basis = getBasis(3)
        xa = nodeList(2, 2, 2, 1, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getPK2(defE, basis[0][0], jac)
        correct = [[2.00001450e+06], [7.50002197], [7.50002197], [0], [0], [0]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1)

    # For 3D large deformation...
    def test_S_3D_1Elem_Large(self):
        deform = [0, 0, 0, 2.0e-1, 0, 0, 0, -0.6e-1, 0, 2.0e-1, -0.6e-1, 0,
                  0, 0, -0.6e-1, 2.0e-1, 0, -0.6e-1,
                  0, -0.6e-1, -0.6e-1, 2.0e-1, -0.6e-1, -0.6e-1]
        ien = get_ien(1, 1, 1)
        basis = getBasis(3)
        xa = nodeList(2, 2, 2, 1, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getPK2(defE, basis[0][0], jac)
        correct = [[2.145e+10], [7.5e+08], [7.5e+08], [0], [0], [0]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1)

############################################################################

# Here, we test the accuracy of the Cauchy stress tensor calculation

class TestCauchyStressTensor(unittest.TestCase):
    # First, we test 1D small deflection
    def test_sigma_1D_1Elem(self):
        deform = [0.0, 2.0e-5]
        ien = get_ien(1)
        basis = getBasis(1)
        xa = nodeList(2, 2, 2, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getCauchy(defE, basis[0][0], jac)
        correct = [[2000030.0001139301]]

        self.assertAlmostEqual(S[0][0], correct[0][0], 3)

    # For large deformation
    def test_sigma_1D_1Elem_Large(self):
        deform = [0.0, 0.2]
        ien = get_ien(1)
        basis = getBasis(1)
        xa = nodeList(2, 2, 2, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getCauchy(defE, basis[0][0], jac)
        correct = [[23100000000.000023]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual(correct[i][0], S[i][0], 3)

    # For 2D small deformation...
    def test_sigma_2D_1Elem(self):
        deform = [0, 0, 2.0e-5, 0, 0, -0.6e-5, 2.0e-5, -0.6e-5]
        ien = get_ien(1, 1)
        basis = getBasis(2)
        xa = nodeList(2, 2, 2, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getCauchy(defE, basis[0][0], jac)
        correct = [[2000037.2859566973], [4.2856711267160321], [0.0]]

        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1)

    # For 2D large deformation...
    def test_sigma_2D_1Elem_Large(self):
        deform = [0, 0, 0.2, 0, 0, -0.06, 0.2, -0.06]
        ien = get_ien(1, 1)
        basis = getBasis(2)
        xa = nodeList(2, 2, 2, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getCauchy(defE, basis[0][0], jac)
        correct = [[23960235640.648041], [377922077.92207932], [0.0]]

        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1)

    # For 3D small deformation...
    def test_sigma_3D_1Elem(self):
        deform = [0, 0, 0, 2.0e-5, 0, 0, 0, -0.6e-5, 0, 2.0e-5, -0.6e-5, 0,
                  0, 0, -0.6e-5, 2.0e-5, 0, -0.6e-5,
                  0, -0.6e-5, -0.6e-5, 2.0e-5, -0.6e-5, -0.6e-5]
        ien = get_ien(1, 1, 1)
        basis = getBasis(3)
        xa = nodeList(2, 2, 2, 1, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getCauchy(defE, basis[0][0], jac)
        correct = [[2000046.5004331144], [7.4999469716521787],
                   [7.4999469717103864], [0], [0], [0]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1)

    # For 3D large deformation...
    def test_sigma_3D_1Elem_Large(self):
        deform = [0, 0, 0, 2.0e-1, 0, 0, 0, -0.6e-1, 0, 2.0e-1, -0.6e-1, 0,
                  0, 0, -0.6e-1, 2.0e-1, 0, -0.6e-1,
                  0, -0.6e-1, -0.6e-1, 2.0e-1, -0.6e-1, -0.6e-1]
        ien = get_ien(1, 1, 1)
        basis = getBasis(3)
        xa = nodeList(2, 2, 2, 1, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getCauchy(defE, basis[0][0], jac)
        correct = [[25077053884.578617], [681818181.81818426],
                   [681818181.81818426], [0.0], [0], [0]]

        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1)

############################################################################

# Now the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(TestProcessUtils)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(TestFGeneration)
Suite3 = unittest.TestLoader().loadTestsFromTestCase(TestPK2Stress)
Suite4 = unittest.TestLoader().loadTestsFromTestCase(TestCauchyStressTensor)

FullSuite = unittest.TestSuite([Suite1, Suite2, Suite3, Suite4])

#SingleSuite = unittest.TestSuite()
#SingleSuite.addTest(PressurizedCylinderTest('test_accuracyPressCylinSol'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)
        
