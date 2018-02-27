# This file implements all necessary tests for the code in Assignment_8


import unittest
from Assignment_1 import *
from Assignment_2 import *
from Assignment_4 import getBasis
from Assignment_6 import getBandScale, getStiff
from Assignment_8 import getEnergyDensity, gaussIntKMat, getStiffMatrix


###################################################################

# The first test to implement will determine if the stiffnes matrix
# integrand in 'getEnergyDensity' is working properly.

class EnergyDensityTest(unittest.TestCase):
    # Here, we perform a simple test on the integrand.
    def test_getEnDen1D(self):
        dims = 1
        intpt = 0
        xa = [[0, 0, 0], [1, 0, 0]]
        basis = getBasis(dims)
        Bmats, scale = getBandScale(dims, basis, intpt, xa)
        D = getStiff(dims)  # the 'D' matrix

        # Next, we test for all combinations of 'a' and 'b'
        for i in range(2):  # for every 'a' or local element #...
            for j in range(2):  # for every 'b'...
                kab = getEnergyDensity(D, Bmats[i], Bmats[j])
                self.assertAlmostEqual(kab[0][0], (-1)**(i+j)*200e9)

    # next, we test in 2D
    def test_getEnDen2D(self):
        dims = 2
        intpt = 0
        xa = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        a = 0
        b = 0
        basis = getBasis(dims)
        Bmats, scale = getBandScale(dims, basis, intpt, xa)
        D = getStiff(dims)  # the 'D' matrix


        kab = getEnergyDensity(D, Bmats[a], Bmats[b])
        
        # come back to this answer and check
        correct = [[1.84551963e+11, 8.88583526e+10], [8.88583526e+10, 1.84551963e+11]]
        for i in range(len(kab)):  # for every row...
            for j in range(len(kab)):  # for every column...
                self.assertAlmostEqual(kab[i][j]/correct[i][j], 1)

    # finally, we test the 3D outputs of the integrand
    def test_getEnDen3D(self):
        dims = 3
        intpt = 0
        xa = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
              [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        a = 0
        b = 0
        basis = getBasis(dims)
        Bmats, scale = getBandScale(dims, basis, intpt, xa)
        D = getStiff(dims)  # the 'D' matrix

        kab = getEnergyDensity(D, Bmats[a], Bmats[b])
        
        # come back to this answer and check
        correct = [[  1.63686149e+11,   7.44027950e+10,   7.44027950e+10],
                   [  7.44027950e+10,   1.63686149e+11,   7.44027950e+10],
                   [  7.44027950e+10,   7.44027950e+10,   1.63686149e+11]]
        
        for i in range(len(kab)):  # for every row...
            for j in range(len(kab)):  # for every column...
                self.assertAlmostEqual(kab[i][j]/correct[i][j], 1)

###################################################################

# Here, we perform testing on the element stiffness matrix
class ElementStiffMatrixTest(unittest.TestCase):
    # Here, we test the integration in 1D
    def test_1DElemStiffMat(self):
        dims = 1  # one dimension
        xa = [[0, 0, 0], [1, 0, 0]]  # the real coordinates of the elem. nodes
        emat = gaussIntKMat(dims, xa)
        
        for i in range(len(emat)):  # for every row...
            for j in range(len(emat[0])):  # for every column...
                self.assertAlmostEqual(emat[i][j], (-1)**(i+j)*2e11)

    # Next, we test the 2D case. Currently, we only test for symmetry and that
    # the matix is the correct size
    def test_2DElemStiffMat(self):
        dims = 2
        xa = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        emat = gaussIntKMat(dims, xa)
        
        for i in range(len(emat)):  # for every row...
            for j in range(len(emat[0])):  # for every column...
                # at the very least, the matrix should be symmetric
                self.assertAlmostEqual(emat[i][j], emat[j][i])
                self.assertEqual(len(emat), 8)

    # Finally, we test the 3D variant for symmetry and the correct size
    def test_3DElemStiffMat(self):
        dims = 3
        xa = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
              [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        emat = gaussIntKMat(dims, xa)
        
        for i in range(len(emat)):  # for every row...
            for j in range(len(emat[0])):  # for every column...
                # at the very least, the matrix should be symmetric
                self.assertAlmostEqual(emat[i][j]/emat[j][i], 1)
                self.assertEqual(len(emat), 24)

################################################################

# This class intends to test the stiffness matrix assembly process

class StiffMatrixAssemblyTest(unittest.TestCase):
    # Here, we test a 1D case of the problem
    def test_1DstiffMat(self):
        enum = 2  # number of elements
        nodes = nodeList(1, 1, 1, enum)
        ien = get_ien(enum)

        cons, loads = load_and_cons(enum, len(nodes), 1)
        #cons[0][0] = 0  # set arbitrary constraints
        #cons[0][2] = 0.1
        ida, ncons = getIDArray(cons)
        
        kmat = getStiffMatrix(nodes, ien, ida, ncons)
        
        correct = [[4e11, -4e11, 0], [-4e11, 8e11, -4e11],
                           [0, -4e11, 4e11]]
        self.assertEqual(len(kmat), len(nodes) - ncons)  # checks number of rows
        self.assertEqual(len(kmat[0]), len(nodes) - ncons)  # checks row size

        for i in range(len(kmat)):  # for every row...
                for j in range(len(kmat[0])):  # for every column...
                                self.assertAlmostEqual((kmat[i][j]+1)/(correct[i][j]+1), 1)

    # Next, we test the 2D case
    def test_2DstiffMat(self):
        enum = 1  # number of elements
        nodes = nodeList(1, 1, 1, enum, 1)
        ien = get_ien(enum, 1)
        
        cons, loads = load_and_cons(enum, len(nodes), 2)  # 2 dimensions
        #cons[0][0] = 0
        #cons[1][0] = 0
        #cons[0][3] = 0
        #cons[1][3] = 0
        ida, ncons = getIDArray(cons)
        
        kmat = getStiffMatrix(nodes, ien, ida, ncons)
        
        self.assertEqual(len(kmat), 2*len(nodes) - ncons)  # checks number of rows
        self.assertEqual(len(kmat[0]), 2*len(nodes) - ncons)  # checks row size

        for i in range(len(kmat)):  # for every row...
                for j in range(len(kmat[0])):  # for every column...
                        self.assertAlmostEqual(kmat[i][j], kmat[j][i])
    
    # Finally, we test in 3 dimensions
    def test_3DstiffMat(self):
        enum = 2  # number of elements
        nodes = nodeList(1, 1, 1, enum, 1, 1)
        ien = get_ien(enum, 1, 1)
        
        cons, loads = load_and_cons(enum, len(nodes), 3)  # 2 dimensions
        cons[0][0] = 0
        cons[1][0] = 0
        cons[2][0] = 0
        cons[0][3] = 0
        cons[1][3] = 0
        cons[2][3] = 0
        cons[0][6] = 0
        ida, ncons = getIDArray(cons)

        kmat = getStiffMatrix(nodes, ien, ida, ncons)

        self.assertEqual(len(kmat), 3*len(nodes) - ncons)  # checks number of rows
        self.assertEqual(len(kmat[0]), 3*len(nodes) - ncons)  # checks row
        
    # These tests allow us to conclude that the output matrix is of the appropriate size

################################################################

# Now the testing.
        
Suite1 = unittest.TestLoader().loadTestsFromTestCase(EnergyDensityTest)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(ElementStiffMatrixTest)
Suite3 = unittest.TestLoader().loadTestsFromTestCase(StiffMatrixAssemblyTest)

SingleSuite = unittest.TestSuite()
SingleSuite.addTest(StiffMatrixAssemblyTest('test_2DstiffMat'))

FullSuite = unittest.TestSuite([Suite1, Suite2, Suite3])

unittest.TextTestRunner(verbosity=2).run(SingleSuite)
