# In this file, we test the functionality needed to make Assignment_5 function
# properly.

import unittest
from Assignment_4 import *
from Assignment_5 import *

#####################################

# This class tests geometric functions contained in 'Assignment_5'. Important
# sub-tests in this area include parent domain to real translations, jacobians,
# integral change of variable scalings, boundary normals, real derivatives of
# the basis function 

class IntegralComponentTest(unittest.TestCase):
    # first, we test the function 'posAndJac'
    def test_posAndJac1D(self):
        basis1 = getBasis(1)
        xa = [[0, 0, 0], [1, 0, 0]]  # 'xa' array of two points (3D domain)
        x1, jac1 = posAndJac(basis1[0][0], xa)
        x2, jac2 = posAndJac(basis1[0][1], xa)
        x0 = [0.211325, 0, 0]  # the expected value 
        jac0 = [0.5, 0, 0]
        x00 = [1.0 - x0[0], 0, 0]
        jac00 = [0.5, 0, 0]
        
        for i in range(3):  # for every component of the position 'x'...
            self.assertAlmostEqual(x1[i], x0[i], places=4)
            self.assertAlmostEqual(jac1[i][0], jac0[i], places=4)
        
        for i in range(3):
            self.assertAlmostEqual(x2[i], x00[i], places=4)
            self.assertAlmostEqual(jac2[i][0], jac00[i], places=4)
    
    
    # This function performs the same tests but for the 2D cases
    def test_posAndJac2D(self):
        basis2 = getBasis(2)
        xa = [[1, 1, 0], [2, 1, 0], [1, 2, 0], [2, 2, 0]]
        x = []  # a vector of all the calculated coordinates [integration point #][dimension x, y, z]
        jac = []  # a vector of the jacobians for every integration point [int. pt #][jacobian 3x2]
        
        for i in range(len(basis2[0])):  # for every integration point...
            xi, jaci = posAndJac(basis2[0][i], xa)
            x.append(xi)
            jac.append(jaci)
        
        d = 0.211325  # constant 
        sets = [d, 1-d]
        x0 = []  # stores the true x-coordinates 
        jac0 = [[0.5, 0], [0, 0.5], [0, 0]]  # stores the true jacobian
        for i in range(2):
            for j in range(2):
                # get the real values for the real integration point coordinates 
                x0.append([1 + sets[j], 1 + sets[i], 0])  
        
        for i in range(len(x0)):  # for every integration point...
            for j in range(3):  # for every component of the vector...
                self.assertAlmostEqual(x[i][j], x0[i][j], places=4)
        
        for i in range(len(jac)):  # for every jacobian...
            for j in range(3):  # for every row of the jacobian...
                for k in range(2):  # for every column of the jacobian...
                    self.assertAlmostEqual(jac[i][j][k], jac0[j][k], places=4)

    def test_posAndJac3D(self):
        basis3 = getBasis(3)
        xa = [[1, 1, 0], [2, 1, 0], [1, 2, 0], [2, 2, 0], [1, 1, 1],
              [2, 1, 1], [1, 2, 1], [2, 2, 1]]
        x = []  # a vector of all the calculated coordinates [integration point #][dimension x, y, z]
        jac = []  # a vector of the jacobians for every integration point [int. pt #][jacobian 3x2]
        
        for i in range(len(basis3[0])):  # for every integration point...
            xi, jaci = posAndJac(basis3[0][i], xa)
            x.append(xi)
            jac.append(jaci)
        
        d = 0.211325  # constant 
        sets = [d, 1-d]
        x0 = []  # stores the true x-coordinates 
        jac0 = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]  # stores the true jacobian
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # get the real values for the real integration point coordinates 
                    x0.append([1 + sets[k], 1 + sets[j], sets[i]])  
        
        for i in range(len(x0)):  # for every integration point...
            for j in range(3):  # for every component of the vector...
                self.assertAlmostEqual(x[i][j], x0[i][j], places=4)
        
        for i in range(len(jac)):  # for every jacobian...
            for j in range(3):  # for every row of the jacobian...
                for k in range(2):  # for every column of the jacobian...
                    self.assertAlmostEqual(jac[i][j][k], jac0[j][k], places=4)

###############################################################################

# The next tests that must be performed focus on the 'scaling' function

class ScalingTests(unittest.TestCase):
    # Here, we set up the problem by getting all the relevant bases...
    def setUp(self):
        bases = []
        jac0 = []
        # element of side lengths 2x1.5x1 (x, y, z)
        xa = [[1, 1, 1], [3, 1, 1], [1, 2.5, 1], [3, 2.5, 1],
              [1, 1, 2], [3, 1, 2], [1, 2.5, 2], [3, 2.5, 2]]
        for i in range(3):  # for every dimension...
            bases.append(getBasis(i+1))
            xi, jaci = posAndJac(bases[i][0][0], xa)
            jac0.append(jaci)

        self.jac = jac0
    
    # first, we test the 1D scaling
    def test_scaling1D(self):
        self.assertAlmostEqual(scaling(0, self.jac[0]), 1)
    def test_scaling2D(self):
        self.assertAlmostEqual(scaling(0, self.jac[1]), 0.75)
    def test_scaling3D(self):
        self.assertAlmostEqual(scaling(0, self.jac[2]), 0.375)

    # now, we test surface integrals
    def test_scaling1D_1(self):
        self.assertAlmostEqual(scaling(1, self.jac[0]), 1)
    def test_scaling2D_side_1(self):
        self.assertAlmostEqual(scaling(1, self.jac[1]), 0.75)
    def test_scaling2D_side_3(self):
        self.assertAlmostEqual(scaling(3, self.jac[1]), 1)
    def test_scaling3D_side_1(self):
        self.assertAlmostEqual(scaling(1, self.jac[2]), 0.375)
    def test_scaling3D_side_4(self):
        self.assertAlmostEqual(scaling(4, self.jac[2]), 0.5)
    def test_scaling3D_side_6(self):
        self.assertAlmostEqual(scaling(6, self.jac[2]), 0.75)
        
############################################################

# Here, we test the real basis functions

class realNTest(unittest.TestCase):
    # Here, we set up the problem by getting all the relevant bases...
    def setUp(self):
        bases0 = []
        jac0 = []

        # element of side lengths 2x1.5x1 (x, y, z)
        xa = [[1, 1, 1], [3, 1, 1], [1, 2.5, 1], [3, 2.5, 1],
              [1, 1, 2], [3, 1, 2], [1, 2.5, 2], [3, 2.5, 2]]
        
        for i in range(3):  # for every dimension...
            bases0.append(getBasis(i+1, 0.5))
            xi, jaci = posAndJac(bases0[i][0][0], xa)
            jac0.append(jaci)

        self.bases = bases0
        self.jac = jac0

    # Here, we test a 1D output
    def test_realN_1D(self):
        dNdx = realN(self.bases[0][0][0], 0, self.jac[0])
        correct1D = [-0.5]

        for i in range(len(dNdx)):
            self.assertAlmostEqual(dNdx[i], correct1D[i])
    
    # Here, we test a single 2D output
    def test_realN_2D(self):
        dNdx = realN(self.bases[1][0][0], 0, self.jac[1])
        correct2D = [-0.375, -0.5]
        
        for i in range(len(dNdx)):
            self.assertAlmostEqual(dNdx[i], correct2D[i])

        

#######################################################

Suite1 = unittest.TestLoader().loadTestsFromTestCase(IntegralComponentTest)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(ScalingTests)
Suite3 = unittest.TestLoader().loadTestsFromTestCase(realNTest)
FullSuite = unittest.TestSuite([Suite1, Suite2, Suite3])

unittest.TextTestRunner(verbosity=2).run(FullSuite)
