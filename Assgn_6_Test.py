# In this file, we implement the test cases used to verify the code being developed for ME 607

import unittest
import numpy as np 
from Assignment_6 import *
from Assignment_5 import *
from Assignment_4 import *
from Assignment_2 import load_and_cons
from Assignment_1 import nodeList, get_ien

        
####################################################################################

# This class implements some tests on the stress and strain computations in the code.
# It verifies basic mechanics capability in the code.

class MechanicsTest(unittest.TestCase):
        # next, we test the 'strainVec' function. This test shows that all simple,
        # one-dimensional strains are correctly calculated
        def setUp(self):  # define basic, needed parameters.
                self.b1 = getBasis(1)
                self.numD = 1
                self.strain = 0.1
                self.deform1 = [0, self.strain]  # deform. of two nodes in 1D
                self.ien1 = get_ien(1)
                self.enum1 = 0
                self.xa1 = [[0, 0, 0], [1, 0, 0]]  # real coordinats of the nodes
                self.Bmats1, self.scale1 = getBandScale(self.numD, self.b1, 0,
                                                        self.xa1)
                self.b2 = getBasis(2)
                # deformation of four nodes in 2D
                self.deform2 = [0, 0, 0.1, 0, 0, -0.03, 0.1, -0.03]  
                self.ien2 = get_ien(1, 1)
                self.enum2 = 0
                self.i2 = 0
                self.xa2 = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
                self.Bmats2, scale2 = getBandScale(2, self.b2, 0, self.xa2)

                self.b3 = getBasis(3)
                # deformation of four nodes in 2D
                self.deform3 = [0, 0, 0, 0.1, 0, 0,
                                0, -0.03, 0, 0.1, -0.03, 0,
                                0, 0, -0.03, 0.1, 0, -0.03,
                                0, -0.03, -0.03, 0.1, -0.03, -0.03]  
                self.ien3 = get_ien(1, 1, 1)
                self.enum3 = 0
                self.i3 = 0
                self.xa3 = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
                self.Bmats3, scale3 = getBandScale(3, self.b3, 0, self.xa3)

        # Here, we test strain in 1D
        def test_1DstrainCalcs(self):
                strain = strainVec(1, self.enum1, self.deform1, self.ien1,
                                   self.Bmats1)
                
                self.assertAlmostEqual(strain[0][0], self.strain)

        # This function tests simple strain in 2D
        def test_2DstrainCalcs(self):
                strain = strainVec(2, 0, self.deform2, self.ien2, self.Bmats2)
                correct = [0.1, -0.03, 0]  # correct answer
                for i in range(len(strain)):  # for every component...
                        self.assertAlmostEqual(strain[i], correct[i])

        # Here, we test for strain in 3D
        def test_3DstrainCalcs(self):
                strain = strainVec(3, 0, self.deform3, self.ien3, self.Bmats3)
                correct = [0.1, -0.03, -0.03, 0, 0, 0]  # correct answer
                for i in range(len(strain)):  # for every component...
                        self.assertAlmostEqual(strain[i], correct[i])

        # This function tests 1D stress
        def test_1Dstress(self):
                strain = strainVec(1, self.enum1, self.deform1, self.ien1,
                                   self.Bmats1)
                stress = stressVec(1, strain)
                correct = 2e10
                self.assertAlmostEqual(stress, correct)

        # This function tests 2D stress
        def test_2Dstress(self):
                strain = strainVec(2, 0, self.deform2, self.ien2, self.Bmats2)
                stress = stressVec(2, strain)
                correct = [2e10, 0, 0]
                for i in range(len(stress)):  # for each component...
                        self.assertAlmostEqual(stress[i], correct[i], 5)

        # Here, we test the 3D stress calculations
        def test_3Dstress(self):
                strain = strainVec(3, 0, self.deform3, self.ien3, self.Bmats3)
                stress = stressVec(3, strain)
                correct = [2e10, 0, 0, 0, 0, 0]
                for i in range(len(stress)):  # for each component...
                        self.assertAlmostEqual(stress[i], correct[i], 5)

        # Next, we test stress and strain rigorously in three dimensions.
        def test_strainVecSimpleNormal(self):
                nodes = nodeList(1, 1, 1, 1, 1, 1)  # single 3D element
                ien = get_ien(1, 1, 1)
                e = 0  # element number 
                numD = numDims(1, 1, 1)  # the number of dimensions
                basis = getBasis(numD)
                xa = getXaArray(e, nodes, ien)  # get the global coordinates of the element nodes
                
                # In this section, we test normal strain computations where 'epsilon = 0.12' in each direction 
                for m in range(3):  # for every direction...
                        deform = numD*len(nodes)*[0.0]  # the deformation vector
                        for i in range(len(nodes)):  # for every element node...
                                for j in range(numD):  # for every dimension
                                        if j == m:  # for the x-direction...
                                                deform[numD*i + j] = nodes[i][j]*0.12  # add a strain 
                        
                        for k in range(len(basis[0])):  # for every integration point...
                                Bmats, scale = getBandScale(numD, basis, k, xa)
                        
                                strain = strainVec(numD, e, deform, ien, Bmats)
                                strain0 = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
                                strain0[m] = 0.12
                        
                                for i in range(len(strain)):
                                        self.assertAlmostEqual(strain[i], strain0[i], places=4)
        
        def test_strainVecShear(self):
                nodes = nodeList(1, 1, 1, 1, 1, 1)  # single 3D element
                ien = get_ien(1, 1, 1)
                e = 0  # element number 
                numD = numDims(1, 1, 1)  # the number of dimensions
                basis = getBasis(numD)
                xa = getXaArray(e, nodes, ien)  # get the global coordinates of the element nodes
                
                # In this section, we test strains 'epsilon_i,i+1' direction where i+1 can loop around to zero.
                for m in range(numD):  # for every direction...
                        deform = numD*len(nodes)*[0.0]  # the deformation vector
                        for i in range(len(nodes)):  # for every element node...
                                for j in range(numD):  # for every dimension
                                        if j == m:  # for the x-direction...
                                                deform[numD*i + (j + 1)%numD] = nodes[i][j]*0.12  # add a strain 
                        
                        for k in range(len(basis[0])):  # for every integration point...
                                Bmats, scale = getBandScale(numD, basis, k, xa)
                        
                                strain = strainVec(numD, e, deform, ien, Bmats)
                                strain0 = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
                                strain0[3 + (2 + m)%numD] = 0.12
                                
                                for i in range(len(strain)):
                                        self.assertAlmostEqual(strain[i], strain0[i], places=4)
                
        def test_stressVecSimpleNormal(self):
                nodes = nodeList(1, 1, 1, 1, 1, 1)  # single 3D element
                ien = get_ien(1, 1, 1)
                e = 0  # element number 
                numD = numDims(1, 1, 1)  # the number of dimensions
                basis = getBasis(numD)
                xa = getXaArray(e, nodes, ien)  # get the global coordinates of the element nodes
                
                for m in range(3):  # for every direction...
                        deform = numD*len(nodes)*[0.0]  # the deformation vector
                        for i in range(len(nodes)):  # for every element node...
                                for j in range(numD):  # for every dimension
                                        if j == m:  # for the x-direction...
                                                deform[numD*i + j] = nodes[i][j]*0.12  # add a strain 
                        
                        for k in range(len(basis[0])):  # for every integration point...
                                Bmats, scale = getBandScale(numD, basis, k, xa)
                        
                                strain = strainVec(numD, e, deform, ien, Bmats)
                                stress = stressVec(numD, strain)
                                ld = 0.12*(200*10**9)*0.3/((1 + 0.3)*(1 - 2*0.3))
                                mu = 0.12*(200*10**9)/((2*(1 + 0.3)))
                                stress0 = np.array([ld, ld, ld, 0.0, 0.0, 0.0])
                                stress0[m] = ld + 2*mu 
                                
                                for i in range(len(strain)):
                                        self.assertAlmostEqual(stress[i], stress0[i], places=4)
        
        # Here, we test the ability of the routine to correctly find the stress vector in Voigt notation
        def test_stressVecShear(self):
                nodes = nodeList(1, 1, 1, 1, 1, 1)  # single 3D element
                ien = get_ien(1, 1, 1)
                e = 0  # element number 
                numD = numDims(1, 1, 1)  # the number of dimensions
                basis = getBasis(numD)
                xa = getXaArray(e, nodes, ien)  # get the global coordinates of the element nodes
                
                for m in range(numD):  # for every direction...
                        deform = numD*len(nodes)*[0.0]  # the deformation vector
                        for i in range(len(nodes)):  # for every element node...
                                for j in range(numD):  # for every dimension
                                        if j == m:  # for the x-direction...
                                                deform[numD*i + (j + 1)%numD] = nodes[i][j]*0.12  # add a strain 
                        
                        for k in range(len(basis[0])):  # for every integration point...
                                Bmats, scale = getBandScale(numD, basis, k, xa)
                        
                                strain = strainVec(numD, e, deform, ien, Bmats)
                                stress = stressVec(numD, strain)
                                stress0 = np.array(6*[0.0])
                                stress0[3 + (2 + m)%numD] = 200*10**9/(2*(1+0.3))*0.12
                                
                                for i in range(len(strain)):
                                        self.assertAlmostEqual(stress[i], stress0[i], places=4)

############################################################################

# Another important test that must be run will determine if the internal
# force vector is assembled properly.

class IntForceVectorFuncTest(unittest.TestCase):
        # Here, we initialize the variables needed to test the function
        def setUp(self):
                self.b1 = getBasis(1)
                self.deform1 = [0, 0.1]  # deformation of two nodes in 1D
                self.ien1 = get_ien(1)
                self.enum1 = 0
                self.i1 = 0
                self.xa1 = [[0, 0, 0], [1, 0, 0]]

                self.b2 = getBasis(2)
                # deformation of four nodes in 2D
                self.deform2 = [0, 0, 0.1, 0, 0, -0.03, 0.1, -0.03]  
                self.ien2 = get_ien(1, 1)
                self.enum2 = 0
                self.i2 = 0
                self.xa2 = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]

        # Next, we test the function evaluations in a few dimenions
        def test_funcOutput(self):
                answer1 = func(self.deform1, self.b1, self.ien1, self.enum1,
                              self.i1, self.xa1)
                correct1 = [-1e10, 1e10]
                
                for i in range(len(answer1)):  # for every component of 'answer'
                        self.assertAlmostEqual(answer1[i], correct1[i])

        # Here, we test the function outputs in 2D
        def test_funcOutput2D(self):
                answer2 = func(self.deform2, self.b2, self.ien2, self.enum2,
                              self.i2, self.xa2)
                correct2 = [[ -3.94337567e+09],
                            [  1.88034805e-07],
                            [  3.94337567e+09],
                            [  5.03837741e-08],
                            [ -1.05662433e+09],
                            [ -1.88034805e-07],
                            [  1.05662433e+09],
                            [ -5.03837741e-08]]
                
                for i in range(len(answer2)):  # for every component of 'answer'
                        self.assertAlmostEqual(answer2[i][0]/correct2[i][0], 1, 5)

#####################################################################

# Now, we must test the internal force vector assembly process to ensure that it functions properly.

class IntForceVecAssemblyTest(unittest.TestCase):
        # Here, we initialize the necessary variables for the test case
        def setUp(self):
                self.nodes1 = nodeList(1, 1, 1, 1)  # for one dimension...
                self.ien1 = get_ien(1)
                self.ndim1 = 1
                self.numE1 = 1
                self.deform1 = [0, 0.1]

                cons, loads = load_and_cons(self.numE1, len(self.nodes1), 1)
                cons[0][0] = 0  # set arbitrary constraints
                self.ida1, self.ncons1 = getIDArray(cons)

                self.nodes2 = nodeList(1, 1, 1, 1, 1)  # for one dimension...
                self.ien2 = get_ien(1, 1)
                self.ndim2 = 2
                self.numE2 = 1
                self.deform2 = [0, 0, 0.1, 0, 0, -0.03, 0.1, -0.03]

                cons, loads = load_and_cons(self.numE2, len(self.nodes2), 2)  # 2 dimensions
                cons[0][0] = 0
                cons[1][0] = 0
                cons[0][3] = 0
                cons[1][3] = 0
                self.ida2, self.ncons2 = getIDArray(cons)

                self.nodes3 = nodeList(1, 1, 1, 1, 1, 1)  # for one dimension...
                self.ien3 = get_ien(1, 1, 1)
                self.ndim3 = 3
                self.numE3 = 1
                self.deform3 = [0, 0, 0, 0.1, 0, 0, 0, -0.03, 0, 0.1, -0.03, 0,
                                0, 0, -0.03, 0.1, 0, -0.03, 0, -0.03, -0.03, 0.1,
                                -0.03, -0.03]

                cons, loads = load_and_cons(self.numE3, len(self.nodes3), 3)  # 2 dimensions
                cons[0][0] = 0
                cons[1][0] = 0
                cons[2][0] = 0
                cons[0][3] = 0
                cons[1][3] = 0
                cons[2][3] = 0
                cons[0][6] = 0
                self.ida3, self.ncons3 = getIDArray(cons)
        
        # next, we test the force vector stackup process for one dimension
        def test_intForceVecOutput1D(self):
                Fint = intForceVec(self.nodes1, self.ien1, self.ida1, self.ncons1,self.ndim1,
                                   self.numE1, self.deform1)
                correct = [2e10]
                
                for i in range(len(Fint)):  # for every element component...
                        self.assertAlmostEqual(Fint[i]/correct[i], 1, 4)

        # This function attempts the same verification in 2D
        def test_intForceVecOutput2D(self):
                Fint = intForceVec(self.nodes2, self.ien2, self.ida2, self.ncons2, self.ndim2,
                                   self.numE2, self.deform2)
                correct = [1e10, 0, -1e10, 0]
                
                for i in range(len(Fint)):  # for every element component...
                        self.assertAlmostEqual((Fint[i]+1)/(correct[i]+1), 1, 4)

        # Lastly, we test the assembly process in 3D
        def test_intForceVecOutput3D(self):
                Fint = intForceVec(self.nodes3, self.ien3, self.ida3, self.ncons3, self.ndim3,
                                   self.numE3, self.deform3)
                correct = [5e9, 0, 0, -5e9, 0, 0, -5e9, 0, 0,
                           5e9, 0, 0, 0, 0, 5e9, 0, 0]
                
                for i in range(len(Fint)):  # for every element component...
                        self.assertAlmostEqual((Fint[i]+1)/(correct[i]+1), 1, 4)

################################################################################

Suite1 = unittest.TestLoader().loadTestsFromTestCase(MechanicsTest)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(IntForceVectorFuncTest)
Suite3 = unittest.TestLoader().loadTestsFromTestCase(IntForceVecAssemblyTest)

FullSuite = unittest.TestSuite([Suite1, Suite2, Suite3])

#singleTestSuite = unittest.TestSuite()
#singleTestSuite.addTest(MechanicsTest('IntForceVecAssemblyTest'))

unittest.TextTestRunner(verbosity=2).run(Suite3)

