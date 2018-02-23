# In this file, we test the functions and operations in 'Assignment_7' needed
# to assemble the external force vector

import unittest
from Assignment_1 import *
from Assignment_2 import *
from Assignment_5 import *
from Assignment_6 import *
from Assignment_7 import *

class BasicTest(unittest.TestCase):
    # We first test the integrand
    def test_Integrand(self):
        basis = []  # a set of all the bases
        for i in range(3):  # for each dimension...
            basis.append(getBasis(i+1))
        
        fj = 2
        jj = 0
        scale = 2
        w = 1
        faj = []
        for i in range(3):  # for each element in 'basis'...
            faj.append(integrand(basis[i][0][0], jj, fj, scale, w))

        for j in range(3):  # for each 'faj'
            for i in range(len(faj[j])):  # for every item in the array...
                if i%(j+1) == jj:
                    self.assertAlmostEqual(faj[j][i], basis[j][0][0][int(i/(j+1))][0]*fj*scale*w)
                else:
                    self.assertAlmostEqual(faj[j][i],0.0)
    
    # This function tests the Gaussian integration process
    def test_gaussInt(self):
        basis = []  # a set of all the bases
        for i in range(3):  # for each dimension...
            basis.append(getBasis(i+1))
        
        fj = 3
        xa0 = [[[0, 0, 0], [1, 0, 0]], [], []]  # 1D
        for i in range(2):   # generate the 'xa' arrays for a unit cube 
            for j in range(2):
                xa0[1].append([j, i, 0])
        for i in range(2):  # for every dimension...
            for j in range(2):
                for k in range(2):
                    xa0[2].append([k, j, i])
        scale = [0.5, 0.25, 0.125]
        w = 1
        area = []
        
        for i in range(3):  # for each dimensional basis...
            area.append([])
            for j in range(i+1):  # for each degree of freedom...
                area[-1].append(gaussInt(basis[i][0], 0, j, fj, xa0[i]))

        for i in range(3):  # for every dimension...
            for j in range(len(area[i])):  # for every degree of fredom...
                for k in range(len(area[i][j])):  # for every 
                    if k%(len(area[i])) == j:
                        self.assertAlmostEqual(area[i][j][k], fj*scale[i]*w)
                    else:
                        self.assertAlmostEqual(area[i][j][k], 0.0)


    # Next, we test integration of a traction force for a boundary
    def test_gaussIntTraction(self):
        basis = []  # a set of all the bases
        for i in range(2):  # for 2D and 3D...
            basis.append(getBasis(i+2))

        hj = 5
        xa0 = [[[0, 0, 0], [1, 0, 0]], [], []]  # 1D
        for i in range(2):   # generate the 'xa' arrays for a unit cube 
            for j in range(2):
                xa0[1].append([j, i, 0])
        for i in range(2):  # for every dimension...
            for j in range(2):
                for k in range(2):
                    xa0[2].append([k, j, i])
                    
        area = []  # stores the integrals
        jj = 0  # degree of freedom number
        s = 4  # the element region
        dim = 2  # the dimension
        correct = [0, 0, 0, 0, 2.5, 0, 2.5, 0]
        other = gaussInt(basis[dim-2][s], s, jj, hj, xa0[dim-1])
        for i in range(len(other)):  # for every component of the vector...
            self.assertAlmostEqual(other[i], correct[i])

        
######################################################################

# Now, we test the external force vector assembly and integration with other load capability
# implemented in Assignment_2

class ExtForceVecLoadAndAssemblyTest(unittest.TestCase):
    # Here, we initialize the necessary load data for the test
    def setUp(self):
        cons2, loads2 = load_and_cons(1, 4, 2)  # one 2D element
        loads2[0][0] = [-1, 0, 0]  # body force
        loads2[2][0] = [1, 0, 0]  # traction force (right face)
        loads2[4][0] = 3
        self.loads2 = loads2
        self.b2 = getBasis(2)

        self.nodes2 = nodeList(1, 1, 1, 1, 1)
        self.ien2 = get_ien(1, 1)

        cons3, loads3 = load_and_cons(1, 8, 3)  # one 2D element
        loads3[0][0] = [0, 0, -4]  # body force
        loads3[2][0] = [1, 0, 0]  # traction force (right face)
        loads3[4][0] = 8
        loads3[6][0] = [2, 0, 0]
        self.loads3 = loads3
        self.b3 = getBasis(3)

        self.nodes3 = nodeList(1, 1, 1, 1, 1, 1)
        self.ien3 = get_ien(1, 1, 1)
        
    # Next, we test the force vector generator ability to process loads
    def test_ExtForceVecLoadProcessing2D(self):
        correct = [-0.25,  0,    0.25,  0,   -0.25,  1.5,   0.25,  1.5]
        for i in range(len(correct)):  # for every component of 'correct'...
            self.assertAlmostEqual(getExtForceVec(self.loads2, self.b2, self.nodes2, self.ien2)[i], correct[i])

    # Next, we perform this test for a 3D case
    def test_ExtForceVecLoadProcess3D(self):
        correct = [ 0, 0, -0.5, 0.25, 0, -0.5, 0, 2, -0.5, 0.25, 2, -0.5, 0.5,
                    0, -0.5, 0.75, 0, -0.5, 0.5, 2, -0.5, 0.75, 2, -0.5]
        answer = getExtForceVec(self.loads3, self.b3, self.nodes3, self.ien3)
        for i in range(len(correct)):  # for every component of 'correct'...
            self.assertAlmostEqual(answer[i], correct[i])
        

#########################################################################

Suite1 = unittest.TestLoader().loadTestsFromTestCase(BasicTest)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(ExtForceVecLoadAndAssemblyTest)
FullSuite = unittest.TestSuite([Suite1, Suite2])

unittest.TextTestRunner(verbosity=2).run(FullSuite)



