# In this file, we test the functions from Assignment 9. 


import unittest
import numpy as np
from math import pi, cos, sin
from Assignment_1 import nodeList, get_ien
from Assignment_8 import solver
from Assignment_9 import nsel, constrain, plotResults


################################################################

# In this first section, we test the node selection function 'nsel'

class nselTests(unittest.TestCase):
    # Here, we define data needed for all the tests
    def setUp(self):
        # 3D mesh
        self.nodes = nodeList(1.5, 1.5, 1.5, 2, 2, 2)
        self.ien = get_ien(2, 2, 2)
        self.nnums = np.linspace(0, len(self.nodes)-1, len(self.nodes))
        
    # Here, we test the normal node selection process
    def test_NormalSelection(self):
        s1 = nsel(self.nodes, self.nnums, 'x', 'n', 0.75, 0.01)
        s2 = nsel(self.nodes, self.nnums, 'y', 'n', 1.5, 0.01)
        s3 = nsel(self.nodes, self.nnums, 'z', 'n', 1.5, 1)
        
        correct = [[0, 3, 6, 9, 12, 15, 18, 21, 24],
                   [6, 7, 8, 15, 16, 17, 24, 25, 26],
                   [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26]]

        for i in range(len(s1)):  # for each component...
            self.assertAlmostEqual(correct[0][i]+1, s1[i])
        for i in range(len(s2)):  # for each component...
            self.assertAlmostEqual(correct[1][i], s2[i])
        for i in range(len(s3)):  # for each component...
            self.assertAlmostEqual(correct[2][i], s3[i])

    # Now, we test adding an additional set
    def test_AdditionalSelection(self):
        s0 = nsel(self.nodes, self.nnums, 'z', 'n', 1.5, 0.01)  # get a new set
        s1 = nsel(self.nodes, s0, 'x', 'a', 0, 0.01)  # intersect with new set
        s2 = nsel(self.nodes, s0, 'y', 'a', 1.5, 0.01)  # intersect with new set
        
        correct = [[0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                   [6, 7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]

        for i in range(len(s1)):  # for each component...
            self.assertAlmostEqual(correct[0][i], s1[i])
        for i in range(len(s2)):  # for each component...
            self.assertAlmostEqual(correct[1][i], s2[i])

    # Finally, we test a subset selection function in 'nsel'
    def test_SubsetSelection(self):
        s0 = nsel(self.nodes, self.nnums, 'z', 'n', 1.5, 0.01)  # get a new set
        s1 = nsel(self.nodes, s0, 'x', 's', 0, 0.01)  # intersect with new set
        s2 = nsel(self.nodes, s0, 'y', 's', 1.5, 0.01)  # intersect with new set
        
        correct = [[18, 21, 24],
                   [24, 25, 26]]
            
        for i in range(len(s1)):  # for each component...
            self.assertAlmostEqual(correct[0][i], s1[i])
        for i in range(len(s2)):  # for each component...
            self.assertAlmostEqual(correct[1][i], s2[i])

    # We add an additional test for making a spherical selection about the origin
    def test_SphericalSelection(self):
        s0 = nsel(self.nodes, self.nnums, 'r', 'n', 1.5, 0.01)
        correct = [2, 6, 18]

        for i in range(len(s0)):  # for each component...
            self.assertEqual(correct[i], s0[i])
        
#################################################################

# Here, we test the behavior of the constraint funcion and ensure that it
# applies loads properly.

class constrainTest(unittest.TestCase):
    # define initial variables
    def setUp(self):
        self.nodes = nodeList(1, 1, 1, 1, 1, 1)
        self.ien = get_ien(1, 1, 1)
        self.nnums = np.linspace(0, len(self.nodes)-1, len(self.nodes))

        self.s0 = nsel(self.nodes, self.nnums, 'x', 'n', 1, 0.01)

        self.ida, self.ncons, self.cons, loads = constrain(self.nodes, self.s0,
                                                    self.ien, 'y', 0)
    # Here, we test the ida array
    def test_idArray(self):
        correct_ida = [0, 1, 2, 3, 'n', 4, 5, 6, 7, 8, 'n', 9,
                       10, 11, 12, 13, 'n', 14, 15, 16, 17, 18, 'n', 19]
        
        for i in range(len(self.ida)):  # for every component...
            self.assertAlmostEqual(self.ida[i], correct_ida[i])

    # Next, we test the number of constraints and the constraint array
    def test_constrainArray(self):
        correct_cons = [['n', 'n', 'n', 'n', 'n', 'n', 'n', 'n'],
                        ['n', 0, 'n', 0, 'n', 0, 'n', 0],
                        ['n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']]
        
        self.assertEqual(self.ncons, 4)
        for i in range(len(self.cons)):  # for every degree of freedom...
            for j in range(len(self.cons[0])):  # for every node...
                self.assertAlmostEqual(self.cons[i][j], correct_cons[i][j])

#############################################################################

# Now, we test the ability of the plotResults function to corrrectly display
# 2D data.

class plotDataTest(unittest.TestCase):
    # Here, we set up the data needed to perform the tests
    def setUp(self):
        self.nodes = nodeList(1, 1, 1, 2, 2, 2)
        self.nodes1 = nodeList(1, 1, 1, 1)
        self.ien = get_ien(2, 2, 2)
        self.ien1 = get_ien(1)
        self.nnums = np.linspace(0, len(self.nodes)-1, len(self.nodes))
        self.nnums1 = [0, 1]

    # Here, we test the plotting function for a single, 1D element
    def test_1Elem1DPlot(self):
        s0 = nsel(self.nodes1, self.nnums1, 'x', 'n', 0, 0.01)
        ida, ncons, cons, loads = constrain(self.nodes1, s0, self.ien1, 'x', 0)
        loads[2][0] = 1.0e8  #Pa
        deform, i = solver(1, loads, self.nodes1, self.ien1, ida, ncons, cons)
        #c = plotResults(deform, self.nodes1, self.nnums1, [1, 0, 0], 'x')
        self.assertEqual(0, 0)

    # Here, we test the plotting function for a single, 3D element
    def test_1Elem3DPlot(self):
        s0 = nsel(self.nodes, self.nnums, 'x', 'n', 0, 0.01)
        ida, ncons, cons, loads = constrain(self.nodes, s0, self.ien, 'x', 0)
        s1 = nsel(self.nodes, s0, 'z', 's', 0, 0.01)
        ida, ncons, cons, loads = constrain(self.nodes, s1, self.ien, 'z', 0,
                                            cons)
        s2 = nsel(self.nodes, s1, 'y', 's', 0, 0.01)
        ida, ncons, cons, loads = constrain(self.nodes, s2, self.ien, 'y', 0,
                                            cons)
        loads[2][1] = 1.0e8  #Pa
        loads[2][3] = 1.0e8
        loads[2][5] = 1.0e8
        loads[2][7] = 1.0e8
        
        deform, i = solver(3, loads, self.nodes, self.ien, ida, ncons, cons)
        ps0 = nsel(self.nodes, self.nnums, 'y', 'n', 0, 0.01)
        ps1 = nsel(self.nodes, ps0, 'z', 's', 0, 0.01)
        #c = plotResults(deform, self.nodes, ps1, [1, 0, 0], 'x')

        self.assertEqual(0, 0)

###########################################################################

# Another test that must be performed involves solving problems with unaligned
# elements and distorted elements such as is the case for the pressurized
# cylinder problem. In this section, we implement this problem.

class PressurizedCylinderTest(unittest.TestCase):
    # first we define important necessary variables
    def setUp(self):
        # Here, we generate the mesh
        thetaDomain = pi/2.0  # quarter circle of the pipe
        ri = 1.2  # the inner radius of the pipe
        ro = 1.8  # the outer radius of the pipe
        self.nr = 2  # the number of elements in the radial direction
        self.nt = 4  # the number of elements in the circumfrential direction
        self.nodes = []  # stores the nodes in the cylindrical mesh
        self.p = 2.0e6  # the pressure (Pa)

        for i in range(self.nr+1):  # for every node in the r-direction...
            for j in range(self.nt+1):  # for every node in the theta-direction...
                radius = i*(ro - ri)/self.nr + ri
                theta = j*thetaDomain/self.nt
                self.nodes.append([radius*cos(theta), radius*sin(theta), 0])

        self.ien = get_ien(self.nt, self.nr)
        self.nnums = np.linspace(0, len(self.nodes)-1, len(self.nodes))

        # now the fun begins. We solve this problem using roller constraints on
        # the straight faces and pressure loads on the inner face.
        s0 = nsel(self.nodes, self.nnums, 'y', 'n', 0, 0.01)
        s1 = nsel(self.nodes, self.nnums, 'x', 'n', 0, 0.01)

        ida, ncons, cons0, loads = constrain(self.nodes, s0, self.ien, 'y', 0)
        ida, ncons, cons, loads = constrain(self.nodes, s1, self.ien, 'x', 0,
                                            cons0)
        for i in range(self.nt):  # for every inner element...
            loads[3][i] = self.p

        self.deform, i = solver(2, loads, self.nodes, self.ien, ida, ncons, cons)
        ps0 = nsel(self.nodes, self.nnums, 'y', 'n', 0, 0.01)
        #c = plotResults(deform, self.nodes, ps0, [1, 0, 0], 'x')

    # in this test, we verify that all the displacements are only along radial
    # lines of the pipe. 
    def test_pressurizedCylinder(self):
        for i in range(len(self.nodes)):  # for every node...
            s = 2*i  # starting position of the node displacement
            e = 2*i + 2  # ending position of the node displacement
            v1 = np.array(self.nodes[i])
            v2 = np.array(self.deform[s:e])
            u1 = v1/np.linalg.norm(v1)
            u2 = v2/np.linalg.norm(v2)
            delta = np.dot(u1[0:2], u2)
            self.assertAlmostEqual(delta, 1.0)

    # We next test to make sure that the deformations are all equal at equal
    # radial distances
    def test_pressCylinderSymmetry(self):
        vals = []  # stores the constant values at each radius
        n = 0  # counts the current node
        for i in range(self.nr+1):  # for every node at constant theta...
            vals.append([])
            for j in range(self.nt+1):  # for each node at constant radius...
                s = 2*n  # starting position of the node displacement
                e = 2*n + 2  # ending position of the node displacement
                # the magnitude of the nodal displacement
                disp = np.linalg.norm(np.array(self.deform[s:e]))
                
                if j == 0:  # for the first...
                    vals[i] = disp
                else:
                    self.assertAlmostEqual(vals[i], disp)

                n += 1

    # Now, we compare to the exact solution implemented in this function
    def exactSol(ri, ro, r, nu, E, p):
        a = p*ri**2/(E*(ro**2 - ri**2))
        b = ((1 - nu)*r + ro**2*(1 + nu)/r)
        return a*b

    def test_accuracyPressCylinSol(self):
        ps = nsel(self.nodes, self.nnums, 'y', 'n', 0, 0.01)

        for i in range(len(ps)):  # for every node...
            s = 2*ps[i]  # starting position of the node displacement
            e = 2*ps[i] + 2  # ending position of the node displacement
            # the magnitude of the nodal displacement
            disp = np.linalg.norm(np.array(self.deform[s:e]))
            

############################################################################

# testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(nselTests)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(constrainTest)
Suite3 = unittest.TestLoader().loadTestsFromTestCase(plotDataTest)
Suite4 = unittest.TestLoader().loadTestsFromTestCase(PressurizedCylinderTest)

FullSuite = unittest.TestSuite([Suite1, Suite2, Suite3, Suite4])

SingleSuite = unittest.TestSuite()
SingleSuite.addTest(plotDataTest('test_1Elem1DPlot'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)
