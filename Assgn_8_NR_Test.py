# This file tests the netwton-raphson algorithm discussed in the notes


import unittest
import numpy as np
from Assignment_1 import nodeList, get_ien, getIDArray
from Assignment_2 import load_and_cons
from Assignment_6 import getStiff
from Assignment_8 import getFullDVec, solver


######################################################################

# This class tests the operation of the solver using a single element
# problem as described in Assignment 8. We also perform tests on 2D elements
# using traction and displacement loads to ensure that 1D elements function
# properly

class SolverTest1D(unittest.TestCase):
    # In this first function, we test the solver written in Assignment_8
    def setUp(self):
        self.numD = [1, 2, 3]  # the number of problem dimensions
        enum = [1, 2]  # number of elements
        self.nodes1 = nodeList(1, 1, 1, enum[0])
        self.ien1 = get_ien(enum[0])
        self.cons1, self.load1 = load_and_cons(enum[0], len(self.nodes1),
                                               self.numD[0])
        self.cons1[0][0] = 0.0
        self.load1[2][0] = [2.0e6, 0, 0]  # load to right end
        self.ida1, self.ncons1 = getIDArray(self.cons1)

        # Here, we set up inputs for a two-element problem with a traction,
        # displacement or pressure load. First, the traction load case.
        self.nodes2 = nodeList(1, 1, 1, enum[1])
        self.ien2 = get_ien(enum[1])
        self.cons2, self.load2 = load_and_cons(enum[1], len(self.nodes2),
                                               self.numD[0])
        self.cons2[0][0] = 0.0
        self.load2[2][1] = [2.0e6, 0, 0]  # load to right end
        self.ida2, self.ncons2 = getIDArray(self.cons2)

        # for the displacement load sub-case...
        self.cons3, self.load3 = load_and_cons(enum[1], len(self.nodes2),
                                               self.numD[0])
        self.cons3[0][0] = 0.0
        self.cons3[0][2] = 1.0e-5
        self.ida3, self.ncons3 = getIDArray(self.cons3)

        # now, we test the pressure load sub-case.
        self.cons4, self.load4 = load_and_cons(enum[1], len(self.nodes2),
                                               self.numD[0])
        self.cons4[0][0] = 0.0
        self.load4[2][1] = 2.0e6
        self.ida4, self.ncons4 = getIDArray(self.cons4)
        
    # Another important test that we can run handles the merging process
    # between the partial deformation vector and the complete version.
    def test_deformationVectorMerge(self):
        deform = [0.1]  # initialize the deformation vector
        result = getFullDVec(self.ida1, deform, self.cons1)
        correct = [0.0, 0.1]

        for i in range(len(result)):  # for every component of the answer...
            self.assertAlmostEqual(correct[i], result[i])

    # Here, we test the 1D solution process for a single element and traction
    def test_solver_1D1Elem(self):
        result, steps = solver(self.numD[0], self.load1, self.nodes1, self.ien1,
                               self.ida1, self.ncons1, self.cons1)
        
        correct = [0.0, self.load1[2][0][0]/(200e9)]
        
        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual((result[i] + 1)/(correct[i] + 1), 1)

    # The next test to run involves applying a traction load to two 1D elements
    def test_solver1D2ElemTrac(self):
        result, steps = solver(self.numD[0], self.load2, self.nodes2, self.ien2,
                               self.ida2, self.ncons2, self.cons2)
        
        correct = [0.0, 0.5*self.load1[2][0][0]/(200e9),
                   self.load1[2][0][0]/(200e9)]
        
        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual((result[i] + 1)/(correct[i] + 1), 1)

    # Now, we perform the same test but with a perscribed displacement on both
    # ends of the two element, 1D problem.
    def test_solver1D2ElemDisp(self):
        result, steps = solver(self.numD[0], self.load3, self.nodes2, self.ien2,
                               self.ida3, self.ncons3, self.cons3)
        
        correct = [0.0, 5.0e-6, 1.0e-5]
        
        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual((result[i] + 1)/(correct[i] + 1), 1)

    # Finally, we test the pressure load case in a 1D problem
    def test_solver1D2ElemPressure(self):
        result, steps = solver(self.numD[0], self.load4, self.nodes2, self.ien2,
                               self.ida4, self.ncons4, self.cons4)
        
        correct = [0.0, 5.0e-6, 1.0e-5]
        
        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual((result[i] + 1)/(correct[i] + 1), 1)
    
###############################################################################

# In this class, we implement testing on 2D elements to ensure that they respond
# properly to different displacement, traction and pressure loads

class SolverTest2D(unittest.TestCase):
    # Here, we initialize all the variables and arrays that will be needed in
    # the analysis
    def setUp(self):
        self.numD = 2  # the number of problem dimensions
        enum = [1, 2]  # number of elements
        self.nodes1 = nodeList(1, 1, 1, enum[0], enum[0])
        self.ien1 = get_ien(enum[0], enum[0])
        self.cons1, self.load1 = load_and_cons(enum[0], len(self.nodes1),
                                               self.numD)
        self.cons1[0][0] = 0.0
        self.cons1[1][0] = 0.0
        self.cons1[0][2] = 0.0
        
        self.load1[2][0] = [2.0e9, 0, 0]  # load to right end
        self.ida1, self.ncons1 = getIDArray(self.cons1)

        # For the pressure load...
        self.cons2, self.load2 = load_and_cons(enum[0], len(self.nodes1),
                                               self.numD)
        self.cons2[0][0] = 0.0
        self.cons2[1][0] = 0.0
        self.cons2[1][1] = 0.0
        
        self.load2[4][0] = 2.0e6  # pressure load to the top
        self.ida2, self.ncons2 = getIDArray(self.cons2)

        # for a 2x2 element array...
        self.nodes3 = nodeList(1, 1, 1, enum[1], enum[1])
        self.ien3 = get_ien(enum[1], enum[1])
        self.cons3, self.load3 = load_and_cons(enum[1]**2, len(self.nodes3),
                                               self.numD)
        self.cons3[0][0] = 0.0
        self.cons3[1][0] = 0.0
        self.cons3[0][3] = 0.0
        self.cons3[0][6] = 0.0
        
        self.load3[2][1] = [2.0e6, 0, 0]  # load to right end
        self.load3[2][3] = [2.0e6, 0, 0]
        self.ida3, self.ncons3 = getIDArray(self.cons3)

        # for a 2x2 element array deformation load...
        self.cons4, self.load4 = load_and_cons(enum[1]**2, len(self.nodes3),
                                               self.numD)
        self.cons4[0][0] = 0.0
        self.cons4[1][0] = 0.0
        self.cons4[0][2] = 1.0e-5
        self.cons4[0][3] = 0.0
        self.cons4[0][5] = 1.0e-5
        self.cons4[0][6] = 0.0
        self.cons4[0][8] = 1.0e-5
        
        self.ida4, self.ncons4 = getIDArray(self.cons4)

        # For the deformation load...
        self.cons5, self.load5 = load_and_cons(enum[0]**2, len(self.nodes1),
                                               self.numD)
        self.cons5[0][0] = 0.0
        self.cons5[1][0] = 0.0
        self.cons5[0][1] = 1.0e-5
        self.cons5[1][1] = 0.0
        self.cons5[0][2] = 0.0
        self.cons5[0][3] = 1.0e-5

        self.ida5, self.ncons5 = getIDArray(self.cons5)

    # Now, we simulate the deformation behavior
    def test_2DTrac1Elem(self):
        result, steps = solver(self.numD, self.load1, self.nodes1, self.ien1,
                               self.ida1, self.ncons1, self.cons1)
        print(result)
        correct = [0.0, 0.0, 1.0e-5, 0.0, 0.0, -0.3e-5, 1.0e-5, -0.3e-5]
        
        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual((result[i] + 1)/(correct[i] + 1), 1)

    def test_2DBody1Elem(self):
        self.load1[2][0] = 'n'
        self.load1[0][0] = [2.0e6, 0, 0]  # body force
        result, steps = solver(self.numD, self.load1, self.nodes1, self.ien1,
                               self.ida1, self.ncons1, self.cons1)
        
        correct = [0.0, 0.0, 5.0e-6, 0.0, 0.0, -1.5e-6, 5.0e-6, -1.5e-6]
        
        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual((result[i] + 1)/(correct[i] + 1), 1)

    # Next, we define a similar test using deformation loads
    def test_2DPress1Elem(self):
        result, steps = solver(self.numD, self.load2, self.nodes1, self.ien1,
                               self.ida2, self.ncons2, self.cons2)
        
        correct = [0.0, 0.0, -0.3e-5, 0.0, 0.0, 1.0e-5, -0.3e-5, 1.0e-5]
        
        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual((result[i] + 1)/(correct[i] + 1), 1)

    # Here, we test the loaded behavior of a 2x2 array of 2D elements
    def test_2DTrac4Elem(self):
        result, steps = solver(self.numD, self.load3, self.nodes3, self.ien3,
                               self.ida3, self.ncons3, self.cons3)
        
        correct = [0.0, 0.0, 5.0e-6, 0.0, 1.0e-5, 0.0,
                   0.0, -1.5e-6, 5.0e-6, -1.5e-6, 1.0e-5, -1.5e-6,
                   0.0, -3.0e-6, 5.0e-6, -3.0e-6, 1.0e-5, -3.0e-6]
        
        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual((result[i] + 1)/(correct[i] + 1), 1)

    # Now, we test the effect of a deformation load on the 2x2 element set
    def test_2DDef4Elem(self):
        result, steps = solver(self.numD, self.load4, self.nodes3, self.ien3,
                               self.ida4, self.ncons4, self.cons4)
        
        correct = [0.0, 0.0, 5.0e-6, 0.0, 1.0e-5, 0.0,
                   0.0, -1.5e-6, 5.0e-6, -1.5e-6, 1.0e-5, -1.5e-6,
                   0.0, -3.0e-6, 5.0e-6, -3.0e-6, 1.0e-5, -3.0e-6]
        
        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual((result[i] + 1)/(correct[i] + 1), 1)

    # Here, we test the deformation on a 1x1 element...
    def test_2DDef1Elem(self):
        result, steps = solver(self.numD, self.load5, self.nodes1, self.ien1,
                               self.ida5, self.ncons5, self.cons5)

        correct = [0.0, 0.0, 1.0e-5, 0.0, 0.0, -0.3e-5, 1.0e-5, -0.3e-5]

        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual((result[i] + 1)/(correct[i] + 1), 1)

##############################################################################

# Here, we write a class that performs some basic testing for a 3D element

class SolverTest3D(unittest.TestCase):
    # first, define important preliminary data
    def setUp(self):
        self.numD = 3  # the number of problem dimensions
        enum = [1, 2]  # number of elements
        self.nodes1 = nodeList(2, 2, 2, enum[0], enum[0], enum[0])
        self.ien1 = get_ien(enum[0], enum[0], enum[0])
        self.cons1, self.load1 = load_and_cons(enum[0], len(self.nodes1),
                                               self.numD)
        self.cons1[1][0] = 0.0  # constrain node 0 in all dof
        self.cons1[2][0] = 0.0
        self.cons1[2][2] = 0.0  # prevent rotation about x
        for i in [0, 2, 4, 6]:  # for every constrained node
            self.cons1[0][i] = 0.0
        
        self.load1[2][0] = [2e8, 0, 0]#[24653148345, 0, 0]  # load to right end
        self.ida1, self.ncons1 = getIDArray(self.cons1)

    # Here, we test to ensure that 3D tractions calculate correctly
    def test_3DTrac1Elem(self):
        result, steps = solver(self.numD, self.load1, self.nodes1, self.ien1,
                               self.ida1, self.ncons1, self.cons1)
        dd = 1.0e-5
        nn = 0.3*dd #0.32012396773595*dd
        correct = [0.0, 0.0, 0.0, dd, 0.0, 0.0,
                   0.0, -nn, 0.0, dd, -nn, 0.0,
                   0.0, 0.0, -nn, dd, 0.0, -nn,
                   0.0, -nn, -nn, dd, -nn, -nn]
        
        for i in range(len(result)):  # for every component...
            self.assertAlmostEqual((result[i] + 1)/(correct[i] + 1), 1)

##############################################################################

# now, the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(SolverTest1D)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(SolverTest2D)
Suite3 = unittest.TestLoader().loadTestsFromTestCase(SolverTest3D)

FullSuite = unittest.TestSuite([Suite1, Suite2, Suite3])

SingleSuite = unittest.TestSuite()
SingleSuite.addTest(SolverTest3D('test_3DTrac1Elem'))

unittest.TextTestRunner(verbosity=2).run(SingleSuite)
