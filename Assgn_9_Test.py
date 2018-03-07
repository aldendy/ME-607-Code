# In this file, we test the functions from Assignment 9. 


import unittest
import numpy as np
from Assignment_1 import nodeList, get_ien
from Assignment_9 import nsel, constrain


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

        self.ida, self.ncons, self.cons = constrain(self.nodes, self.s0,
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

#####################################################################

# testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(nselTests)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(constrainTest)

FullSuite = unittest.TestSuite([Suite1, Suite2])

#SingleSuite = unittest.TestSuite()
#SingleSuite.addTest(SolverTest3D('test_3DTrac1Elem'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)
