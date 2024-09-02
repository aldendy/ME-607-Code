# In this file, we test the basic functionality in Assignment_4.

import unittest
from assignment_4 import getIntPts, bb1, bb2, bb3, bb4, threeDBasisFunc

##########################################################

class IntPointTest(unittest.TestCase):
    # First, we test the ability of 'getIntPts' to correctly find the gauss
    # point locations
    def test_1Dpts(self):
        pts1 = getIntPts(1)
        self.assertEqual(len(pts1), 3)  # should have three regions

    # In 2D...
    def test2Dpts(self):
        pts2 = getIntPts(2)
        self.assertEqual(len(pts2), 5)
        self.assertEqual(len(pts2[0]), 4)
        self.assertEqual(len(pts2[1]), 2)
        self.assertEqual(len(pts2[3]), 2)
    # in 3D....
    def test_3Dpts(self):
        pts3 = getIntPts(3)
        self.assertEqual(len(pts3), 7)
        self.assertEqual(len(pts3[0]), 8)
        self.assertEqual(len(pts3[1]), 4)
        self.assertEqual(len(pts3[3]), 4)
        self.assertEqual(len(pts3[5]), 4)

##########################################################

# This class implements tests on the basis functions

class BasisTest(unittest.TestCase):
    # First, we test that each basis function is 'one' at the correct nodes
    def test_bb1(self):
        self.assertAlmostEqual((bb1(-1, -1))[0], 1)
    def test_bb2(self):
        self.assertAlmostEqual((bb2(1, -1))[0], 1)
    def test_bb3(self):
        self.assertAlmostEqual((bb3(-1, 1))[0], 1)
    def test_bb4(self):
        self.assertAlmostEqual((bb4(1, 1))[0], 1)

    def test_bbb1(self):
        self.assertAlmostEqual((threeDBasisFunc(-1, -1, -1, 0))[0], 1)
    def test_bbb2(self):
        self.assertAlmostEqual((threeDBasisFunc(1, -1, -1, 1))[0], 1)
    def test_bbb3(self):
        self.assertAlmostEqual((threeDBasisFunc(-1, 1, -1, 2))[0], 1)
    def test_bbb4(self):
        self.assertAlmostEqual((threeDBasisFunc(1, 1, -1, 3))[0], 1)
    def test_bbb5(self):
        self.assertAlmostEqual((threeDBasisFunc(-1, -1, 1, 4))[0], 1)
    def test_bbb6(self):
        self.assertAlmostEqual((threeDBasisFunc(1, -1, 1, 5))[0], 1)

#########################################################

Suite1 = unittest.TestLoader().loadTestsFromTestCase(IntPointTest)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(BasisTest)
FullSuite = unittest.TestSuite([Suite1, Suite2])

unittest.TextTestRunner(verbosity=2).run(FullSuite)
