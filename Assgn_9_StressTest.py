# In this file, we perform further testing to ensure that the code is functioning as planned


import unittest
from Assignment_1 import get_ien
from Assignment_8 import solver
from Assignment_9 import getMises, constrain, contourPlot


###########################################################################

# In this first class, we test the getMises function

class MisesTest(unittest.TestCase):
    # Here, we test multiple 2D stress states
    def test_2DMises(self):
        s1 = [[120.0], [-40.0], [50.0]]
        s2 = [[68.0], [32.0], [0]]
        sm1 = getMises(s1)
        sm2 = getMises(s2)
        
        self.assertAlmostEqual(sm1, 168.226, 2)
        self.assertAlmostEqual(sm2, 58.924, 2)

    # Here, we test multiple 3D stress states
    def test_3DMises(self):
        s1 = [[200.0], [100.0], [-50.0], [-30.0], [0.0], [0.0]]
        s2 = [[80.0], [50.0], [20.0], [40.0], [40.0], [40.0]]
        sm1 = getMises(s1)
        sm2 = getMises(s2)
        
        self.assertAlmostEqual(sm1, 224.054, 2)
        self.assertAlmostEqual(sm2, 130.767, 2)

##########################################################################

# Now, we define a tilted element test to ensure that rotation does not
# affect solutions

class RotatedElementTest(unittest.TestCase):
    # Here, we set up the problem
    def setUp(self):
        self.nodes = [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [2.0, 3.0, 0.0],
                      [3.0, 2.0, 0.0]]
        p = 2.0e6  # pressure load
        self.ien = get_ien(1, 1)
        selset = [2, 3]  # selected node numbers
        ida, ncons, cons0, loads = constrain(self.nodes, selset,
                                             self.ien, 'x', 0)
        ida, ncons, cons, loads = constrain(self.nodes, selset,
                                            self.ien, 'y', 0, cons0)
        loads[3][0] = p
        self.deform, i = solver(2, loads, self.nodes, self.ien, ida,
                                ncons, cons)

    # Here, we simulate a tilted 2D solution
    def test_2DTiltedElement(self):
        c = contourPlot(self.deform, self.ien, self.nodes, 'd_abs', 'z')

##########################################################################

# Now testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(MisesTest)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(RotatedElementTest)

FullSuite = unittest.TestSuite([Suite1])

SingleSuite = unittest.TestSuite()
SingleSuite.addTest(MisesTest('test_2DMises'))

unittest.TextTestRunner(verbosity=2).run(Suite2)
