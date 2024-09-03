"""In this file, we create the plots needed to demonstrate the solving
capability of the non-linear solver."""

import unittest
from Assignment_1 import nodeList, get_ien, getIDArray
from Assignment_2 import load_and_cons
from Assignment_8 import solver
from Assignment_9 import contourPlot


class get3D_Trac_Plot(unittest.TestCase):
    """Here, we write a class that performs some basic testing for a 3D
    element."""

    # first, define important preliminary data
    def setUp(self):
        self.numD = 3  # the number of problem dimensions
        enum = 1  # number of elements
        self.eLx = 1.0  # element length in the x-direction
        self.eLy = 1.0  # element length in the y-direction
        self.eLz = 1.0  # element length in the z-direction
        self.t = 2.0e10  # traction stress
        self.nodes1 = nodeList(self.eLx, self.eLy, self.eLz, enum, enum, enum)
        self.ien1 = get_ien(enum, enum, enum)
        self.cons1, self.load1 = load_and_cons(enum, len(self.nodes1),
                                               self.numD)
        self.cons2, self.load2 = load_and_cons(enum, len(self.nodes1),
                                               self.numD)
        self.cons1[1][0] = 0.0  # constrain node 0 in all dof
        self.cons1[2][0] = 0.0
        self.cons1[2][2] = 0.0  # prevent rotation about x
        for i in [0, 2, 4, 6]:  # for every constrained node
            self.cons1[0][i] = 0.0

        self.load1[2][0] = [self.t, 0, 0]  # load to right end

        self.load2[1][0] = [0, -self.t, 0]  # shear xy
        self.load2[2][0] = [0, self.t, 0]
        self.load2[3][0] = [-self.t, 0, 0]
        self.load2[4][0] = [self.t, 0, 0]

        self.ida1, self.ncons1 = getIDArray(self.cons1)

    # Here, we plot the tension 3D stresses
    def test_3DplotStretch(self):
        result, steps = solver(self.numD, self.load1, self.nodes1, self.ien1,
                               self.ida1, self.ncons1, self.cons1)
        dd = self.eLx*self.t/2.0e11
        nn = 0.3*self.eLx*self.t/2.0e11
        correct = [0.0, 0.0, 0.0, dd, 0.0, 0.0,
                   0.0, -nn, 0.0, dd, -nn, 0.0,
                   0.0, 0.0, -nn, dd, 0.0, -nn,
                   0.0, -nn, -nn, dd, -nn, -nn]

        selset = [[0, 1, 2, 3], [0], 'z']
        contourPlot(result, self.ien1, self.nodes1, 'd_x', 'z', selset,
                    [2e11, 0.3], 'yes')
        self.assertEqual(0, 0)

    # Here, we plot the shear stress results
    def test_3DplotShear(self):
        result, steps = solver(self.numD, self.load2, self.nodes1, self.ien1,
                               self.ida1, self.ncons1, self.cons1)
        dd = self.eLx*self.t/2.0e11
        nn = 0.3*self.eLx*self.t/2.0e11
        correct = [0.0, 0.0, 0.0, dd, 0.0, 0.0,
                   0.0, -nn, 0.0, dd, -nn, 0.0,
                   0.0, 0.0, -nn, dd, 0.0, -nn,
                   0.0, -nn, -nn, dd, -nn, -nn]

        selset = [[0, 1, 2, 3], [0], 'z']
        contourPlot(result, self.ien1, self.nodes1, 'd_x', 'z', selset,
                    [2e11, 0.3], 'yes')
        self.assertEqual(0, 0)
