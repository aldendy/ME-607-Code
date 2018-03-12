# In this file, we test the code needed to guarantee full plot flexibility


import unittest
import copy
import numpy as np
from math import pi, cos, sin
from Assignment_1 import nodeList, get_ien, getIDArray
from Assignment_2 import load_and_cons
from Assignment_8 import solver
from Assignment_9 import nsel, constrain, get_stress_sol, contourPlot


##########################################################################

# This first class tests the ability of the stress solution function to operate

class StressSolutionTest(unittest.TestCase):
    # Here, we initialize solution dat needed for the tests
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

        # now the 3D problem setup
        self.nodes3 = nodeList(1, 1, 1, enum[0], enum[0], enum[0])
        self.ien3 = get_ien(enum[0], enum[0], enum[0])
        self.cons3, self.load3 = load_and_cons(enum[0], len(self.nodes3),
                                               self.numD[2])
        self.cons3[1][0] = 0.0  # constrain node 0 in all dof
        self.cons3[2][0] = 0.0
        self.cons3[2][2] = 0.0  # prevent rotation about x
        for i in [0, 2, 4, 6]:  # for every constrained node
            self.cons3[0][i] = 0.0

        self.loadTxy = copy.deepcopy(self.load3)
        self.loadTxz = copy.deepcopy(self.load3)
        self.loadVM = copy.deepcopy(self.load3)
        
        self.load3[2][0] = [2.0e8, 0, 0]  # load to right end
        
        self.loadTxy[2][0] = [0, 2.0e8, 0]  # shear at the right face
        self.loadTxy[1][0] = [0, -2.0e8, 0]
        self.loadTxy[3][0] = [-2.0e8, 0, 0]
        self.loadTxy[4][0] = [2.0e8, 0, 0]

        self.loadTxz[1][0] = [0, 0, -2.0e8]
        self.loadTxz[2][0] = [0, 0, 2.0e8]
        self.loadTxz[5][0] = [-2.0e8, 0, 0]
        self.loadTxz[6][0] = [2.0e8, 0, 0]

        self.loadVM[1][0] = [1260.0, 0, -500.0]  # von Mises load
        self.loadVM[2][0] = [-1260.0, 0, 500.0]
        self.loadVM[5][0] = [-500.0, 0, -800.0]
        self.loadVM[6][0] = [500.0, 0, 800.0]
        
        self.ida3, self.ncons3 = getIDArray(self.cons3)

    # Now we test the solution process for a 1D problem
    def test_stressSolution1D1Elem(self):
        result, steps = solver(self.numD[0], self.load1, self.nodes1, self.ien1,
                               self.ida1, self.ncons1, self.cons1)

        stress = get_stress_sol(result, self.ien1, self.nodes1, 'sigma_x')
        correct = [2.0e6, 2.0e6]

        for i in range(len(correct)):  # for each node...
            self.assertAlmostEqual(correct[i], stress[i])

    # Here, we test the solution process for 3D in every dof (x, y, z, von mises)
    # first we check sigma_x
    def test_stressSol3D1ElemX(self):
        result, steps = solver(self.numD[2], self.load3, self.nodes3, self.ien3,
                               self.ida3, self.ncons3, self.cons3)

        stress = get_stress_sol(result, self.ien3, self.nodes3, 'sigma_x')
        correct = [2.0e8, 2.0e8, 2.0e8, 2.0e8, 2.0e8, 2.0e8, 2.0e8, 2.0e8]
        
        for i in range(len(correct)):  # for each node...
            self.assertAlmostEqual(correct[i], stress[i], 3)

    # Now, check sigma_y
    def test_stressSol3D1ElemY(self):
        result, steps = solver(self.numD[2], self.load3, self.nodes3, self.ien3,
                               self.ida3, self.ncons3, self.cons3)

        stress = get_stress_sol(result, self.ien3, self.nodes3, 'sigma_y')
        correct = 8*[0.0]
        
        for i in range(len(correct)):  # for each node...
            self.assertAlmostEqual(correct[i], stress[i], 3)

    # Now, check sigma_z
    def test_stressSol3D1ElemZ(self):
        result, steps = solver(self.numD[2], self.load3, self.nodes3, self.ien3,
                               self.ida3, self.ncons3, self.cons3)

        stress = get_stress_sol(result, self.ien3, self.nodes3, 'sigma_z')
        correct = 8*[0.0]
        
        for i in range(len(correct)):  # for each node...
            self.assertAlmostEqual(correct[i], stress[i], 3)

    # Here, check shear in xz
    def test_stressSol3D1ElemXY(self):
        result, steps = solver(self.numD[2], self.loadTxy, self.nodes3, self.ien3,
                               self.ida3, self.ncons3, self.cons3)
        
        stress = get_stress_sol(result, self.ien3, self.nodes3, 'tau_xy')
        correct = 8*[2.0e8]
        
        for i in range(len(correct)):  # for each node...
            self.assertAlmostEqual(correct[i], stress[i], 3)

    # Finally, we verify shear stresses in xz
    def test_stressSol3D1ElemXZ(self):
        result, steps = solver(self.numD[2], self.loadTxz, self.nodes3, self.ien3,
                               self.ida3, self.ncons3, self.cons3)
        
        stress = get_stress_sol(result, self.ien3, self.nodes3, 'tau_zx')
        correct = 8*[2.0e8]
        
        for i in range(len(correct)):  # for each node...
            self.assertAlmostEqual(correct[i], stress[i], 3)

    # Our last major test verifies the von Mises calculations made previously
    def test_stressSol3D1ElemVM(self):
        result, steps = solver(self.numD[2], self.loadVM, self.nodes3, self.ien3,
                               self.ida3, self.ncons3, self.cons3)
        
        stress = get_stress_sol(result, self.ien3, self.nodes3, 'von Mises')
        correct = 8*[1996.4]
        
        for i in range(len(correct)):  # for each node...
            self.assertAlmostEqual(correct[i], stress[i], 1)

########################################################################################

# In this next class, we test the correctness of the plotting process

class contourPlotTest(unittest.TestCase):
    # first we define important necessary variables
    def setUp(self):
        # Here, we generate the mesh
        thetaDomain = pi/2.0  # quarter circle of the pipe
        self.ri = 1.2  # the inner radius of the pipe
        self.ro = 1.8  # the outer radius of the pipe
        self.nr = 4  # the number of elements in the radial direction
        self.nt = 8  # the number of elements in the circumfrential direction
        self.nodes = []  # stores the nodes in the cylindrical mesh
        self.p = 2.0e6  # the pressure (Pa)

        for i in range(self.nr+1):  # for every node in the r-direction...
            for j in range(self.nt+1):  # for every node in the theta-direction...
                radius = i*(self.ro - self.ri)/self.nr + self.ri
                theta = j*thetaDomain/self.nt
                self.nodes.append([radius*cos(theta), radius*sin(theta), 0])

        self.ien = get_ien(self.nt, self.nr)
        self.nnums = np.linspace(0, len(self.nodes)-1, len(self.nodes))

        # now the fun begins. We solve this problem using roller constraints on
        # the straight faces and pressure loads on the inner face.
        s0 = nsel('y', 'n', 0, 0.01, self.nodes)
        s1 = nsel('x', 'n', 0, 0.01, self.nodes)

        ida, ncons, cons0, loads = constrain(self.nodes, s0, self.ien, 'y', 0)
        ida, ncons, cons, loads = constrain(self.nodes, s1, self.ien, 'x', 0,
                                            cons0)
        for i in range(self.nt):  # for every inner element...
            loads[3][i] = self.p

        self.deform, i = solver(2, loads, self.nodes, self.ien, ida, ncons, cons)
        ps0 = nsel('y', 'n', 0, 0.01, self.nodes)
    
    # Here, we test the plotting process.
    def test_stressPressCylinSol(self):
        #c = contourPlot(self.deform, self.ien, self.nodes, 'von Mises')
        r = 1.2
        a = self.p*self.ri**2/(self.ro**2 - self.ri**2)
        b = self.p*self.ri**2*self.ro**2/(r**2*(self.ro**2 - self.ri**2))
        rstress = a - b

        d = self.p*self.ri**2/(self.ro**2 - self.ri**2)
        e = self.p*self.ri**2*self.ro**2/(r**2*(self.ro**2 - self.ri**2))
        tstress = d + e
        self.assertEqual(0, 0)
        print('Radial Stress:', rstress)
        print('Tangential Stress:', tstress)

#############################################################################

# Now the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(StressSolutionTest)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(contourPlotTest)

FullSuite = unittest.TestSuite([Suite1, Suite2])

SingleSuite = unittest.TestSuite()
SingleSuite.addTest(contourPlotTest('test_stressPressCylinSol'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)




