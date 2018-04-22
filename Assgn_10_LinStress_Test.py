# In this file, we test the ability of the Cauchy stress function to simulate
# linear stress relationships (small strain)

import unittest
from Assignment_1 import nodeList
from Assignment_4 import getBasis
from Assignment_5 import posAndJac
from Assignment_10 import getCauchy

############################################################################

# Here, we test the linear stress small strain capabilities of the function
class SmallStressStrainCauchyTest(unittest.TestCase):
    # Here, we test 1D small strain
    def test_1D_Linear_Stress(self):
        ym = 2.0e11  # Young's modulus
        defE = [[0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0]]  # element deformations
        eLx = 2.0  # element length in the x-direction
        ex = defE[1][0]/eLx  # strain in the x-direction
        basis = getBasis(1)
        xa = nodeList(eLx, eLx, eLx, 1)
        x, jac = posAndJac(basis[0][0], xa)
        S = getCauchy(defE, basis[0][0], jac, 'lin')

        correct = [[ym*ex*(defE[1][0]/eLx + 1)]]

        self.assertAlmostEqual(S[0][0], correct[0][0], 3)

#############################################################################

# Now the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(SmallStressStrainCauchyTest)

FullSuite = unittest.TestSuite([Suite1])

#SingleSuite = unittest.TestSuite()
#SingleSuite.addTest(TestCauchyStressTensor('test_sigma_3D_1Elem_example'))

unittest.TextTestRunner(verbosity=2).run(FullSuite)
        
