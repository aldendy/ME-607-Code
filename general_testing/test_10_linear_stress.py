"""In this file, we test the ability of the Cauchy stress function to simulate
linear stress relationships (small strain)."""

import unittest
from Assignment_1 import nodeList
from Assignment_4 import getBasis
from Assignment_5 import posAndJac
from Assignment_10 import getCauchy


class SmallStressStrainCauchyTest(unittest.TestCase):
    """Here, we test the linear stress small strain capabilities of the
    function."""

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

        correct = [[ym*ex]]

        self.assertAlmostEqual(S[0][0], correct[0][0], 3)

    # Next, we test 2D small strain
    def test_2D_Linear_Stress(self):
        ym = 2.0e11  # Young's modulus
        d = 1.0e-4  # deformation in the x-direction
        v = 0.3  # Poisson's ratio
        defE = [[0.0, 0.0, 0.0], [d, 0.0, 0.0],
                [0.0, -v*d, 0.0], [d, -v*d, 0.0]]  # element deformations
        eLx = 2.0  # element length in the x-direction
        eLy = 2.0  # element length in the y-direction
        ex = defE[1][0]/eLx  # strain in the x-direction
        ey = defE[2][1]/eLy  # strain in the y-direction
        basis = getBasis(1)
        xa = nodeList(eLx, eLx, eLx, 1)
        x, jac = posAndJac(basis[0][0], xa)
        S = getCauchy(defE, basis[0][0], jac, 'lin')

        correct = [[ym*ex], [ym*ey], [0]]

        for i in range(len(S)):  # for every row...
            self.assertAlmostEqual(S[i][0], correct[i][0], 3)
