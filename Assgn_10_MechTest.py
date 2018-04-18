# In this file, we test all the code written for the non-linear code


import unittest
import numpy as np
from Assignment_1 import nodeList, get_ien
from Assignment_4 import getBasis
from Assignment_5 import posAndJac
from Assignment_6 import getElemDefs
from Assignment_10 import getPK2, getCauchy

############################################################################

# Here, we test the second Piola-Kirchhoff stress tensor calculation
class TestPK2Stress(unittest.TestCase):
    # For large deformation
    def test_S_1D_1Elem_Large(self):
        d = 0.2e-4  # deformation amount
        eL = 2  # element length
        deform = [0.0, d]
        ym = 200e9  # Young's modulus
        ien = get_ien(1)
        basis = getBasis(1)
        xa = nodeList(eL, eL, eL, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getPK2(defE, basis[0][0], jac)
        
        a = d/eL
        correct = [[(a + 0.5*a**2)*ym]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0]+1)/(S[i][0]+1),1)

    # For 2D large deformation...
    def test_S_2D_1Elem_Large(self):
        d = 2.0e-1
        eLx = 2.0  # element length in the x-direction
        eLy = 2.0  # element length in the y-direction
        E = 200.0*10**9    # modulus of elasticity (Pa)
        v = 0.3         # Poisson's ratio
        deform = [0, 0, d, 0, 0, 0, d, 0]
        ien = get_ien(1, 1)
        basis = getBasis(2)
        xa = nodeList(eLx, eLy, 2, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getPK2(defE, basis[0][0], jac)
        
        a = d/eLx  # deformation ratio
        ex = ((a + 1)**2/2 - 0.5)
        correct = [[(E/(1-v**2))*ex], [(E*v/(1-v**2))*ex], [0]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1)

    # For 2D large deformation...
    def test_S_2D_1Elem_Large_Shear(self):
        d = 2.0e-1
        eLx = 2.0  # element length in the x-direction
        eLy = 2.0  # element length in the y-direction
        E = 200.0*10**9    # modulus of elasticity (Pa)
        v = 0.3         # Poisson's ratio
        deform = [0, 0, 0, 0, d, 0, d, 0]
        ien = get_ien(1, 1)
        basis = getBasis(2)
        xa = nodeList(eLx, eLy, 2, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getPK2(defE, basis[0][0], jac)
        
        a = d/eLy  # deformation ratio
        aa = E/(1-v**2)  # 2D stiffness parameters
        bb = aa*(1 - v)  # multiplied by 2 since it's not engineering strain
        cc = aa*v
        correct = [[0.5*cc*a**2], [0.5*aa*a**2], [0.5*bb*a]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1)

    # For 3D large deformation...
    def test_S_3D_1Elem_Large(self):
        d = 2.0e-1  # deformation amount
        eLx = 2.0  # element length in the x-direction
        eLy = 2.0  # element length in the y-direction
        eLz = 2.0  # element length in the z-direction
        E = 200.0*10**9    # modulus of elasticity (Pa)
        v = 0.3         # Poisson's ratio
        ld = E*v/((1 + v)*(1 - 2*v))  # Lame parameters
        mu = E/(2*(1 + v))
        deform = [0, 0, 0, d, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, d, 0, 0,
                  0, 0, 0, d, 0, 0]
        ien = get_ien(1, 1, 1)
        basis = getBasis(3)
        xa = nodeList(eLx, eLy, eLz, 1, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getPK2(defE, basis[0][0], jac)

        a = d/eLx  # displacement ratio
        ex = a + 0.5*a**2
        correct = [[(ld + 2*mu)*ex], [ld*ex], [ld*ex], [0], [0], [0]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1, 5)

    # For 3D large deformation...
    def test_S_3D_1Elem_Large_ShearXZ(self):
        d = 2.0e-1  # deformation amount
        eLx = 2.0  # element length in the x-direction
        eLy = 2.0  # element length in the y-direction
        eLz = 2.0  # element length in the z-direction
        E = 200.0*10**9    # modulus of elasticity (Pa)
        v = 0.3         # Poisson's ratio
        ld = E*v/((1 + v)*(1 - 2*v))  # Lame parameters
        mu = E/(2*(1 + v))
        deform = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  d, 0, 0, d, 0, 0, d, 0, 0, d, 0, 0]
        ien = get_ien(1, 1, 1)
        basis = getBasis(3)
        xa = nodeList(eLx, eLy, eLz, 1, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getPK2(defE, basis[0][0], jac)

        a = d/eLz  # displacement ratio
        e3 = a**2/2  # strain in zz
        e5 = a/2  # strain in xz
        correct = [[ld*e3], [ld*e3], [(ld + 2*mu)*e3], [0], [2*mu*e5], [0]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1, 5)

############################################################################

# Here, we test the accuracy of the Cauchy stress tensor calculation

class TestCauchyStressTensor(unittest.TestCase):
    # For large deformation
    def test_sigma_1D_1Elem_Large(self):
        d = 0.2  # deformation amount
        eLx = 2.0  # element length in the x-direction
        deform = [0.0, d]
        ien = get_ien(1)
        basis = getBasis(1)
        xa = nodeList(eLx, 2, 2, 1)
        x, jacXxi = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getCauchy(defE, basis[0][0], jacXxi)

        a = d/eLx  # deformation ratio
        ex = a + 0.5*a**2  # strain in the x-direction
        ym = 200*10**9  # Young's Modulus
        correct = [[ex*ym*(1 + a)]]  # (1 + a) comes from the push-forward
        print(correct)
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual(correct[i][0]/S[i][0], 1)

    # For 2D small deformation...
    def test_sigma_2D_1Elem_Large(self):
        d = 0.2  # deformation amount in the x-direction
        eLx = 2.0 # element length in the x-direction
        eLy = 2.0 # element length in the y-direction
        deform = [0, 0, d, 0, 0, 0, d, 0]
        ien = get_ien(1, 1)
        basis = getBasis(2)
        xa = nodeList(eLx, eLy, 2, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getCauchy(defE, basis[0][0], jac)

        a = d/eLx  # deformation ratio
        ex = a + 0.5*a**2  # strain in the x-direction
        ym = 200*10**9  # Young's Modulus
        nu = 0.3  # Poisson's Ratio
        aa = ym/(1 - nu**2)  # 2D stiffness parameters
        bb = aa*(1 - nu)  # multiplied by 2 since its not engineering strain
        cc = aa*nu
        correct = [[aa*ex*(a + 1)], [cc*ex/(a + 1)], [0]]
        print(correct)
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1)

    # For 3D large deformation...
    def test_sigma_3D_1Elem_Large(self):
        d = 0.2  # deformation amount
        s = -d*0.32012396773595
        eLx = 2.0 # element length in the x-direction
        eLy = 2.0 # element length in the y-direction
        eLz = 2.0 # element length in the z-direction

        a = d/eLx  # deformation ratio
        ex = a + 0.5*a**2  # strain in the x-direction
        E = 200*10**9  # Young's Modulus
        v = 0.3  # Poisson's Ratio
        ld = E*v/((1 + v)*(1 - 2*v))  # Lame parameters
        mu = E/(2*(1 + v))
        
        deform = [0, 0, 0, d, 0, 0, 0, 0, 0, d, 0, 0,
                  0, 0, 0, d, 0, 0, 0, 0, 0, d, 0, 0]
        deform1D = [0, 0, 0, d, 0, 0, 0, s, 0, d, s, 0,
                    0, 0, s, d, 0, s, 0, s, s, d, s, s]
        
        ien = get_ien(1, 1, 1)
        basis = getBasis(3)
        xa = nodeList(eLx, eLy, eLz, 1, 1, 1)
        x, jac = posAndJac(basis[0][0], xa)
        defE = getElemDefs(0, deform, ien)
        S = getCauchy(defE, basis[0][0], jac)
        print(S)
        correct = [[ex*(a + 1)*(ld + 2*mu)], [ex*ld/(a+1)], [ex*ld/(a+1)],
                   [0], [0], [0]]
        correct1D = [[ex*E*(1 + a)], [0], [0], [0], [0], [0]]
        
        for i in range(len(S)):  # for every component...
            self.assertAlmostEqual((correct[i][0] + 1)/(S[i][0] + 1), 1, 5)

############################################################################

# Now the testing

Suite1 = unittest.TestLoader().loadTestsFromTestCase(TestPK2Stress)
Suite2 = unittest.TestLoader().loadTestsFromTestCase(TestCauchyStressTensor)

FullSuite = unittest.TestSuite([Suite1, Suite2])

SingleSuite = unittest.TestSuite()
SingleSuite.addTest(TestCauchyStressTensor('test_sigma_3D_1Elem_Large'))

unittest.TextTestRunner(verbosity=2).run(Suite2)
        
