# In this file, we calculate some of the large deflection behavior of a simple
# tension test symbollically.

from sympy import symbols, Matrix, pprint, eye, simplify, latex, solveset, Eq

a, v, x, y, z, ym = symbols('a v x y z, ym')
e11, e22, e33, e23, e13, e12, L, M = symbols('e11 e22 e33 e23 e13 e12 L M')

# For a simple tension test in 3D, the deformation field is given by
u = Matrix([a*x, -v*a*y, -v*a*z])
xs = Matrix([x, y, z])  # The independent variables in the derivatives

G = u.jacobian(xs)
F = eye(3) + G  # the Green strain
E = simplify(0.5*(F.T*F - eye(3)))
J = F.det()

# The stiffness matrix 'D'
ld = ym*v/((1 + v)*(1 - 2*v))  # Lame parameters
mu = ym/(2*(1 + v))

D = Matrix([[ld + 2*mu,    ld,             ld,             0,      0,      0],
            [ ld,           ld + 2*mu,      ld,             0,      0,      0],
            [ ld,           ld,             ld + 2*mu,      0,      0,      0],
            [ 0,            0,              0,              mu,     0,      0],
            [ 0,            0,              0,              0,      mu,     0],
            [ 0,            0,              0,              0,      0,      mu]])

# Here, we get the Voigt notation for the strain vector
Ev = (Matrix([[E[0,0], E[1,1], E[2,2], E[1,2], E[0,2], E[0,1]]])).T
S = simplify(D.subs(ym, 200e9)*Ev.subs(a, 0.1))
Sn = S.subs(v, 0.3)

Ssq = Matrix([[Sn[0,0], Sn[5,0], Sn[4,0]], [Sn[5,0], Sn[1,0], Sn[3,0]],
              [Sn[4,0], Sn[3,0], Sn[2,0]]])

EE = Matrix([[e11, e12, e13], [e13, e22, e23], [e13, e23, e33]])
sigma = F*Ssq*F.T/J
SeX = simplify(ld*EE.trace()*eye(3) + 2*mu*EE)
pprint(SeX)



