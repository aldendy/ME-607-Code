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

aa = simplify(ld + 2*mu - ld**2/(ld + 2*mu))

cc = simplify(ld - ld**2/(ld + 2*mu))

# Here, we find the lambda form for the 1D constant
