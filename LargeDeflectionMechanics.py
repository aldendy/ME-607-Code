# In this file, we calculate some of the large deflection behavior of a simple
# tension test symbollically.

from sympy import symbols, Matrix, pprint, eye, simplify, latex, solveset, Eq

a, v, x, y, z, ld, mu, ex = symbols('a v x y z ld mu ex')

# For a simple tension test in 3D, the deformation field is given by
u = Matrix([a*x, 0, 0])
xs = Matrix([x, y, z])  # The independent variables in the derivatives

G = u.jacobian(xs)
pprint(G)
F = eye(3) + G  # the deformation gradient
E = simplify(0.5*(F.T*F - eye(3)))  # Green strain
J = F.det()

S = Matrix([[(ld + 2*mu)*ex, 0, 0], [0, ld*ex, 0], [0, 0, ld*ex]])

sigma = F*S*F.T/J

pprint(sigma)
