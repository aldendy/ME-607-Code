# In this document, we calculate some basic Green strain fields

from sympy import *

x, y, z, a, v = symbols('x y z a v')

# First, define a deformation field for a lengthening in x with poisson's ratio
u = Matrix([[a*x], [0], [0]])
ivar = Matrix([x, y, z])
F = u.jacobian(ivar) + eye(3)
E = (F.T*F - eye(3))/2

# Next, we calculate strain in the xz plane
u = Matrix([a*z, 0, 0])
ivar = Matrix([x, y, z])
F = u.jacobian(ivar) + eye(3)

E = (F.T*F - eye(3))/2
pprint(E)

# Finally, we calculate simple Green strain in the yz plane
u = Matrix([0, a*z, 0])
ivar = Matrix([x, y, z])
F = u.jacobian(ivar) + eye(3)

E = (F.T*F - eye(3))/2

