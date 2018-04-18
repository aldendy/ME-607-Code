# In this document, we calculate some basic Green strain fields

from sympy import *

x, y, z, a, v, nu= symbols('x y z a v nu')

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

# Finally, we calculate simple Green strain in the yz plane
u = Matrix([0, a*z, 0])
ivar = Matrix([x, y, z])
F = u.jacobian(ivar) + eye(3)

E = (F.T*F - eye(3))/2

# Here, we calculate the deformation field necessary for 1D stress
u = Matrix([[a*x], [-v*a*y], [-v*a*z]])
ivar = Matrix([x, y, z])
F = u.jacobian(ivar) + eye(3)
E = (F.T*F - eye(3))/2
pprint(E)

rat = Eq(-E[1,1]/E[0,0], nu)
eq1 = simplify(rat)
pprint(eq1)
eq2 = eq1.subs(a, 0.1).subs(nu, 0.3)

# For 3D, uniaxial tension...
S = Matrix([[2.1e10, 0, 0], [0, 0, 0], [0, 0, 0]])
sigma = F*S*F.T/F.det()
pprint(sigma.subs(a, 0.1).subs(v, 0.32012396773595))
