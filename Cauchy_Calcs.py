from sympy import Matrix, eye, symbols, pprint

# In this file, we calculate the Cauchy stress from the second Piola-Kirchhoff
# stress and the green strain

a, m, n = symbols('a m n')

# for a pure, 1D extension (see Example 4.3, p. 10 in Cont. Mech. Textbook)
G = Matrix([[a, 0, 0], [0, 0, 0], [0, 0, 0]])  # gradient tensor
F = G + eye(3)

S = Matrix([[m, 0, 0], [0, n, 0], [0, 0, n]])
sigma = F*S*F.T/F.det()

pprint(sigma)
