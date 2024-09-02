"""In this file, we implement Gaussian quadrature rules over one, two and three
dimensions."""


from collections.abc import Callable


def oneDGauss(func: Callable[[float], float]):
    """
    This function implements two-point Gaussian quadrature in one dimension
    over the domain (-1, 1) in 'xi'.

    The inputs are:
    'func' - a function describing the integrand
    """
    w = [1, 1]  # a vector of weights
    xi = [-1.0/3**0.5, 1.0/3**0.5]  # a vector of integration points

    area = w[0]*func(xi[0]) + w[1]*func(xi[1])  # compute the integral
    return area


def oneDTest():
    """In this function, we test the ability of 'oneDGauss' to integrate"""

    # for a function...
    def testfunc1(xi):
        return xi**2 + 1

    def testfunc2(xi):
        return xi**3

    # integrate
    ans = oneDGauss(testfunc1)
    print('1D Test:')
    print('Function 1 error = ', (ans-2.0/3-2)/(8.0/3))

    ans = oneDGauss(testfunc2)
    print('Function 2 error = ', ans)


def twoDGauss(func: Callable[[float, float], float], s: int):
    """Next, we implement a 2D, four-point Gauss quadrature rule over a square
    domain (-1, 1) x (-1, 1) in 'xi' and 'eta'.

    The inputs are:
    'func' - a function describing the integrand (a function of 'xi' and 'eta')
    's' - the part of the element over which to integrate (0 - interior, 1, 2,
                                                           3, 4 - sides)
    """
    d = 1.0/3**0.5
    # A table of integration points and weights where the indexes are
    # [interior (0) or one of four surfaces, integration point #] and
    # the data stores is a [point coordinate in 'xi' and 'eta', weight]
    intpts = [[[[-d, -d], 1], [[-d, d], 1], [[d, -d], 1], [[d, d], 1]],
              [[[-1, -d], 1], [[-1, d], 1]],
              [[[1, -d], 1], [[1, d], 1]],
              [[[-d, -1], 1], [[d, -1], 1]],
              [[[-d, 1], 1], [[d, 1], 1]]]

    area = 0  # initialize the area variable
    for i in range(len(intpts[s])):
        area += intpts[s][i][1]*func(intpts[s][i][0][0],
                                     intpts[s][i][0][1])

    return area


def twoDTest():
    """Here, we create a function to test the functions of 'twoDGauss'."""

    def funcA(xi, eta):
        return xi**2 + eta**2 + 1

    def funcB(xi, eta):
        return xi**3 + eta**3 + xi**2 + 1

    error1 = (twoDGauss(funcA, 0) - 20.0/3)*3/20
    error2 = (twoDGauss(funcB, 0) - 16.0/3)*3/16
    print('2D Test')
    print('Function 1 error:', error1)
    print('Function 2 error:', error2)


def threeDGauss(func: Callable[[float, float, float], float], s: int):
    """Finally, we implement a 3D Gauss quadrature rule.

    The inputs are:
    'func' - the integrand with inputs xi, eta, zeta and domain (-1, 1) x
             (-1, 1) x (-1, 1)
    's' - part of element over which to integrate (0 for the interior, 1, 2, 3,
                                                   4, 5, 6 for the sides)
    """
    d = 1.0/3**0.5
    # A table of integration points and weights where the indexes are
    # [interior (0) or one of four surfaces, integration point #] and
    # the data stores is a [point coordinate in 'xi' and 'eta', weight]
    intpts = [[[[-d, -d, -d], 1], [[d, -d, -d], 1], [[-d, d, -d], 1],
               [[d, d, -d], 1], [[-d, -d, d], 1], [[d, -d, d], 1],
               [[-d, d, d], 1], [[d, d, d], 1]],
              [[[-1, -d, -d], 1], [[-1, d, -d], 1],
                  [[-1, -d, d], 1], [[-1, d, d], 1]],
              [[[1, -d, -d], 1], [[1, -d, -d], 1],
                  [[1, -d, -d], 1], [[1, -d, -d], 1]],
              [[[-d, -1, -d], 1], [[d, -1, -d], 1],
                  [[-d, -1, d], 1], [[d, -1, d], 1]],
              [[[-d, 1, -d], 1], [[d, 1, -d], 1], [[-d, 1, d], 1],
               [[d, 1, d], 1]],
              [[[-d, -d, -1], 1], [[d, -d, -1], 1],
                  [[-d, d, -1], 1], [[d, d, -1], 1]],
              [[[-d, -d, 1], 1], [[d, -d, 1], 1], [[-d, d, 1], 1],
               [[d, d, 1], 1]]]

    area = 0  # initialize the integral result
    for i in range(len(intpts[s])):
        area += intpts[s][i][1]*func(intpts[s][i][0][0],
                                     intpts[s][i][0][1],
                                     intpts[s][i][0][2])
    return area


def threeDTest():
    """Here, we test the 3D Gauss quadrature"""

    def funcA(xi, eta, zeta):
        return xi**2 + eta**2 + 4*zeta**2

    def funcB(xi, eta, zeta):
        return xi**3 + 3*eta**3 + 4*zeta**3 + 1

    error1 = (threeDGauss(funcA, 0) - 16)/16
    error2 = (threeDGauss(funcB, 0) - 8)/8
    print('3D Test')
    print('Function 1 error:', error1)
    print('Function 2 error:', error2)


def GaussInt(n: int, s: int, func: Callable):
    """In this function, we bundle all the integration routines.

    The inputs are:
    'n'   - the number of problem dimensions
    's'   - the region of the element on which we operate
    'func' - the integrand function
    """
    area = 0  # initialize the integral
    if n == 1:
        area = oneDGauss(func)
    if n == 2:
        area = twoDGauss(func, s)
    if n == 3:
        area = threeDGauss(func, s)

    return area
