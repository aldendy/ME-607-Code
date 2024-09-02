# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 22:19:48 2024

@author: AldenYellowhorse
"""

from Assignment_4 import oneDBasis, twoDBasis, threeDBasis, getIntPts, bb4
from Assignment_4 import threeDBasisFunc


def oneDBasisTest():
    """Here, we define a function to test the output of oneDBasis()"""
    basis = oneDBasis()
    d = 1.0/3**0.5  # the distance of a point from the origin
    h1 = 0.5*(1-d)  # b1's output at 'd'
    h2 = 0.5*(1+d)  # b1's output at '-d'
    s1 = -0.5  # b1's slope
    s2 = 0.5  # b2's slope
    test = [[[[h2, s1], [h1, s2]], [[h1, s1], [h2, s2]]]]  # expected output
    print('Is 1D broken?:', not (basis == test))  # is expected?


def twoDBasisTest():
    """Next, we write a test for the 2D basis output"""
    basis = twoDBasis()  # get the basis
    intpts = getIntPts(2)

    problem = False  # indicates whether a value is incorrect
    for i in range(len(basis)):  # for every element region...
        for j in range(len(basis[i])):  # for every integration point...
            k = 3  # the basis number
            # get the appropriate values form one of the bases
            f, fd, fdd = bb4(intpts[i][j][0][0], intpts[i]
                             [j][0][1])  # get real values
            test_a = (f != basis[i][j][k][0])
            test_b = (fd != basis[i][j][k][1])
            test_c = (fdd != basis[i][j][k][2])
            if (test_a or test_b or test_c):
                problem = True  # the stored value is incorrect

    print('Is 2D broken?:', problem)


def threeDBasisTest():
    """Next, we write a test for the 3D basis output"""
    basis = threeDBasis()  # get the basis
    # A table of integration points and weights where the indexes are
    # [interior (0) or one of four surfaces, integration point #] and
    # the data stores is a [point coordinate in 'xi' and 'eta', weight]
    intpts = getIntPts(3)

    problem = False  # is the code broken?

    for i in range(len(basis)):  # for every element region...
        for j in range(len(basis[i])):  # for every integration point...
            for k in range(8):  # for every basis function...
                # get the appropriate values form one of the bases
                f, fd1, fd2, fd3 = threeDBasisFunc(intpts[i][j][0][0],
                                                   intpts[i][j][0][1],
                                                   intpts[i][j][0][2],
                                                   k)
                test_a = (f != basis[i][j][k][0])
                test_b = (fd1 != basis[i][j][k][1])
                test_c = (fd2 != basis[i][j][k][2])
                if (test_a or test_b or test_c):
                    problem = True  # the stored value is incorrect

    print('Is 3D broken?:', problem)