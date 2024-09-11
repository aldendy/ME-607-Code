# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:25:15 2024

@author: AldenYellowhorse
"""

import unittest
from elements import LinearElement


class TestBasis(unittest.TestCase):
    """Test the properties of our basis."""

    def setUp(self):
        """Instantiate an element and test the characteristics of the gaussian
        quadrature."""
        self.element = LinearElement()

    def test_1D_1_point(self):
        """Check the 1D points and make sure they are returned correctly."""
        points = self.element.get_gaussian_quad_points(1, 1)
        # we should get a dictionary of a single point at [0]
        self.assertEqual(len(points), 1)
        # This point should be located at xi = 0
        self.assertEqual(points[0].point, [0])
        # It's weight should be '2'
        self.assertEqual(points[0].weight, 2)

    def test_1D_2_points(self):
        """With two points in 1D, the values are different."""
        points = self.element.get_gaussian_quad_points(2, 1)
        # we should get two points
        self.assertEqual(len(points), 2)
        # the points should be at [-1/3**0.5] and [1/3**0.5]
        self.assertEqual(points[0].point, [-1/3**0.5])
        self.assertEqual(points[1].point, [1/3**0.5])
        # weights should be '1' for both
        self.assertEqual(points[0].weight, 1)
        self.assertEqual(points[1].weight, 1)

    def test_2D_1_point(self):
        """Check the 2D points and make sure that they are returned correctly.
        """
        points = self.element.get_gaussian_quad_points(1, 2)
        # we should get a dictionary of a single point at [0, 0]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].point, [0, 0])
        # we should get a point with weight 4
        self.assertEqual(points[0].weight, 4)

    def test_2D_3_points(self):
        """Check this higher-order case."""
        points = self.element.get_gaussian_quad_points(3, 2)
        # we should get a dictionary of 9 points at the center and at offsets
        # of (3/5)**0.5
        self.assertEqual(len(points), 9)
        offset = (3/5)**0.5
        correct_points = [[-offset, -offset],
                          [0,       -offset],
                          [offset,  -offset],
                          [-offset,  0],
                          [0,        0],
                          [offset,   0],
                          [-offset,  offset],
                          [0,        offset],
                          [offset,   offset]]

        good_weights = [5/9 * 5/9,
                        5/9 * 8/9,
                        5/9 * 5/9,
                        8/9 * 5/9,
                        8/9 * 8/9,
                        8/9 * 5/9,
                        5/9 * 5/9,
                        5/9 * 8/9,
                        5/9 * 5/9]
        for i, point in enumerate(correct_points):
            self.assertEqual(points[i].point[0], correct_points[i][0])
            self.assertEqual(points[i].point[1], correct_points[i][1])
            # The weight should be right also
            self.assertAlmostEqual(points[i].weight, good_weights[i])

    def test_3D_1_point(self):
        """Test the points in a 3D domain."""
        points = self.element.get_gaussian_quad_points(1, 3)
        # we should get a dictionary of 1 point at [0, 0, 0]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].point, [0, 0, 0])
        # the weight should be '8'
        self.assertEqual(points[0].weight, 8)

    def test_3D_2_points(self):
        """Test the case of two points in a 3D domain."""
        points = self.element.get_gaussian_quad_points(2, 3)
        # we should get a dictionary of 8 points offset by 1/3**0.5 from origin
        self.assertEqual(len(points), 8)
        offset = 1/3**0.5
        correct_points = [[-offset, -offset, -offset],
                          [offset,  -offset, -offset],
                          [-offset,  offset, -offset],
                          [offset,   offset, -offset],
                          [-offset, -offset,  offset],
                          [offset,  -offset,  offset],
                          [-offset,  offset,  offset],
                          [offset,   offset,  offset]]
        for index, point in enumerate(correct_points):
            # the points should be right
            self.assertEqual(points[index].point[0], point[0])
            self.assertEqual(points[index].point[1], point[1])
            self.assertEqual(points[index].point[2], point[2])
            # the weights should all be '1'
            self.assertEqual(points[index].weight, 1)
