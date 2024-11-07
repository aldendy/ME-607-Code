# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:25:15 2024

@author: AldenYellowhorse
"""

import unittest
from elements import LinearElement


class TestQuadraturePoints(unittest.TestCase):
    """Test the properties of our basis."""

    def setUp(self):
        """Instantiate an element and test the characteristics of the gaussian
        quadrature."""
        self.element = LinearElement()

    def test_1D_1_point(self):
        """Check the 1D points and make sure they are returned correctly."""
        points = self.element.quad_points(1, 1)
        # we should get a dictionary of a single point at [0]
        self.assertEqual(len(points), 1)
        # This point should be located at xi = 0
        self.assertEqual(points[0].point, [0])
        # It's weight should be '2'
        self.assertEqual(points[0].weight, 2)

    def test_1D_2_points(self):
        """With two points in 1D, the values are different."""
        points = self.element.quad_points(2, 1)
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
        points = self.element.quad_points(1, 2)
        # we should get a dictionary of a single point at [0, 0]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].point, [0, 0])
        # we should get a point with weight 4
        self.assertEqual(points[0].weight, 4)

    def test_2D_3_points(self):
        """Check this higher-order case."""
        points = self.element.quad_points(3, 2)
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
            self.assertEqual(points[i].point, point)
            # The weight should be right also
            self.assertAlmostEqual(points[i].weight, good_weights[i])

    def test_3D_1_point(self):
        """Test the points in a 3D domain."""
        points = self.element.quad_points(1, 3)
        # we should get a dictionary of 1 point at [0, 0, 0]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].point, [0, 0, 0])
        # the weight should be '8'
        self.assertEqual(points[0].weight, 8)

    def test_3D_2_points(self):
        """Test the case of two points in a 3D domain."""
        points = self.element.quad_points(2, 3)
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
            self.assertEqual(points[index].point, point)
            # the weights should all be '1'
            self.assertEqual(points[index].weight, 1)


class TestElementQuadraturePoints(unittest.TestCase):
    """This class tests that quadrature points for the entire element are
    assembled correctly. For a 2D element, for example, we should have quad
    points for a 2D interior domain and 4 sets of points for 1D domains for the
    boundary."""

    def test_1D_1_point_element(self):
        """Check that the domains are generated correctly for a 1D element."""
        element = LinearElement(1, 1)
        # the element must have keys 'interior', 'left bound' and 'right bound'
        regions = element.element_quad_points.keys()
        right_regions = ['interior', 'left bound', 'right bound']
        self.assertEqual(right_regions, list(regions))

    def test_1D_2_point_element(self):
        """Check that the domains are generated correctly for a 2D element."""
        element = LinearElement(2, 2)
        # the element must have keys 'interior', 'left bound' and 'right bound'
        # 'bottom bound' and 'top bound'
        regions = element.element_quad_points.keys()
        right_regions = ['interior', 'left bound', 'bottom bound',
                         'right bound', 'top bound']
        self.assertEqual(right_regions, list(regions))
