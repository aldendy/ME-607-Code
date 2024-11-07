# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:24:15 2024

@author: AldenYellowhorse
"""

from math import floor
from utils import is_scalar, domain_names


class IntegrationPoint:
    """Encapsulate an integration point as a new data type for better code
    clarity."""

    def __init__(self, coordinate: list, weight: float):
        """Every integration point is defined by a vector if 1, 2 or 3
        dimensions and an associated weight. The coordinate must be a list of
        1, 2 or 3 elements in length. Weight must be a scalar."""
        self.point = coordinate  # must be a list
        self.weight = weight     # must be a scalar
        self.check_values()

    def check_values(self):
        """Check the user input and throw exceptions if something is wrong."""
        if not isinstance(self.point, list):
            raise TypeError("expected list coordinate for integration point")
        for value in self.point:
            if not is_scalar(value):
                raise TypeError("expected integer or float in coordinate")
        if not is_scalar(self.weight):
            raise TypeError(('expected scalar value for integration point '
                            'weight'))


class LinearElement:
    """This class defines an element of 1, 2 or 3 dimensions which contains all
    the right utilities for calculation based on this element type. The
    intention is to encapsulate all the element complication in this class. If
    linear elements are not desired, a different class can be used."""

    def __init__(self, num_integration_points: int = 1,
                 num_dimensions: int = 1):
        """By default, set the number of integration points to 1."""
        self.__data = {'num quad points': num_integration_points,
                       'num dimensions': num_dimensions}
        self.__data['element quad points'] = self.__element_quad_points()

    @staticmethod
    def quad_points(num_points: int = 1, num_dimensions: int = 1) -> dict:
        """Return a dictionary where the keys are integers beginning with '0'
        and the values are integration point objects for the element dimension
        and number of quadrature points in a single dimension. The integration
        takes place over a 1D domain of [-1, 1], a 2D domain of [-1, 1][-1, 1]
        etc."""
        if num_points == 1:  # for a single integration point...
            points = [0]
            if num_dimensions == 0:
                # if we integrate at the boundary of a 1D line...
                weights = [1]
            else:
                weights = [2]
        elif num_points == 2:  # for two integration points...
            points = [-1/3**0.5, 1/3**0.5]
            weights = [1, 1]
        elif num_points == 3:
            points = [-(3/5)**0.5, 0, (3/5)**0.5]
            weights = [5/9, 8/9, 5/9]
        elif num_points == 4:
            small = (3/7 - 2/7 * (6/5)**0.5)**0.5
            large = (3/7 + 2/7 * (6/5)**0.5)**0.5
            points = [-large, -small, small, large]
            w1, w2 = (18 + 30**0.5) / 36, (18 - 30**0.5) / 36
            weights = [w2, w1, w1, w2]
        elif num_points == 5:
            small = 1/3 * (5 - 2 * (10/7)**0.5)**0.5
            large = 1/3 * (5 + 2 * (10/7)**0.5)**0.5
            w1, w2 = (322 + 13 * 70**0.5) / 900, (322 - 13 * 70**0.5) / 900
            points = [-large, -small, 0, small, large]
            weights = [w2, w1, 128/225, w1, w2]
        else:
            raise ValueError("expected either 1 or 2 integration points")

        quad_pnts = {}

        for number in range(num_points**num_dimensions):
            xi_i = number % num_points
            nu_i = floor(number/num_points) % num_points
            ze_i = floor(number/num_points**2) % num_points
            xi, nu, ze = points[xi_i], points[nu_i], points[ze_i]

            if num_dimensions == 1:
                quad_pnts[number] = IntegrationPoint([xi], weights[xi_i])
            elif num_dimensions == 2:
                weight = weights[xi_i] * weights[nu_i]
                quad_pnts[number] = IntegrationPoint([xi, nu], weight)
            elif num_dimensions == 3:
                weight = weights[xi_i] * weights[nu_i] * weights[ze_i]
                quad_pnts[number] = IntegrationPoint([xi, nu, ze], weight)
        return quad_pnts

    def __element_quad_points(self) -> dict[dict[IntegrationPoint]]:
        """This method assembles the quadrature points into a single dictionary
        where the keys are the domain number (0, 1, 2 ...) and the values are
        dictionaries of integration points."""
        elem_quad_points = {}
        num = self.__data['num quad points']
        if self.__data['num dimensions'] == 1:
            elem_quad_points[] = self.quad_points(num, 1)
            elem_quad_points['left bound'] = self.quad_points(num, 0)
            elem_quad_points['right bound'] = self.quad_points(num, 0)
        elif self.__data['num dimensions'] == 2:
            elem_quad_points['interior'] = self.quad_points(num, 2)
            elem_quad_points['left bound'] = self.quad_points(num, 1)
            elem_quad_points['bottom bound'] = self.quad_points(num, 1)
            elem_quad_points['right bound'] = self.quad_points(num, 1)
            elem_quad_points['top bound'] = self.quad_points(num, 1)
        elif self.__data['num dimensions'] == 3:
            elem_quad_points['interior'] = self.quad_points(num, 3)
            for name in domain_names.three_d:
                elem_quad_points[name] = self.quad_points(num, 2)
        return elem_quad_points

    @property
    def element_quad_points(self) -> dict[dict[IntegrationPoint]]:
        """Return a dictionary of dictionaries of integration point objects
        grouped by element region."""
        return self.__data['element quad points']
