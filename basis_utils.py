# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:09:38 2024

@author: AldenYellowhorse
"""


class Basis:
    """This function groups all basis functions together for a linear, Lagrage
    basis function set. This function set varies in dimension from 1 to 3D on
    command."""

    @staticmethod
    def N(xi='empty', nu='empty', zeta='empty', is_down: bool = True):
        """This function gets the basis function and its derivatives for a
        given set of input values (xi, nu and zeta in the parent space). This
        is the element space and not real space."""
        if nu != 'empty' and is_valid_number(nu):
            Nvalue = 0.5 - 0.5 * xi
            dN_dxi = -0.5
            return NValue, dN_dxi
        elif ()