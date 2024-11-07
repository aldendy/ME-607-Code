# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:43:44 2024

@author: AldenYellowhorse
"""


class domain_names:
    one_d = ['interior', 'left bound', 'right bound']
    two_d = ['interior', 'left bound', 'bottom bound', 'right bound',
             'top bound']
    three_d = ['-xy bnd', '+xy bnd', '-yz bnd', '+yz bnd', '-xz bnd',
               '+xz bnd']


def is_scalar(number):
    """Figure out if the number is a scalar."""
    return isinstance(number, float) or isinstance(number, int)
