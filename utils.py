# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:43:44 2024

@author: AldenYellowhorse
"""


def is_scalar(number):
    """Figure out if the number is a scalar."""
    return isinstance(number, float) or isinstance(number, int)
