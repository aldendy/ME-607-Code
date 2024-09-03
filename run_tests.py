# In this code, we automate the process of running all applicable tests for
# the finite-element code in ME 607.

"""
Created on Mon Sep  2 13:47:37 2024

@author: AldenYellowhorse
"""

import os
from unittest import TestLoader, TextTestRunner

loader = TestLoader()

# use 'test*.py' to run all tests
suite = loader.discover(start_dir=os.getcwd(), pattern='test*.py')

runner = TextTestRunner()
runner.run(suite)
