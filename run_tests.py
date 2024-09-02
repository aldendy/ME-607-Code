# -*- coding: utf-8 -*-
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
