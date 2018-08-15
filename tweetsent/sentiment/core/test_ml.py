'''
    Unit tests for ml.py
'''

from .ml   import *
import os

def test_get_added_name():
    '''
        Tests ml.get_added_name
    '''
    paths       = [
        os.path.join('a', 'b', 'c', 'd.ext'),
        os.path.join('a', 'b', 'c')]

    expected    = [
        os.path.join('a', 'b', 'c', 'd_save.ext'),
        os.path.join('a', 'b', 'c_save')]

    results     = [get_added_name(x) for x in paths]

    assert expected == results
