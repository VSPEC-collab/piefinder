"""
Tests for piefinder.jax.functions
"""

from piefinder.jax.functions import BaseFunction, add, multiply

def test_basefunction():
    """
    Test the BaseFunction class
    """
    def func(t):
        return t
    func = BaseFunction(func)
    assert func(1) == 1

def test_add():
    """
    Test the add function
    """
    def func1(t):
        return t
    def func2(t):
        return t
    func1 = BaseFunction(func1)
    func2 = BaseFunction(func2)
    func = add(func1, func2)
    assert func(1) == 2

def test_multiply():
    """
    Test the multiply function
    """
    def func1(t):
        return t
    k = 2
    func1 = BaseFunction(func1)
    func = multiply(func1, k)
    assert func(1) == 2