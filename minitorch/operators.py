"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if x is less than y."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if x is equal to y."""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float, eps: float = 1e-2) -> bool:
    """Checks if two numbers are close to each other."""
    return abs(x - y) < eps


def sigmoid(x: float) -> float:
    """Calculates the sigmoid of x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Calculates the ReLU of x."""
    return max(0.0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm of x."""
    if x <= 0:
        raise ValueError("Logarithm undefined for non-positive values.")
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential of x."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the inverse of x."""
    if x == 0:
        raise ValueError("Inverse undefined for zero.")
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Calculates the derivative of log at y."""
    if y <= 0:
        raise ValueError("Logarithm derivative undefined for non-positive values.")
    return x / y


def inv_back(x: float, y: float) -> float:
    """Calculates the derivative of inv at y."""
    if x == 0:
        raise ValueError("Inverse derivative undefined for zero.")
    return -y / x**2


def relu_back(x: float, y: float) -> float:
    """Calculates the derivative of ReLU at y."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float], it: Iterable[float]) -> Iterable[float]:
    """Maps a function over a list."""
    return [fn(x) for x in it]


def zipWith(
    fn: Callable[[float, float], float], it1: Iterable[float], it2: Iterable[float]
) -> Iterable[float]:
    """Zips two lists together with a function."""
    return [fn(x, y) for x, y in zip(it1, it2)]


def reduce(fn: Callable[[float, float], float], it: Iterable[float]) -> float:
    """Reduces a list to a single value."""
    if len(it) == 0:
        return 0.0
    if len(it) == 1:
        return it[0]
    return fn(it[0], reduce(fn, it[1:]))


def negList(it: Iterable[float]) -> Iterable[float]:
    """Negates a list."""
    return map(neg, it)


def addLists(it1: Iterable[float], it2: Iterable[float]) -> Iterable[float]:
    """Adds two lists together."""
    return zipWith(add, it1, it2)


def sum(it: Iterable[float]) -> float:
    """Sums a list."""
    return reduce(add, it)


def prod(it: Iterable[float]) -> float:
    """Takes the product of a list."""
    return reduce(mul, it)
