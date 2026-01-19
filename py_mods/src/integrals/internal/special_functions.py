import numpy as np
import scipy


def pochhammer(a: float, k: int) -> float:
    """
    Pochhammer symbol of the falling factorial (a)_k defined from gamma functions.

    Defined as:

    (a)_k = a * (a-1) * (a-2) * ... * (a-n+1)

    Implemented using only the Gamma Function due to portability with standard
    libraries in other codes:

    (a)_k = Gamma(a+k) / Gamma(a)

    Parameters
    ----------
    a : float
        a value in definition of (a)_k.
    k : int
        Non-negative integer.

    Returns
    -------
    float
        The value of the Pochhammer symbol (a)_k.
    """
    return float(scipy.special.gamma(a + k) / scipy.special.gamma(a))


def M(a: float, b: float, x: float, k: int) -> float:
    """
    Definition of the confluent Hypergeometric function as a series

    Defined as:

    M(a, b, x) = sum_{k=0}^k (a)_k / (k! * (b)_k) * x^k

    Uses the Pochhammer symbol. Implemented using only the Gamma Function due
    to portability with standard libraries in other codes.

    Parameters
    ----------
    a : float
        a value in definition of M(a, b, x).
    b : float
        b value in definition of M(a, b, x).
    x : float
        b value in definition of M(a, b, x).
    k : int
        Non-negative integer. Upper limit of the series summation.

    Returns
    -------
    float
        The value of the M(a, b, x) series computed with k terms.
    """
    term = 1.0
    m_sum = 1.0

    for n in range(1, k):
        numerator = (a + n - 1) * x
        denominator = (b + n - 1) * n

        term *= numerator / denominator
        m_sum += term

    return float(m_sum)


def boys_hypergeom(n: int, x: float, k: int):
    """
    Definition of the Boys function from the confluent Hypergeometric function.

    Defined as:

    F_n(x) = M(n + 1/2, n+3/2, -x) / (2n+1)

    Uses the confluent Hypergeometric function as a finite series.

    Parameters
    ----------
    n : int
        Order of the Boys function.
    x : float
        `x` value.
    k : int
        Non-negative integer. Upper limit of the series summation in the Hypergeometric function.

    Returns
    -------
    float
        The value of F_n(x) computed with k terms.

    Notes
    -------
    Convergence was tested and k = 80 converges with other definitions up to x = 29.130.
    """
    a = n + 0.5
    b = n + 1.5

    return M(a, b, -x, k + 10) / (2 * n + 1)


def boys_upward_order_n(F_0: float, n: int, x: float) -> float:
    """
    Upward recursion relation of the Boys function.

    Parameter
    ----------
    F_0 : float
        Value of the 0th order Boys function.
    n : int
        Order of the Boys function.
    x : float
        `x` value.

    Returns
    -------
    float
        The value of F_n(x).

    Notes
    -------
    Works up to k = 14, after that it breaks due to division by 0.
    """
    if n == 0:
        return F_0

    F_n1 = F_0

    for i in range(n):
        F_n1 = ((2 * i + 1) * F_n1 - np.exp(-x)) / (2 * x)

    return F_n1
