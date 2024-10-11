# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # normal equation: w = (X^T * X)^(-1) * X^T * y
    w = np.linalg.solve(tx.T.dot(tx), tx.T @ y) # can put both : either do .dot() or @, equiv
    
    # Calculate error and MSE
    e = y - tx.dot(w)
    mse = np.mean(e**2) / 2
    
    return w, mse
