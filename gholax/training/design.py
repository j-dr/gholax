#!/usr/bin/env python3
#
# Some simple Python code to make a low discrepancy sequence
# in multiple dimensions based upon:
# http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
#
import numpy as np


class QuasiRandomSequence:
    """A simple class to generate quasi-random, or low-discrepancy, sequences
    in multiple dimensions.  Based upon
    http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    This is currently written in pure Python, but if speed ever becomes
    an issue it could likely be heavily optimized using Numba.
    """

    def __init__(self, n_dim=1, seed=0.5):
        """Initializes the class and sets the vector alpha.  The sequences
        will consist of n_dim-dimensional vectors in [0,1)^n_dim."""
        self.seed = seed
        # Get the generalized golden ratio for n_dim dimensions, aka
        # the "harmonius numbers" of Hans van de Laan.  This solves
        # x^{d+1}=x+1 using Newton-Raphson:
        phi = 1.0
        for i in range(20):
            phi = phi - (pow(phi, n_dim + 1) - phi - 1) / (
                (n_dim + 1) * pow(phi, n_dim) - 1
            )
        self.phiinv = 1.0 / phi
        # and hence generate our base vector, alpha.
        self.alpha = np.array([(self.phiinv ** (i + 1)) % 1 for i in range(n_dim)])
        #

    def __call__(self, n):
        """Returns the first n vectors in the (Korobov) sequence, with x[i,:]
        being the i'th vector."""
        tmp = self.seed + 0 * self.alpha
        ret = np.zeros((n, self.alpha.size))
        for i in range(n):
            tmp += self.alpha
            ret[i, :] = tmp
        ret = ret % 1
        return ret
        #

    def get_vector(self, n):
        """Returns just the nth vector in the sequence, starting at n=1."""
        ret = (self.seed + n * self.alpha) % 1
        return ret
        #


class Design:
    """Returns a vector of parameters to sample at."""

    def __init__(self, pmin, pmax, scheme="qrs", ntot=None):
        """Takes the range of parameters, only varying those where pmin!=pmax.
        The dimensions of pmin and pmax should equal the total number
        of parameters for the model."""
        self.pmin = pmin
        self.pmax = pmax
        self.ndim = np.sum(np.abs(pmax - pmin) > 0)
        self.scheme = scheme
        if scheme == "qrs":
            self.seq = QuasiRandomSequence(self.ndim)
        elif scheme == "sobol":
            import chaospy as cp

            self.seq = cp.create_sobol_samples(ntot, self.ndim).T
        elif scheme == "lhs":
            import pyDOE

            self.seq = pyDOE.lhs(self.ndim, samples=ntot)
        elif scheme == "lhs_maximin":
            import pyDOE

            self.seq = pyDOE.lhs(self.ndim, samples=ntot, criterion="maximin")
        elif scheme != "from_file":
            self.seq = np.genfromtxt(scheme)

    def sample(self, n):
        """Gets the n'th vector of parameter values."""
        pars = self.pmin.copy()
        vary = np.abs(self.pmax - self.pmin) > 0
        if self.scheme == "qrs":
            v = self.seq.get_vector(n)
        else:
            v = self.seq[n]

        if self.scheme != "from_file":
            pars[vary] = self.pmin[vary] + (self.pmax - self.pmin)[vary] * v
        else:
            pars = self.seq[n]
        return pars
