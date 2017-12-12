# coding: utf-8

import numpy as np
import scipy.stats as scs


class MCMC(object):
    def __init__(self, step_size=1., loc=0., scale=1., pdf):
        """
        Sample from a 1D gaussian PDF with uniform step proposal.

        Parameters
        ----------
        loc, scale : float
            Mean and standard deviation for the gaus to sample from.
        step_size = float
            Step sized used symmetrically around the current step to propose
            the next one from a uniform PDF in ``[-step_size, step_size]``.
        """
        self.step_size = step_size
        self._pdf = pdf
        self.loc = loc
        self.scale = scale
        
    def _propose_step(self, xi):
        """
        Calculate the next proposed step from the current one from the
        step proposal PDF (here: uniform).

        Parameters
        ----------
        xi : float
            Current step from which the next positions is calculated.

        Returns
        -------
        xj : float
            Next proposed step.
        """
        #xj = np.random.normal(xi, self.step_size)
        xj = np.random.uniform(-self.step_size, self.step_size) + xi  # ist das echt sinnvoll?
        return xj

    def _accept_step(self, xi, xj):
        """
        Decide wether to accept the next step or not using the
        Metropolis-Hastings detailed balance condition.

        Parameters
        ----------
        xi : float
            Current step from which the next positions is calculated.
        xj : float
            Next proposed step.

        Returns
        -------
        acc : bool
            ``True``if the next step is accepted, ``False`` if not.
        """
        a = np.exp(-(xj-self.loc)**2/(2*self.scale**2)) / np.exp(-(xi-self.loc)**2/(2*self.scale**2))
        u = np.random.uniform(0,1)
        return(a>u)
        #return(True)
        
        
        
    def sample(self, x0, n=1):
        """
        Sample ``n`` points from the gaussian PDF using the MCMC algorithm.

        Parameters
        ----------
        x0 : float
            Start value where the Markov chain is started.
        n : int
            How many samples to create.

        Returns
        -------
        x : array-like
            Created sample points. Has length ``n``.
        """
        x = np.empty(n, dtype=float)
        x[0] = x0
        i = 1
        while i < len(x):
            s = self._propose_step(x[i-1])
            if self._accept_step(x[i-1], s):
                x[i] = s
                i = i+1
        return x
