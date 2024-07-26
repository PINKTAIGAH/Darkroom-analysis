import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from utils import find_max
from scipy.integrate import quad

import numpy as np
import matplotlib.pyplot as plt

"""
########## Patent Classes ##########
"""

class ProbabilityDensityFunction(object):
    """
    Parent class containing common methods and members to be used by all pdf classes
    """

    def __init__(self, bounds):

        if (not isinstance(bounds, tuple)):
            raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
        if (not len(bounds) == 2):
            raise ValueError("Variable bound must have form (boundMin, boundMax)")
        if (not bounds[0] < bounds[1]):
            raise ValueError("First element in tuple must be smaller than second")
        
        # Initialise class variables
        self.boundMin, self.boundMax = bounds
        # Initialise list to hold randomly generated mass values
        self.mass = []

    def integrate(self, limits):
        """
        Evaluate the integral of the pdf within the specified bounds
        ##### NOTE: Integral is not normalised within the specified bounds of the class #####
        """

        if (not isinstance(limits, tuple)):
            raise TypeError("Variable bound must be a tuple with the form (limitMin, limitMax)")
        if (not len(limits) == 2):
            raise ValueError("Variable bound must have form (limitMin, limitMax)")
        if (not limits[0] < limits[1]):
            raise ValueError("First element in tuple must be smaller than second")
        if (not limits[0] >= self.boundMin):
            raise ValueError("Lower integral limit must be larger than lower bound of pdf")
        if (not limits[1] <= self.boundMax):
            raise ValueError("Higher integral limit must be smaller than upper bound of pdf")
    
        limitLow, limitHigh = limits
        integralResult, IntegralError = quad(self._evaluate, limitLow, limitHigh) 
        return integralResult
    
    def getMass(self,):
        """
        Return numpy array containing all generated values
        """

        return np.array(self.mass)

class MinimisationStatistic(object):
    """
    Class containing minimisation statistic to be for pdf fitting
    """

    def __init__(self, pdf, data):

        self.pdf = pdf
        self.data = data

    def setData(self, data):
        """
        Assign data class member to new dataset for the reuse of this class
        """
        
        self.data = data

    def findNormalisationFactor(self,):
        """
        Find integral of pdf 
        """
        
        # Define integration limits
        normalisationLimits = (self.pdf.boundMin, self.pdf.boundMax)

        return self.pdf.integrate(normalisationLimits)

"""
########## Child Classes ###########
"""

class NegativeLogLikelihood(MinimisationStatistic):
    """
    Class constaining Negative log likelihood statistic for optimisation
    """

    def __init__(self, pdf, data,):

        # Initialise parent class
        super().__init__(pdf, data)

    def evaluateNull(self, slope, intercept):
        """
        evaluate negative log likelihood statisctic for passed parameters
        """

        # set new parameters
        self.pdf.setParameters(slope=slope, intercept=intercept)

        # compute likelyhood using passed parameters
        normalisation = self.pdf.integrate((self.pdf.boundMin, self.pdf.boundMax))
        likelihood = self.pdf._evaluate(self.data,) / normalisation 
        # set any negative likelihoods to neglegable positive values
        if (likelihood <= 0).any():
            likelihood[likelihood<=0] = 1e-6
        loglikelihood = np.log(likelihood)
        return -loglikelihood.sum()

    def evaluateAlternative(self, signalFraction, slope, intercept):
        """
        evaluate negative log likelihood statisctic for passed parameters
        """

        # set new parameters
        self.pdf.setParameters(signalFraction, slope=slope, intercept=intercept)

        # compute likelyhood using passed parameters
        normalisation = self.pdf.integrate((self.pdf.boundMin, self.pdf.boundMax))
        likelihood = self.pdf._evaluate(self.data,) / normalisation
        # set any negative likelihoods to neglegable positive values
        if (likelihood <= 0).any():
            likelihood[likelihood<=0] = 1e-6
        loglikelihood = np.log(likelihood)
        return -loglikelihood.sum()


class Linear(ProbabilityDensityFunction):
    """
    Class that will generate a random value according to a linear distribution using a box method
    """

    def __init__(self, slope, intercept, bounds):

        # Initialise parent class
        super().__init__(bounds)

        # Initialise class variables
        self.intercept = intercept
        self.slope = slope
        # Find maximum value of the distribution within the bounds
        self.maxValue = find_max(self._evaluate, self.boundMin, self.boundMax)

    def _evaluate(self, x,):
        """
        Evaluate the linear function of the distribution
        NOTE: Returns un-normalised values
        """
        
        return self.intercept + self.slope * x

        
    def setParameters(self, slope=None, intercept=None):
        """
        Set passed variables as parameters for pdf
        """

        # Use default values for parameters of none are passed through kwargs
        if not slope == None:                   self.slope = slope
        if not intercept == None:               self.intercept = intercept       

    