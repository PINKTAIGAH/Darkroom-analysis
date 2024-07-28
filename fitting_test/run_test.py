from fitting_test import *
import numpy as np
import matplotlib.pyplot as plt
import random
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

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

class NegativeLogLikelihood(MinimisationStatistic):
    """
    Class constaining Negative log likelihood statistic for optimisation
    """

    def __init__(self, pdf, data,):

        # Initialise parent class
        super().__init__(pdf, data)

    def evaluate(self, slope, intercept):
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

class ChiSquared(MinimisationStatistic):
    """
    Class constaining chi squared statistic for optimisation
    """

    def __init__(self, pdf, data, dataUncertanty):

        # Initialise parent class
        super().__init__(pdf, data)
        # Define class members
        self.dataErrors = dataUncertanty

    def evaluate(self, *fittingParameters):
        """
        Evaluate chi squared statisctic for passed parameters
        """

        # Assign fitting parametes
        match len(fittingParameters):
            case 2: 
                slope, intercept = fittingParameters
                signalFraction = None
            case 3:
                signalFraction, slope, intercept = fittingParameters
            case _:
                raise ValueError("Variable fitting parameter has too many or too few elements. Should have 2 or 3")

        # Set new parameters
        self.pdf.setParameters(slope=slope, intercept=intercept)

        # Compute predicted value by model
        predicted_data = self.pdf._evaluate(self.data,)

        return (predicted_data-self.data)**2/self.dataUncertanty

time = np.linspace(0, 10, 100)
signal = time*2 + 4 + np.random.normal(3, 1, size=time.shape)

plt.scatter(time, signal, s=2.5)
plt.show()

hypothesis = Linear(1, 1, (0, 10))

plt.scatter(time, signal, s=2.5, label="data")
plt.plot(time, hypothesis._evaluate(time), label="hypothesis")
plt.legend()
plt.show()

nnl = NegativeLogLikelihood(hypothesis, time) 

minimiser = Minuit(nnl.evaluate, slope=2.1, intercept=5)
# minimiser.limits = [(0, None), (0, 10)]

results = minimiser.migrad()

hypothesis.setParameters(*results.values)
plt.scatter(time, signal, s=2.5, label="data")
plt.plot(time, hypothesis._evaluate(time), label="hypothesis")
plt.legend()
plt.show()
