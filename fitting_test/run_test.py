from fitting_test import *
import numpy as np
import matplotlib.pyplot as plt
import random


time = np.linspace(0, 10, 100)
signal = time*2 + 4 + np.random.normal(20, 1, size=time.shape)

plt.scatter(time, signal, s=0.5)
plt.show()

toybox_linear = Linear(3, 4, (0, 10))
toybox_statistic = NegativeLogLikelihood(toybox_linear, signal)

minimiser = Minuit(toybox_statistic.evaluateNull, slope=2, intercept=2)
results = minimiser.migrad()
print(results)