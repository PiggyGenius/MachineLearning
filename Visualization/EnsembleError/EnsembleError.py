#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.misc import comb

# We define a function that gives us the error probability of the ensemble
def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probas = [comb(n_classifier, k) * error**k * (1 - error)**(n_classifier - k) for k in range(k_start, n_classifier + 1)]
    return sum(probas)

print(ensemble_error(11, 0.25))
# Create a list of values from 0.0 to 1 with step 0.01
error_range = np.arange(0.0, 1.01, 0.01)
ensemble_errors = [ensemble_error(11, error) for error in error_range]
plt.plot(error_range, ensemble_errors, label = 'Ensemble error', linewidth = 2)
plt.plot(error_range, error_range, linestyle = '--', label = 'Base error', linewidth = 2)
plt.xlabel('Base error')
plt.ylabel('Base / Ensemble error')
plt.legend(loc = 'upper left')
plt.grid()
plt.show()
