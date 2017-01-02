#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example of weighted class prediction weighted majority vote
print('Class prediction: {}'.format(np.argmax(np.bincount([0, 0, 1], weights = [0.2, 0.2, 0.6]))))
ex = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])
p = np.average(ex, axis = 0, weights = [0.2, 0.2, 0.6])
print('Probability associated to each prediction: {}'.format(p))
print('Predicted class after majority vote: {}'.format(np.argmax(p)))
