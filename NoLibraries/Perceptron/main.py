#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None) #  Read csv file, loads of options.
# Do a df.head() to know which lines/columns to select
y = df.iloc[0:100, 4].values # picks lines from 0 to 99, column 4 which is the names,values for data.
y = np.where(y == 'Iris-setosa', -1, 1) # Replace names by -1 and 1

# extract sepal length and petal length
x = df.iloc[0:100, [0, 2]].values

# raw data
# f1 = plt.figure(1) # Put first figure in f1
# # X[50,0] is x coordinates and X[50,1] is y
# plt.scatter(x[:50, 0], x[:50, 1],color='red', marker='o', label='setosa')
# plt.scatter(x[50:100, 0], x[50:100, 1],color='blue', marker='x', label='versicolor')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left')
# plt.tight_layout() # Removes unecessary border
# #plt.savefig('./images/02_06.png', dpi=300)

# error function
# f2 = plt.figure(2)
ppn = Perceptron(eta=0.1,n_iter=10) # Perceptron object
ppn.train(x,y) # Training the model
# plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of missclassifications')
# plt.tight_layout()

# Decision graph
#f3 = plt.figure(3)
ppn.plot_decision(x,y)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()

plt.show()
