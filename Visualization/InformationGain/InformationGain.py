#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import matplotlib.pyplot as plt

def gini(p):
    return p * (1 - p) + (1 - p) *(1 - (1 - p))

def entropy(p):
    if p == 0:
        return None
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def error(p):
    return 1 - np.max([p,1 - p])

# We create a vector x from 0 to 1 with values every 0.01
x = np.arange(0.0,1.0,0.01)

# vector of entropy
x_h = [entropy(p) for p in x]
# we standardize
x_hsc = [e * 0.5 if e else None for e in x_h]

# vector of errors
x_e = [error(i) for i in x]

figure = plt.figure()
ax = plt.subplot(111)

data = [x_h,x_hsc,gini(x),x_e]
labels = ['Entropy','Entropy (scaled)','Gini impurity','Misclassification Error']
linestyles = ['-','-','--','-.']
colors = ['black','lightgray','red','green','cyan']
for i,lab,ls,c in zip(data,labels,linestyles,colors):
    line = ax.plot(x,i,label=lab,linestyle=ls,lw=2,color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()
