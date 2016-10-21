#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(values, classes, classifier, test_start = None, resolution = 0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(classes))])

    x1_min, x1_max = values[:,0].min()-1, values[:,0].max()+1
    x2_min, x2_max = values[:,1].min()-1, values[:,1].max()+1

    # Here we create a new set of values that will fill the graph
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution), np.arange(x2_min,x2_max,resolution))
    class_prediction = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    class_prediction = class_prediction.reshape(xx1.shape)

    # Fill the graph with correct colors
    plt.contourf(xx1,xx2,class_prediction,alpha=0.4,cmap=cmap)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)

    # plot class samples, enumerate associates an iterator to a value
    for i, cl in enumerate(np.unique(classes)):
        plt.scatter(x=values[classes==cl,0],y=values[classes==cl,1],alpha=0.8,c=cmap(i),marker=markers[i],label=cl)

    # Highlight test samples
    if test_start:
        test_values, test_classes = values[test_start:], classes[test_start:]
        plt.scatter(test_values[:,0],test_values[:,1],c='',alpha=1.0,linewidth=1,marker='o',s=55,label='Test')
