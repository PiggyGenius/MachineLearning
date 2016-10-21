#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    def __init__(self,eta=0.01,n_iter=10):
        """Perceptron classifier.
        Parameters
        ------------
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_iter : int
            Passes over the training dataset.
        Attributes
        -----------
        w_ : 1d-array
            Weights after fitting.
        errors_ : list
            Number of misclassifications in every epoch.
        """
        self.eta = eta
        self.n_iter = n_iter
    def train(self,x,y):
        """Train model on data.
        Parameters
        ----------
        x : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : object
        """
        # Threshold seems to be defined as 0, why isn't it a parameter ?
        self.w_ = np.zeros(1+x.shape[1]) # add _ at the end if variable not declared in init.
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi,target in zip(x,y): # xi takes x values and target y values with zip, list fusion.
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update # We update the threshold without xi
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self,x):
        """ Calculate net input: z = W.X+W0, with W0 the threshold."""
        return np.dot(x,self.w_[1:]) + self.w_[0]
    def predict(self,x):
        """ Return class label after unit step """
        # Because we added -theta, threshold, in w so the new boundary is 0.
        return np.where(self.net_input(x) >= 0.0,1,-1)
    def plot_decision(self,x,y,resolution=0.02):
        markers = ('s','x','o','^','v')
        colors = ('red','blue','lightgreen','gray','cyan')
        # creates a colormap
        cmap = ListedColormap(colors[:len(np.unique(y))])
        # plot decisision surface
        x1_min,x1_max = x[:,0].min()-1,x[:,0].max()+1
        x2_min,x2_max = x[:,1].min()-1,x[:,1].max()+1
        # np.arrange creates a vector that starts from i, filled with every i+step,ends at j-step
        # np.meshgrid creates an array of two matrices
        # First one is filled with len(vector_1) raws where each raw is vector_1
        # Second one is filled with len(vector_2) raws where each raw is vector_2
        # xx1 is matrix_1 and xx2 is matrix_2
        xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
        # ravel transforms matrix into a vector where each value of matrix in a value of vector
        # .T gives the trampose of a matrix/vector
        # Since we already trained the model we already found the correct w vector.
        # Now, we just need to create a new data set composed of every point in graph.
        # The model we associate each point to 1 or -1 giving us the blue and red area.
        # What looks like a function plot seperating is simply blue and red being close
        z = self.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
        # makes z take the shape of xx1 which is a 305*235 matrix instead of being a vector
        t = np.array([xx1.ravel(),xx2.ravel()]).T
        z = z.reshape(xx1.shape)
        #plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
        plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
        plt.xlim(xx1.min(),xx1.max())
        plt.ylim(xx2.min(),xx2.max())
        # np.unique create an array of unique values
        # enumerate adds a counter to an iterable, idx takes value of counter and cl of y[idx]
        for idx,cl in enumerate(np.unique(y)):
            # x[y==cl,0] selects x coordinates of x if on the same indice y=cl
            plt.scatter(x=x[y==cl,0],y=x[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)




##############       Notes about the perceptron model         ###############
#                                                                           #
# We can initialize new Perceptron objects with a given learning rate       #
# eta and n_iter, the number of passes over the training set.               #
# Via the fit method we initialize the weighs in self.w_ to a zero vector   #
# of dimension m+1 where m stands for the number of features in the data    #
# and we add 1 for the zero-weight, the threshold.                          #
#                                                                           #
# Afer the weights have been initialized, the fit method loops over all     #
# samples in the training set and updates the weights according to the      #
# learning rule.                                                            #
# The class labels are predicted by the predict method which is also called #
# in the fit method to predict the class label for the weight update.       #
# Predict can alose be used to predict the class labels of new data after   #
# we have fitted our model.                                                 #
# Be sure to notice that the threshold is at 0 so W0=0 at init.             #
#                                                                           #
# We also collect the number of misclassifications during each epoch in the #
# list self.errors_ so that we can later analyze how well our perceptron    #
# performed during the training.                                            #
# The np.dot function that is used in the net_input simplu calculates z.    #
#                                                                           #
##############       Notes about the perceptron model         ###############
