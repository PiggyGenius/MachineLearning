#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd

class LogisticRegression(object):

    def __init__(self,rate,epoch):
        self.rate = rate
        self.epoch = epoch
        self.threshold = 0
        self.cost = []
        self.weight = None

    def fit(self,values,classes):
        self.weight = np.zeros(values.shape[1])
        for i in range(self.epoch):
            errors = classes - self.activation(values)
            self.weight += self.rate * np.dot(errors.T,values)
            self.threshold += self.rate * errors.sum()

    def netInput(self,values):
        return np.dot(values,self.weight) + self.threshold

    def sigmoid(self,net_input):
        return 1.0/(1.0+np.exp(-net_input))

    def activation(self,values):
        return self.sigmoid(self.netInput(values))

    def predict(self,values):
        return np.where(self.activation(values) >= 0.5,1,0)
    
    def predict_proba(self,values):
        return self.activation(values)

    def normalize(self,values):
        values[:,0] = (values[:,0] - values[:,0].mean()) / values[:,0].std()
        values[:,1] = (values[:,1] - values[:,1].mean()) / values[:,1].std()
        return values

    def shuffle(self,values,classes):
        values_classes = np.concatenate((values,classes[np.newaxis].T),axis=1)
        np.random.shuffle(values_classes)
        return values_classes[:,0:2], values_classes[:,2]
