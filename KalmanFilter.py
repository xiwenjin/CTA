#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from numpy.random import normal
from numpy.linalg import inv as inverse


class KalmanFilter:
    def __init__(self, A, B, H, Q, R):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R    
        
        # update the into matrix format
        self.update_variable()

    def update_variable(self):
        '''
        convert attributes to matrix format in case they are single numbers 
        or input as a list.
        '''
        for v in ['A', 'B', 'H', 'Q', 'R']:
            if isinstance(getattr(self, v), int) or isinstance(getattr(self, v), float):
                att = getattr(self, v)
                setattr(self, v, np.array([[att]]))

            if isinstance(getattr(self, v), list):
                att = getattr(self, v)
                setattr(self, v, np.array(att).reshape(len(att), -1))
        
    def predict(self, x, p, u):
        # type convert in case they are single numbers or input as a list.
        if isinstance(x, int) or isinstance(x, float):
            x = np.array([[x]])
        if isinstance(x, list):
            x = np.array(x).reshape(len(x), -1)
            
        if isinstance(p, int) or isinstance(p, float):
            p = np.array([[p]])
        if isinstance(x, list):
            p = np.array(u).reshape(len(p), -1)
            
        if isinstance(u, int) or isinstance(u, float):
            u = np.array([[u]])
        if isinstance(x, list):
            u = np.array(u).reshape(len(u), -1)
        
        # calculation
        x_ = np.dot(self.A, x) + np.dot(self.B, u)
        P_ = np.dot(np.dot(self.A, p), self.A.T) + self.Q
        
        return x_, P_
    
    def update(self, z, x_, P_):
        # convert obervation into matrix format
        if isinstance(z, int) or isinstance(z, float):
            z = np.array([[z]])
        if isinstance(z, list):
            z = np.array(z).reshape(len(z), -1)
        
        # the update process
        S = np.dot(np.dot(self.H, P_), self.H.T) + self.R
        K = np.dot(np.dot(P_, self.H.T), inverse(S))
        eps = z - np.dot(self.H, x_)
        
        x = x_ + np.dot(K, eps)
        P = P_ - np.dot(np.dot(K, self.H), P_)
        
        return x, P
    
    def tracking(self, z, x, p, u=0):
        x_, P_ = self.predict(x, p, u)
        x, P = self.update(z, x_, P_)
        return x_, P_, x, P
    
    def to_array(self, ds, pos=0):
        return np.array(list(map(lambda x: x.flatten()[pos], ds)))
    
    def filt(self, data, x, p, u=0):
        '''
        data: the data series needed to be filtered
        x: the initial states
        p: the initial covariance matrix of the state porcess
        u: control input for the state process
        '''
        updates, preds = [], []
        p_updates, p_preds = [], []
        for d in data:
            x_, P_, x, P = self.tracking(d, x, p, u=u)
            preds.append(x_)
            p_preds.append(P_)
            updates.append(x)
            p_updates.append(P)
        
        return updates, p_updates, preds, p_preds
            
        

