# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:53:53 2018

@author: leeva
"""

import numpy as np

class Muon_decay:
    
    def __init__ (self,tau,n,sim):
        self.t = tau
        self.d_num = n
        self.runs = sim
        
    def r_single(self):
        y_p = []
        x_p = []
        for i in range (self.d_num):
            y_1 = np.random.uniform()
            y_p.append(y_1)
            x_1 = -self.t*np.log(1-y_p[i])
            x_p.append(x_1)
        return(x_p)
        
    def r_full(self):
        y_prime = []
        x_prime = []
        tau = [] 
        for i in range (self.runs):
            for i in range (self.d_num):
                y_1 = np.random.uniform()
                y_prime.append(y_1)
                x_1 = -self.t*np.log(1-y_prime[i])
                x_prime.append(x_1)
            np.array(x_prime)
            tau_avg = np.mean(x_prime)
            tau.append(tau_avg)
            del x_prime[:]
            del y_prime[:]
        return(tau)