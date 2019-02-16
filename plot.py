# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:26:54 2018

@author: leeva
"""

import matplotlib.pyplot as plt

class Plotter:
        def __init__(self,x_in,y_in,name):
            self.x = x_in
            self.y = y_in
            self.outfile = name
        
        def draw(self):
            plt.figure()
            plt.plot(self.x, self.y)
            #plt.errorbar(tau,norm_sig,yerr = yer,fmt = 'x')
            #plt.plot(tau_0, final, 'r',label=('T1 = %.3fms' % decay_const))
            #plt.title(str(titlename))
            #plt.legend(loc='best')
            #plt.xlim(xmin=0)
            #plt.xlabel('Tau time (ns)')
            #plt.ylabel('Normalised Signal')
            plt.savefig(self.outfile)
            print(self.outfile + ' ----done')
            plt.close()
            
class Hist:
    
    def __init__(self,x_in,name,label,x_name,y_name):
        self.x = x_in
        self.outfile = name
        self.label = label
        self.x_name = x_name
        self.y_name = y_name
            
        
    def draw_hist(self):
        plt.figure()
        plt.hist(self.x,edgecolor='black', linewidth=1,bins = 50,lable = self.label)
        plt.title(str(self.name))
        plt.xlabel(self.x_name)
        plt.ylabel(self.y_name)
        plt.legend(loc= 'best')
        plt.show()
        plt.close()