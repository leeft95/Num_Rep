# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 01:17:23 2018

@author: Leevan T
"""
import os
from sys import path
path.append(os.getcwd() + "\\classes")
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from iminuit import Minuit
import time

start_time = time.time()
'''
Defining inital parameters as given by the report discription 

P1 = pdf of first decay lifetime
P2 = pdf of second decay lifetime

F = fraction of decay lifetime
t1 = lifetime of P1
t2 = lifetime of P2

NOTE: limits of observables are t:[0,10] and theta:[0,2pi]

'''

############
#__Part 1__#
############

'''
Determine the normalisation function of the pdf P1 and P2

Generate 10000 random events with a distribution of t and theta given by the 
pdfs using the box method

'''

def normFunc(t1,t2):
    N1 = quad(lambda t: 3*np.pi*(np.exp(-t/t1)), 0, 10)[0]
    N2 = quad(lambda t: 3*np.pi*(np.exp(-t/t2)), 0, 10)[0]
    print(N1,N2)
    return N1,N2

class boxRand():
    
    def __init__(self, F):
        self.F = F
        
    def RandGen(self,nRand, a, thetaUp, tUp,N1,N2,t1,t2):

        tValues = []
        thetaValues = []

        while len(tValues) < nRand and len(thetaValues) < nRand:

            thetaRand = np.random.uniform()
            tRand = np.random.uniform()
            y = np.random.uniform()
                    
            theta = a + (thetaUp - a) * thetaRand
            t = a + (tUp - a) * tRand

            pdf1 = (1/N1)*(1+(np.cos(theta))**2)*(np.exp(-t/t1))
            pdf2 = (1/N2)*(3*(np.sin(theta))**2)*(np.exp(-t/t2))
            pdf = (self.F)*pdf1 + (1-self.F)*pdf2


            if y <= pdf:
                tValues.append(t)
                thetaValues.append(theta)
                
        
        return tValues,thetaValues
    
############
#__Part 2__#
############    
'''
The Negative log likelyhood class that determines works with the minuit 
minimization function

tOnlypdf works for the pdf that keeps the theta constant
nllTF returns the NLL for the minuit minimization function

ttpdf works for the total pdf using the full data set
nllT1F returns the NLL for the minuit minimization function
'''        
class NLL():
    
    def __init__(self,x,theta):
        self.t = x
        self.theta = theta
    
    def tOnlypdf(self,t,F,t1,t2):
        
        N1 = quad(lambda t: (np.exp(-t/t1)), 0, 10)[0]
        N2 = quad(lambda t: (np.exp(-t/t2)), 0, 10)[0]
        
        
        pdf1 = (1/N1)*(np.exp(-t/t1))
        pdf2 = (1/N2)*(np.exp(-t/t2))
        pdf = (F)*pdf1 + (1-F)*pdf2
        return pdf
    
    def nllTF(self,F,t1,t2):
        
            pdf = self.tOnlypdf(self.t,F,t1,t2) 
            pdf = np.clip(pdf, 1.0e-06, np.inf)
            return -np.sum(np.log(pdf))
        

  
############
#__Part 3__#
############   
    
    def ttpdf(self,t,theta,F,t1,t2):
        
        N1 = quad(lambda t: 3*np.pi*(np.exp(-t/t1)), 0, 10)[0]
        N2 = quad(lambda t: 3*np.pi*(np.exp(-t/t2)), 0, 10)[0]
        
        pdf1 = (1/N1)*(1+(np.cos(theta))**2)*(np.exp(-t/t1))
        pdf2 = (1/N2)*(3*(np.sin(theta))**2)*(np.exp(-t/t2))
        pdf = (F)*pdf1 + (1-F)*pdf2
        return pdf
    
    def nllT1F(self,F,t1,t2):
        
            pdf = self.ttpdf(self.t,self.theta,F,t1,t2) 
            pdf = np.clip(pdf, 1.0e-06, np.inf)
            return -np.sum(np.log(pdf))
        

        
############
#__Part 4__#
############            
'''
Determine the proper errors on the parameters

Achive this by varying the parameters and re minising the others until the 
point the NLL rises by by 0.5

Below is function for data set with 2D
''' 

def errorPropF2(F,T1,T2,dx,data,frnt,param):
   
    er_fpp = Minuit(data.nllT1F, F = F, t1 = T1, t2 = T2, 
                    errordef = 0.5, pedantic = False,
                    print_level = 0,fix_F = True)

        
    fminF, paramF = er_fpp.migrad()
    
    mini_1 = er_fpp.fval
    #forward and backwards error determination loops
    #break conditoin when NLL > 0.5
    if frnt == 1:
        while True:
            '''   
        if statements to switch between each parameter so as to reminimise the 
        correct parameters while keeping the right one fixed
            '''
            if param == 1:
                f = F + dx
                er_fp = Minuit(data.nllT1F, F = f, t1 = T1, t2 = T2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_F = True)
            if param == 2:
                t1 = T1 + dx
                er_fp = Minuit(data.nllT1F, F = F, t1 = t1, t2 = T2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_t1 = True)
            if param == 3:
                t2 = T2 + dx
                er_fp = Minuit(data.nllT1F, F = F, t1 = T1, t2 = t2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_t2 = True)
            
            fminF, paramF = er_fp.migrad()
        
            mini_2 = er_fp.fval
            #break if statement
            if (mini_2 - mini_1) >= 0.4999999:
                #print('here')
                if param == 1:
                    return f
                if param == 2:
                    return t1
                if param == 3:
                    return t2
            else:
                if param == 1:
                    F = f
                if param == 2:
                    T1 = t1
                if param == 3:
                    T2 = t2
    if frnt == 0:
        while True:
                    
            if param == 1:
                f = F - dx
                er_fp = Minuit(data.nllT1F, F = f, t1 = T1, t2 = T2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_F = True)
            if param == 2:
                t1 = T1 - dx
                er_fp = Minuit(data.nllT1F, F = F, t1 = t1, t2 = T2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_t1 = True)
            if param == 3:
                t2 = T2 - dx                
                er_fp = Minuit(data.nllT1F, F = F, t1 = T1, t2 = t2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_t2 = True)
            
            fminF, paramF = er_fp.migrad()
        
            mini_2 = er_fp.fval
            if (mini_2 - mini_1) >= 0.4999999:
                if param == 1:
                    return f
                if param == 2:
                    return t1
                if param == 3:
                    return t2
            else:
                if param == 1:
                    F = f
                if param == 2:
                    T1 = t1
                if param == 3:
                    T2 = t2
'''
Same function as above but using only the t dependence of the function
'''        
def errorPropF1(F,T1,T2,dx,data,frnt,param):
     er_fpp = Minuit(data.nllTF, F = F, t1 = T1, t2 = T2, 
                     errordef = 0.5, pedantic = False,
                     print_level = 0,fix_F = True)
     
     fminF, paramF = er_fpp.migrad()
     
     mini_1 = er_fpp.fval
     
     if frnt == 1:
        while True:
            
            if param == 1:
                f = F + dx
                er_fp = Minuit(data.nllTF, F = f, t1 = T1, t2 = T2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_F = True)
            if param == 2:
                t1 = T1 + dx
                er_fp = Minuit(data.nllTF, F = F, t1 = t1, t2 = T2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_t1 = True)
            if param == 3:
                t2 = T2 + dx
                er_fp = Minuit(data.nllTF, F = F, t1 = T1, t2 = t2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_t2 = True)
            
            fminF, paramF = er_fp.migrad()
            
            mini_2 = er_fp.fval
            if (mini_2 - mini_1) >= 0.499999:
                if param == 1:
                    return f
                if param == 2:
                    return t1
                if param == 3:
                    return t2
            else:
                if param == 1:
                    F = f
                if param == 2:
                    T1 = t1
                if param == 3:
                    T2 = t2
     if frnt == 0:
        while True:
            
        
            if param == 1:
                f = F - dx
                er_fp = Minuit(data.nllTF, F = f, t1 = T1, t2 = T2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_F = True)
            if param == 2:
                t1 = T1 - dx
                er_fp = Minuit(data.nllTF, F = F, t1 = t1, t2 = T2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_F = True)
            if param == 3:
                t2 = T2 - dx                
                er_fp = Minuit(data.nllTF, F = F, t1 = T1, t2 = t2, 
                               errordef = 0.5, pedantic = False,
                               print_level = 0,fix_F = True)
            
            fminF, paramF = er_fp.migrad()
            
            mini_2 = er_fp.fval
            if (mini_2 - mini_1) >= 0.499999:
                if param == 1:
                    return f
                if param == 2:
                    return t1
                if param == 3:
                    return t2
            else:
                if param == 1:
                    F = f
                if param == 2:
                    T1 = t1
                if param == 3:
                    T2 = t2
    
'''
The main function that is used to complete the requried tasks of the report
'''

def main():
    
############
#__Part 1__#
############    
    '''
Initilising the task parameters and calling the functions required to complete
part 1
    '''
    t1 = 1.0
    t2 = 2.0
    thetaUp = 2.0*np.pi
    tUp = 10
    nRand = 10000
    lowLim = 0.0
    dx = 0.001
    
    N1,N2 = normFunc(t1,t2)
    '''
    creating a range of F values to analyse the distribution and then calling
    relavant methods to plot the required results
    '''
    F = [0.5, 0.0, 1.0]
    for i in F:
        box = boxRand(i)
        tVals,thetaVals = box.RandGen(nRand, lowLim, thetaUp,
                                              tUp,N1,N2,t1,t2)
        
        plt.figure()
        plt.hist(tVals,edgecolor='black', linewidth=1,bins = 50,
                 label = 'F = ' + str(i))
        plt.title('t values distribution of pdf')
        plt.xlabel('Lifetime')
        plt.ylabel('Count')
        plt.legend(loc= 'best')
        plt.savefig('Part 1a ' + str(i) + 'F.png')
        plt.show()
        
        plt.figure()
        plt.hist(thetaVals,edgecolor='black', linewidth=1,bins = 50,
                 label = 'F = ' + str(i))
        plt.title('Theta values distribution of pdf')
        plt.xlabel('Theta')
        plt.ylabel('Count')
        plt.legend(loc= 'best')
        plt.savefig('Part 1b ' + str(i) + 'F.png' )
        plt.show()
        
############
#__Part 2__#
############     
        
    '''
Importing the relevant data set using a numpy function and then creating 
the NNL object used for minimzation
    '''
    filename = str('datafile-Xdecay.txt')
    
    data = np.loadtxt(filename,delimiter = ' ')
    t = data[:,0]
    theta = data[:,1]
    nnlData = NLL(t,theta)

    '''
Creating the Minuit object and running the minimization using the migrad 
function. Then determining the errors of the parameters using the simple method
which varys only one parameter while keeping the others fixed
    '''
    print('Parameter estimation using only t data')
    mini1 = Minuit(nnlData.nllTF, F = 0.5, t1 = 1, t2 = 2, 
                 errordef = 0.5, pedantic = False)
    
    '''
Results of minuit without fixing any parameters
    '''
    fminMini1, paramMini1 = mini1.migrad()
    
    print('Value (F) = '+ str(mini1.values['F']) +'\n')
    print('Error (F)  = '+ str(mini1.errors['F']) +'\n')
    print('Value (t1) = '+ str(mini1.values['t1']) +'\n')
    print('Error (t1) = '+ str(mini1.errors['t1']) +'\n')
    print('Value (t2) = '+ str(mini1.values['t2']) +'\n')
    print('Error (t2) = '+ str(mini1.errors['t2']) +'\n')
    
    print('correlation matrix ' +str(mini1.matrix(correlation = True)))
    
    print('-------------------------------------\n\n')
    print('Error determination \'F\' (Simple)')

    
    er_f = Minuit(nnlData.nllTF, F = mini1.values['F'], t1 = mini1.values['t1']
    , t2 = mini1.values['t2'],errordef = 0.5, fix_t1 = True, fix_t2 = True,
    pedantic = False)
    

    fminF, paramF = er_f.migrad()
    print('Value (F) = '+ str(er_f.values['F']) +'\n')
    print('Error (F)  = '+ str(er_f.errors['F']) +'\n')
    print('-------------------------------------\n\n')
    print('Error determination \'t1\' (Simple)')
    er_t1 = Minuit(nnlData.nllTF, F = mini1.values['F'], t1 =mini1.values['t1']
    , t2 = mini1.values['t2'], errordef = 0.5, fix_t2 = True, fix_F = True,
    pedantic = False)
    

    fminT1, paramT1 = er_t1.migrad()

    print('Value (t1) = '+ str(er_t1.values['t1']) +'\n')
    print('Error (t1) = '+ str(er_t1.errors['t1']) +'\n')
    print('-------------------------------------\n\n')
    print('Error determination \'t2\' (Simple)')

    er_t2 = Minuit(nnlData.nllTF, F = mini1.values['F'], t1 =mini1.values['t1']
                   ,t2 = mini1.values['t2'], errordef = 0.5, fix_t1 = True,
                   fix_F = True, pedantic = False)
    

    fminT2, paramT2 = er_t2.migrad()
    print('Value (t2) = '+ str(er_t2.values['t2']) +'\n')
    print('Error (t2) = '+ str(er_t2.errors['t2']) +'\n')        
    print('===================================\n\n')
############
#__Part 3__#
############   
    
    '''
Creating the Minuit object and running the minimization using the migrad 
function for the full data set. Then determining the errors of the parameters 
using the simple method which varys only one parameter while 
keeping the others fixed
    '''
    
    print('Parameter estimation using full data set')
    mini2 = Minuit(nnlData.nllT1F, F = 0.5, t1 = 1, t2 = 2, 
                 errordef = 0.5, pedantic = False)
    

    fminMini2, paramMini2 = mini2.migrad()
    print('Value (F) = '+ str(mini2.values['F']) +'\n')
    print('Error (F)  = '+ str(mini2.errors['F']) +'\n')
    print('Value (t1) = '+ str(mini2.values['t1']) +'\n')
    print('Error (t1) = '+ str(mini2.errors['t1']) +'\n')
    print('Value (t2) = '+ str(mini2.values['t2']) +'\n')
    print('Error (t2) = '+ str(mini2.errors['t2']) +'\n')
    
    print('correlation matrix ' +str(mini2.matrix(correlation = True)))
            
    print('-------------------------------------\n\n')
    
    print('Error determination \'F\' (Simple)')

    
    er_f1 = Minuit(nnlData.nllT1F, F = mini2.values['F'],t1 =mini2.values['t1']
    , t2 = mini2.values['t2'], errordef = 0.5, fix_t1 = True, fix_t2 = True,
    pedantic = False)
    

    fminF1, paramF1 = er_f1.migrad()
    print('Value (F) = '+ str(er_f1.values['F']) +'\n')
    print('Error (F)  = '+ str(er_f1.errors['F']) +'\n')
    print('-------------------------------------\n\n')
    print('Error determination \'t1\' (Simple)')
    er_tt1 = Minuit(nnlData.nllT1F,F = mini2.values['F'],t1 =mini2.values['t1']
    , t2 = mini2.values['t2'], errordef = 0.5, fix_t2 = True, fix_F = True, 
    pedantic = False)
    

    fminTT1, paramTT1 = er_tt1.migrad()

    print('Value (t1) = '+ str(er_tt1.values['t1']) +'\n')
    print('Error (t1) = '+ str(er_tt1.errors['t1']) +'\n')
    print('-------------------------------------\n\n')
    print('Error determination \'t2\' (Simple)')

    er_tt2 = Minuit(nnlData.nllT1F,F = mini2.values['F'],t1=mini2.values['t1'],
                    t2 = mini2.values['t2'],errordef = 0.5, fix_t1 = True,
                    fix_F = True, pedantic = False)
    

    fminTT2, paramTT2 = er_tt2.migrad()
    print('Value (t2) = '+ str(er_tt2.values['t2']) +'\n')
    print('Error (t2) = '+ str(er_tt2.errors['t2']) +'\n')        
    print('===================================\n\n')
    
    
############
#__Part 4__#
############
    '''
    Using the functions of errorPropF1/F2 the proper errors are determined for
    each of the parameters
    '''
    #F parameter
    errFfr = errorPropF1(mini1.values['F'],mini1.values['t1'],
                         mini1.values['t2'],dx,nnlData,1,1)
    errFbck = errorPropF1(mini1.values['F'],mini1.values['t1'],
                          mini1.values['t2'],dx,nnlData,0,1)
    errF1fr = errorPropF2(mini2.values['F'],mini2.values['t1'],
                          mini2.values['t2'],dx,nnlData,1,1)
    errF1bck = errorPropF2(mini2.values['F'],mini2.values['t1'],
                           mini2.values['t2'],dx,nnlData,0,1)
    print('final value of F(1D) and its proper errors\n')
    print('Value (F) = '+ str(mini1.values['F']) +'\n')
    print('Error (F)  = +'+ str(errFfr - mini1.values['F']) +'/ -' +
          str(mini1.values['F'] - errFbck) + '\n\n')
    print('final value of F(2D) and its proper errors\n')
    print('Value (F) = '+ str(mini2.values['F']) +'\n')
    print('Error (F)  = +'+ str(errF1fr - mini2.values['F']) +'/ -' +
          str(mini2.values['F'] - errF1bck) + '\n\n')
    ########################
    #t1 parameter
    errFfr = errorPropF1(mini1.values['F'],mini1.values['t1'],
                         mini1.values['t2'],dx,nnlData,1,2)
    errFbck = errorPropF1(mini1.values['F'],mini1.values['t1'],
                          mini1.values['t2'],dx,nnlData,0,2)
    errF1fr = errorPropF2(mini2.values['F'],mini2.values['t1'],
                          mini2.values['t2'],dx,nnlData,1,2)
    errF1bck = errorPropF2(mini2.values['F'],mini2.values['t1'],
                           mini2.values['t2'],dx,nnlData,0,2)
    print('final value of t1(1D) and its proper errors\n')
    print('Value (t1) = '+ str(mini1.values['t1']) +'\n')
    print('Error (t1)  = +'+ str(errFfr - mini1.values['t1']) +'/ -' +
          str(mini1.values['t1'] - errFbck) + '\n\n')
    print('final value of t1(2D) and its proper errors\n')
    print('Value (t1) = '+ str(mini2.values['t1']) +'\n')
    print('Error (t1)  = +'+ str(errF1fr - mini2.values['t1']) +'/ -' +
          str(mini2.values['t1'] - errF1bck) + '\n\n')
    ########################
    #t2 parameter
    errFfr = errorPropF1(mini1.values['F'],mini1.values['t1'],
                         mini1.values['t2'],dx,nnlData,1,3)
    errFbck = errorPropF1(mini1.values['F'],mini1.values['t1'],
                          mini1.values['t2'],dx,nnlData,0,3)
    errF1fr = errorPropF2(mini2.values['F'],mini2.values['t1'],
                          mini2.values['t2'],dx,nnlData,1,3)
    errF1bck = errorPropF2(mini2.values['F'],mini2.values['t1'],
                           mini2.values['t2'],dx,nnlData,0,3)
    print('final value of t2(1D) and its proper errors\n')
    print('Value (t2) = '+ str(mini1.values['t2']) +'\n')
    print('Error (t2)  = +'+ str(errFfr - mini1.values['t2']) +'/ -' +
          str(mini1.values['t2'] - errFbck) + '\n\n')
    print('final value of t2(2D) and its proper errors\n')
    print('Value (t2) = '+ str(mini2.values['t2']) +'\n')
    print('Error (t2)  = +'+ str(errF1fr - mini2.values['t2']) +'/ -' +
          str(mini2.values['t2'] - errF1bck) + '\n\n')

if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))

