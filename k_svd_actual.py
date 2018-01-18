'''
#Test training for multiple y training sets such that K-SVD does not overfit
#one signal only
'''

import sys
import math
import numpy as np
import tables as tb
import pandas as pd
from mp_functions import *
import threading

from utils import file_create
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV

from k_svd_object import ApproximateKSVD    

def l2_norm(v):
    return np.linalg.norm(v) / np.sqrt(len(v))

def print_sizes(Phi,y):
    print np.shape(Phi), np.shape(y)
    sys.exit(0)



#process to learn and update Phi for (assumed) k numbers of vectoral y (each y is y_freq x 1 sized)
def Phi_designer_k_svd(Phi_list, y_list, index=None, maxiter=1000):
    '''
    # y_list is expected to have size of k x y_freq (k rows of y_freq sized horizontal signals)
    '''
    global mp_process
    global m
    global n
    #global k
    Phi = np.zeros((m,n))
    Phi[:,:] = Phi_list[:,:]
    y_cur = None
    #If no index input, then indicates the inputted Phi and y are actual objects already, not a list!
    if index is None: 
        y_cur = np.reshape(y_list, (n,m))
        y_cur = y_cur.T

    #else, then that means inputted Phi and y are lists, and thus need to pick object first.
    else:
        y_cur = np.reshape(y_list[index], (n,m))
        y_cur = y_cur.T

        with print_lock:
            print threading.current_thread().name, ":", np.shape(y_cur), y_cur[0,0], y_cur[m-1,n-1]

        #y_cur = np.reshape(y, (len(y), 1)) #making sure to stay consistent for mp calculations     
    
    '''
    with print_lock:
        print threading.current_thread().name, ":", np.shape(y), np.shape(Phi)
    '''
    x_mp = np.zeros((n,n))
    for i in range(0,maxiter):
        Phi_old = np.zeros(Phi.shape)
        Phi_old[:, :] = Phi[:, :]
        #x_mp = np.zeros((n,n))
        #print_sizes(x_mp[:,0], y_cur[:,0])
        #print Phi

        #for every column of x_mp, calculate the sparse representation of the column
        #(find the representation of x_mp that would give minimum solution for ||y - Phi*x_mp||)
        for j in range(0, n): 
            # find approximation of x signal
            if chosen_mp == "omp-scikit":
                mp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity, tol=tol)
                mp_process.fit(Phi, y_cur[:,j])
                x_mp[:,j] = mp_process.coef_

            else:
            #elif max_iter is None:
                x_mp[:,j], _, _ = mp_process(Phi, y_cur[:,j], ncoef=sparsity, verbose=vbose)
        '''
        else:
            x_mp, _, _ = mp_process(Phi, y, ncoef=sparsity, maxit=max_iter, verbose=vbose)
        '''

        # Update dictionary (Phi)
        

        '''
        with print_lock:
            print threading.current_thread().name, "nonzeros for x_mp", x_mp.nonzero()
        '''
        
        #for every t-th atom in Phi...
        #update the dictionary atoms so that it minimizes errors obtained from compressed x_mp
        for t in range(0, n):
            #pick the t-th column from Phi
            #atom = Phi[:,t]
            
            
            #Choose the COLUMN indexes in the t-th ROW of x_mp that is NON zero!
            #(synonymous with picking signals (which column) of x contributed 
            # directly by the t-th atom of the dictionary)
            
            I = x_mp[t] != 0

            #if there are no contributions made by this atom for x_mp, then continue to the next atom.
            if np.sum(I) == 0:
                continue
            

            '''
            # only columns containing nonzero elements from t-th row of x_mp is used (showing indices using t-th atom in dict), 
            # rest are ignored
            '''
            x_copy = x_mp[:, I]

            #zero the t-th row as it will not be used in actual calculation
            x_copy[t] = 0

            #create a copy of Phi with the value of the t-th atom zeroed (to disable contribution from the t-th atom of Phi)
            copy = np.zeros(Phi.shape)
            copy[:] = Phi[:]
            copy[:,t] = 0

            #calculate error produced from contribution of t-th atom only (thus ignoring the rest of the zero elements in initial x_mp.
            error = y_cur[:,I] - np.dot(copy, x_copy)


            #produce a SVD decomp of the obtained error matrix
            U,s,V = np.linalg.svd(error, full_matrices=True)

            Phi[:, t] = U[:, 0]

            '''
            #update only the picked non-zero elements of x_mp (as previously mentioned) to be updated. 
            #(sizes of s and V should have already matched this indices group as well)
            '''
            x_mp[t, I] = s[0] * V[:,0]

        '''
        previous_norm = l2_norm(Phi_old)
        detected_norm = l2_norm(Phi)

        #print previous_norm, detected_norm  

        norm_diff = previous_norm - detected_norm
        '''
        #if l2_norm(Phi - Phi_old) < tol:
        E = np.linalg.norm(y_cur - np.dot(Phi,x_mp))
        #if abs(norm_diff) < tol:
        if E < tol:
            with print_lock:
                print threading.current_thread().name, ":" , E, "converged"

            break
            #return Phi

        with print_lock:
            print threading.current_thread().name, ": Updated for ->", i, "-th iteration. Current error =", E, l2_norm(Phi-Phi_old)

    if index is None:
        return
    else:
        Phi_list = Phi
        with print_lock:
            print threading.current_thread().name, ": Done for ", index, "-th y."
        return


print_lock = threading.Lock()

if len(sys.argv) == 9:
    chosen_mp = sys.argv[1]
    y_file = sys.argv[2]
    y_col = sys.argv[3]
    #Phi_file = sys.argv[4]
    f_name = sys.argv[4]
    sparsity = int(sys.argv[5])
    training_sets = int(sys.argv[6])
    m = int(sys.argv[7])
    n = int(sys.argv[8])
else:
    #print "Please give arguments with the form of:  [MP_to_use] [y_file] [chosen_column] [Phi_file] [output_file] [#mp_sparsity]  [#training_sets]"
    print "Please give arguments with the form of:  [MP_to_use] [y_file] [chosen_column] [output file] [#mp_sparsity]  [#training_sets] [#signal_sample] [#features(columns of y)]"
    sys.exit(0)

mp_process = None

if chosen_mp == 'omp':
    mp_process = omp
elif chosen_mp == 'bomp':
    mp_process = bomp
elif chosen_mp == 'cosamp':
    mp_process = cosamp
elif chosen_mp == 'bmpnils':
    mp_process = bmpnils
elif chosen_mp == 'omp-scikit':
    mp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
else: 
    print "Invalid MP chosen, please input a valid MP!"
    sys.exit(0)
'''
vbose = input("Verbose? ==> ")

max_iter = input("Max iteration of K_SVD? ==> ")
max_iter = int(max_iter)
'''
'''
#take y from reading the .h5 file 
y_file = tb.open_file("y_large.h5", 'r')
y = y_file.root.data[:]
y_file.close()
'''

#take y from csv file
y_f = pd.read_csv(y_file)
y = y_f[y_col].as_matrix()
print type(y)

y_freq = 25600

y_start = input("Start_index? =>")
y_end = y_start + y_freq * training_sets
y = y[y_start:y_end]

#rescale y to make original k-svd algo work properly again
y = y*1
k = training_sets

#sys.exit(0)

#custom = input("Use custom k-svd or no? [True/False] =>")

'''
Note that this particular testing would use a Phi that was created by using a cosine basis function (DCT) on 
the existing y. This is in attempt to follow the results of recent paper that was read
'''


#take Phi from reading the other .h5 file as well.
#file = tb.open_file("Phi_small.h5", 'r')
'''
file = tb.open_file(Phi_file, 'r')
Phi = file.root.data[:]
file.close()
m, n = np.shape(Phi)
'''
#Phi = np.random.normal(0, 0.5, [m,n])


y_mp = np.reshape(y, (k,y_freq))
g = np.shape(y_mp)[1]
#Phi_designer_k_svd(Phi_test, y_mp,maxiter = max_iter)
ksvd = ApproximateKSVD(n_components = 40, transform_n_nonzero_coefs=sparsity)
y_cur = np.reshape(y_mp, (n,m))
y_cur =  y_cur.T
Phi_test = ksvd.fit(y_cur).components_

Phi = Phi_test


#for super large y signals

'''
print Phi_init
print Phi
'''
#m,n = np.shape(Phi)
#print m, n, len(Phi)
m, n = np.shape(Phi)
'''
for x in Phi:
    print x
    print '######'
'''
#sys.exit(0)
file_create(f_name, Phi, m ,n)
