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

if len(sys.argv) == 10:
    chosen_mp = sys.argv[1]
    y_file = sys.argv[2]
    y_col = sys.argv[3]
    #Phi_file = sys.argv[4]
    f_name = sys.argv[4]
    sparsity = int(sys.argv[5])
    training_sets = int(sys.argv[6])
    dict_component = int(sys.argv[7])
    m = int(sys.argv[8])
    k = int(sys.argv[9])
else:
    #print "Please give arguments with the form of:  [MP_to_use] [y_file] [chosen_column] [Phi_file] [output_file] [#mp_sparsity]  [#training_sets]"
    print "Please give arguments with the form of:  [MP_to_use] [y_file] [chosen_column] [output file] [#mp_sparsity]  [#training_sets] [# nonzero element in dictionary] [#signal_sample] [#features(columns of y)]"
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
t = training_sets



y_mp = np.reshape(y, (t,y_freq))
g = np.shape(y_mp)[1]
#Phi_designer_k_svd(Phi_test, y_mp,maxiter = max_iter)
ksvd = ApproximateKSVD(n_components = dict_component, transform_n_nonzero_coefs=sparsity)
y_cur = np.reshape(y_mp, (t * k,m))
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
print m,n
'''
for x in Phi:
    print x
    print '######'
'''
#sys.exit(0)
file_create(f_name, Phi, m ,n)
