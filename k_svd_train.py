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

def l2_norm(v):
    return np.linalg.norm(v) / np.sqrt(len(v))

def print_sizes(Phi,y):
    print np.shape(Phi), np.shape(y)
    sys.exit(0)


#process to learn and update Phi for (assumed) vectoral y (y is m x 1 sized)
def Phi_learner(Phi_list, y_list, index=None):
    global mp_process
    global m
    global n
    Phi = np.zeros((m,n))
    #If no index input, then indicates the inputted Phi and y are actual objects already, not a list!
    if index is None: 
        Phi = Phi_list[:]
        y = y_list

    #else, then that means inputted Phi and y are lists, and thus need to pick object first.
    else:
        Phi[:] = Phi_list[index][:]
        y = np.zeros(np.shape(y_list[:, index]))
        y[:] = y_list[:, index]
        y = np.reshape(y, (len(y), )) #making sure to stay consistent for mp calculations
         
        with print_lock:
            print threading.current_thread().name, ":", np.shape(y), y[0], y[len(y)-1]

        y = np.reshape(y, (len(y), 1)) #making sure to stay consistent for mp calculations     
    
    '''
    with print_lock:
        print threading.current_thread().name, ":", np.shape(y), np.shape(Phi)
    '''
    for i in range(0,1000):
        Phi_old = np.zeros(Phi.shape)
        Phi_old[:, :] = Phi[:, :]
        #print Phi

        # find approximation of x signal
        if chosen_mp == "omp-scikit":
            mp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity, tol=tol)
            mp_process.fit(Phi, y)
            x_mp = mp_process.coef_

        else:
        #elif max_iter is None:
            x_mp, _, _ = mp_process(Phi, y, ncoef=sparsity, verbose=vbose)
        '''
        else:
            x_mp, _, _ = mp_process(Phi, y, ncoef=sparsity, maxit=max_iter, verbose=vbose)
        '''

        '''
        # Update dictionary (Phi)
        '''

        #for every column in Phi...
        for k in range(0, n):
            #pick the kth column from Phi
            atom = Phi[:,k]

            #make a copy of Phi, with the atom column ZEROED
            copy = np.zeros(Phi.shape)
            copy[:] = Phi[:]
            copy[:,k] = 0   

            #compute error between y and dot product of Phi's copy and x_mp
            
            y_noatom = np.dot(copy,x_mp)
            y_noatom = np.reshape(y_noatom, (len(y_noatom),1))

            Error = y - y_noatom
            #print_sizes(y, Error)

            '''
            # Restrict error to be contribution from chosen atom. 
            # In this case, error would have size of one column only.
            # Hence apply full SVD decomp on Error vector.

            # Example of SVD result: 
                For a (100x1) y with a Phi size of (100x256), the sizes of: 
                U => (100 x 1), s => (1 x 1), V => (1 x 1)
            '''
            #print "Error:", np.shape(Error)
            U,s,V = np.linalg.svd(Error, full_matrices=True)

            # update kth atom of Phi = 1st column of U
            Phi[:,k] = U[:,0]

            #print_sizes(U, V)

            #update the k-th row of x_mp to be product of first column from V with first element value of s
            x_mp[k] = np.dot(s[0], V[:,0])

        previous_norm = l2_norm(Phi_old)
        detected_norm = l2_norm(Phi)

        #print previous_norm, detected_norm  

        norm_diff = previous_norm - detected_norm

        #if l2_norm(Phi - Phi_old) < tol:
        if abs(norm_diff) < tol:
            with print_lock:
                print threading.current_thread().name, ":" , abs(norm_diff), "converged"

            break
            #return Phi

        with print_lock:
            print threading.current_thread().name, ": Updated for ->", i, "-th iteration. Current norm =", l2_norm(Phi-Phi_old)

    if index is None:
        return
    else:
        Phi_list[index] = Phi
        return


#process to learn and update Phi for (assumed) k numbers of vectoral y (each y is y_freq x 1 sized)
def Phi_designer_custom(Phi_list, y_list, index=None):
    '''
    # y_list is expected to have size of k x y_freq (k rows of y_freq sized horizontal signals)
    '''
    global mp_process
    global m
    global n
    global k
    Phi = np.zeros((m,n))
    y_cur = None
    #If no index input, then indicates the inputted Phi and y are actual objects already, not a list!
    if index is None: 
        Phi = Phi_list[:]
        y_cur = y_list

    #else, then that means inputted Phi and y are lists, and thus need to pick object first.
    else:
        Phi[:] = Phi_list[index][:]
        y_cur = np.zeros(np.shape(y_list[:, index*m : (index+1)*m]))
        y_cur[:] = y_list[:, index * m : (index+1) * m  ]
        #print y_cur
        #print y_list
        y_cur = y_cur.T#making sure to stay consistent for mp calculations
        #print y_cur
        #sys.exit(0)
        #k = np.shape()
        
        with print_lock:
            print threading.current_thread().name, ":", np.shape(y_cur), y_cur[0,0], y_cur[m-1,k-1]

        #y_cur = np.reshape(y, (len(y), 1)) #making sure to stay consistent for mp calculations     
    
    '''
    with print_lock:
        print threading.current_thread().name, ":", np.shape(y), np.shape(Phi)
    '''
    for i in range(0,1000):
        Phi_old = np.zeros(Phi.shape)
        Phi_old[:, :] = Phi[:, :]
        x_mp = np.zeros((n,k))
        #print_sizes(x_mp[:,0], y_cur[:,0])
        #print Phi

        #for every column of x_mp, calculate the sparse representation of the column
        for j in range(0, k): 
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
        for t in range(0, n):
            #pick the t-th column from Phi
            #atom = Phi[:,t]

            #make a copy of Phi, with the atom column ZEROED
            copy = np.zeros(Phi.shape)
            copy[:] = Phi[:]
            copy[:,t] = 0

            #compute error between y and dot product of Phi's copy and x_mp
            #print_sizes(np.dot(copy,x_mp), y_cur)
            
            y_noatom = np.dot(copy,x_mp)
            #y_noatom = np.reshape(y_noatom, (len(y_noatom),1))

            Error = y_cur - y_noatom
            #print_sizes(y, Error)

            # Restrict error to be contribution from chosen atom. 
            # In this case, error would have size of one column only.
            # Hence apply full SVD decomp on Error vector.

            # Example of SVD result: 
            #   For a (100x1) y with a Phi size of (100x256), the sizes of: 
            #   U => (100 x 1), s => (1 x 1), V => (1 x 1)

            #print "Error:", np.shape(Error)
            U,s,V = np.linalg.svd(Error, full_matrices=True)
            #print_sizes(U, V)

            # update kth atom of Phi = 1st column of U
            Phi[:,t] = U[:,0] 

            #update the t-th row of x_mp to be product of first column from V with first element value of s
            x_mp[t] = np.dot(s[0], V[:,0])
            

        previous_norm = l2_norm(Phi_old)
        detected_norm = l2_norm(Phi)

        #print previous_norm, detected_norm  

        norm_diff = previous_norm - detected_norm

        #if l2_norm(Phi - Phi_old) < tol:
        if abs(norm_diff) < tol:
            with print_lock:
                print threading.current_thread().name, ":" , abs(norm_diff), "converged"

            break
            #return Phi

        with print_lock:
            print threading.current_thread().name, ": Updated for ->", i, "-th iteration. Current norm =", abs(norm_diff), l2_norm(Phi-Phi_old)

    if index is None:
        return
    else:
        Phi_list[index] = Phi
        with print_lock:
            print threading.current_thread().name, ": Done for ", index, "-th Phi."
        return



#process to learn and update Phi for (assumed) k numbers of vectoral y (each y is y_freq x 1 sized)
def Phi_designer_k_svd(Phi_list, y_list, index=None):
    '''
    # y_list is expected to have size of k x y_freq (k rows of y_freq sized horizontal signals)
    '''
    global mp_process
    global m
    global n
    global k
    Phi = np.zeros((m,n))
    y_cur = None
    #If no index input, then indicates the inputted Phi and y are actual objects already, not a list!
    if index is None: 
        Phi = Phi_list[:]
        y_cur = y_list

    #else, then that means inputted Phi and y are lists, and thus need to pick object first.
    else:
        Phi[:] = Phi_list[index][:]
        y_cur = np.zeros(np.shape(y_list[:, index*m : (index+1)*m]))
        y_cur[:] = y_list[:, index * m : (index+1) * m  ]
        #print y_cur
        #print y_list
        y_cur = y_cur.T#making sure to stay consistent for mp calculations
        #print y_cur
        #sys.exit(0)
        #k = np.shape()
        
        with print_lock:
            print threading.current_thread().name, ":", np.shape(y_cur), y_cur[0,0], y_cur[m-1,k-1]

        #y_cur = np.reshape(y, (len(y), 1)) #making sure to stay consistent for mp calculations     
    
    '''
    with print_lock:
        print threading.current_thread().name, ":", np.shape(y), np.shape(Phi)
    '''
    x_mp = np.zeros((n,k))
    for i in range(0,50):
        Phi_old = np.zeros(Phi.shape)
        Phi_old[:, :] = Phi[:, :]
        #x_mp = np.zeros((n,k))
        #print_sizes(x_mp[:,0], y_cur[:,0])
        #print Phi

        #for every column of x_mp, calculate the sparse representation of the column
        #(find the representation of x_mp that would give minimum solution for ||y - Phi*x_mp||)
        for j in range(0, k): 
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
        E = np.linalg.norm(y_cur - np.dot(Phi, x_mp))

        #if l2_norm(Phi - Phi_old) < tol:
        #if abs(norm_diff) < tol:
        if E < tol:
            with print_lock:
                #print threading.current_thread().name, ":" , abs(norm_diff), "converged"
                print threading.current_thread().name, "-", E, "Converged..."

            break
            #return Phi

        with print_lock:
            #print threading.current_thread().name, ": Updated for ->", i, "-th iteration. Current norm =", abs(norm_diff), l2_norm(Phi-Phi_old)
            print threading.current_thread().name, " - Updated for ->", i, "-th iteration. Current error is = ", E, l2_norm(Phi-Phi_old)
    if index is None:
        return
    else:
        Phi_list[index] = Phi
        with print_lock:
            print threading.current_thread().name, ": Done for ", index, "-th Phi."
        return


print_lock = threading.Lock()

if len(sys.argv) == 8:
    chosen_mp = sys.argv[1]
    y_file = sys.argv[2]
    y_col = sys.argv[3]
    Phi_file = sys.argv[4]
    f_name = sys.argv[5]
    sparsity = int(sys.argv[6])
    training_sets = int(sys.argv[7])
else:
    print "Please give arguments with the form of:  [MP_to_use] [y_file] [chosen_column] [Phi_file] [output_file] [#mp_sparsity]  [#training_sets]"
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

vbose = input("Verbose? ==> ")

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

#sys.exit(0)

custom = input("Use custom k-svd or no? [True/False] =>")

#take Phi from reading the other .h5 file as well.
#file = tb.open_file("Phi_small.h5", 'r')
file = tb.open_file(Phi_file, 'r')
Phi = file.root.data[:]
file.close()
m, n = np.shape(Phi)
k = training_sets
#print_sizes(Phi, y)

#Phi = np.random.normal(0, 0.5, [m,n])

Phi_all = None

#tol = 1e-10
tol = 10

if np.shape(y)[0] <= 1000 and training_sets == 1 :
    Phi_learner(Phi, y)

else:
    #only done if you want to train the dicitonary set for 1 particular y (though it will overfit it badly)
    if k == 1:

        length_y = np.shape(y)[0] / m #divide y into smaller len(y)/m pieces of sizes m
        #y_mp = np.reshape(y, (m, length_y)) #reshape y into m rows of length_y small signals
        print "y first and y final:", y[0], y[len(y)-1] 
        y = np.reshape(y, (length_y, m)) #reshape y into length_y rows of m small signals
        y_mp = y.T #transpose y first so that the m sinals are put on the side so for more accurate recovery by Phi (y=[mxk])
        g = np.shape(y_mp)[1]
        Phi_all = [Phi] * length_y
        '''
        for g in Phi_all:
            print g
        sys.exit(0)
        '''
        print np.shape(Phi_all)
        threads = [None] * length_y
        print np.shape(y_mp[:,0])

        #sys.exit(0)
        
        '''
        for index in range(0, g, 2):
            threads[index] = threading.Thread(target=Phi_learner, args=(Phi_all, y_mp, index))
            threads[index].daemon = True
            threads[index].start()
            #Phi_learner(Phi_all, y_mp, index)

        for thread in threads:
            thread.join()
        '''
        for index in range(0, g, 2):
            
            threads[index] = threading.Thread(target = Phi_learner, args=(Phi_all, y_mp, index))
            threads[index].daemon = True
            threads[index].start()
            if index+1 <= (g-1):
                threads[index+1] = threading.Thread(target = Phi_learner, args=(Phi_all, y_mp, index+1))
                threads[index+1].daemon = True
                threads[index+1].start()

            threads[index].join()
            if index+1 <= (g-1):
                threads[index+1].join()
        

        Phi = Phi_all
        #sys.exit(0)

    #for training dictionary set for multiple y values (multiple 'training sets')
    else:
        #reshape y into a row of signals instead
        #print y
        #print length_y
        y_mp  = np.reshape(y, (k, y_freq))
        #print y_mp
        '''
        print_sizes (y, y_mp)
        y_mp = np.reshape(y, (1, len(y))) 
        length_y = np.shape(y_mp)[1] / m
        y_mp = np.reshape(y)
        '''
        length_y = np.shape(y_mp)[1] / m
        #print np.shape(y_mp)

        #prepare length_y number of smaller Phi dictionaries to be trained for each 'slices' of y 
        Phi_all = [Phi] * length_y
        
        #for each small Phi subdictionary... 
        #for t in range(0, length_y):
            
        #prepare length_y amount of threads for each small Phi subdictionary
        threads = [None] * length_y

        #for each small Phi subdictionary...
        for index in range(0,length_y, 2):
            
            if custom:
                threads[index] = threading.Thread(target=Phi_designer_custom, args=(Phi_all, y_mp, index))
            else:
                threads[index] = threading.Thread(target=Phi_designer_k_svd, args=(Phi_all, y_mp, index))
            #threads[index] = threading.Thread(target = Phi_designer, args=(Phi_all, y_mp, index))
            threads[index].daemon = True
            threads[index].start()
            if index+1 <= (length_y-1):
                if custom:
                    threads[index+1] = threading.Thread(target=Phi_designer_custom, args=(Phi_all, y_mp, index+1))
                else:
                    threads[index+1] = threading.Thread(target=Phi_designer_k_svd, args=(Phi_all, y_mp, index+1))
                #threads[index+1] = threading.Thread(target = Phi_designer, args=(Phi_all, y_mp, index+1))
                threads[index+1].daemon = True
                threads[index+1].start()

            threads[index].join()
            if index+1 <= (length_y-1):
                threads[index+1].join()
            
            #print "test", index
        #sys.exit(0)
        #Phi_designer(Phi_all, y_mp, t)

        '''
        for thread in threads:
            thread.join()
        '''

        '''
        y_mp = y[k * y_freq: (k+1) * y_freq]
        length_y = np.shape(y_mp)[0] / m #divide y into smaller len(y)/m pieces of sizes m
        #y_mp = np.reshape(y, (m, length_y)) #reshape y into m rows of length_y small signals
        #print "y first and y final:", y_mp[0], y_mp[len(y)-1]
        print "Phi #:", k 
        y_mp = np.reshape(y_mp, (length_y, m)) #reshape y into length_y rows of m small signals
        y_mp = y_mp.T #transpose y first so that the m sinals are put on the side so for more accurate recovery by Phi (y=[mxk])
        g = np.shape(y_mp)[1]
        if Phi_all == None:
            Phi_all = [Phi] * length_y

        # Phi_all = [Phi] * length_y

        print np.shape(Phi_all)
        threads = [None] * length_y
        print np.shape(y_mp[:,0])

        #sys.exit(0)
        
        for index in range(0, g):
            threads[index] = threading.Thread(target=Phi_learner, args=(Phi_all, y_mp, index))
            threads[index].daemon = True
            threads[index].start()
            #Phi_learner(Phi_all, y_mp, index)

        for thread in threads:
            thread.join()
        '''

        Phi = Phi_all
            #sys.exit(0)



#for super large y signals

'''
print Phi_init
print Phi
'''
#m,n = np.shape(Phi)
print m, n, len(Phi)
'''
for x in Phi:
    print x
    print '######'
'''
#sys.exit(0)
file_create(f_name, Phi, m ,n)
