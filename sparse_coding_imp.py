import sys
import numpy as np
import tables as tb
import pandas as pd
import threading
import csv
#from queue import Queue
import time

from sklearn.linear_model import OrthogonalMatchingPursuit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mp_functions import *
from utils import *
from sklearn.linear_model import orthogonal_mp_gram

from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column

def process_y(Phi, y, s_thread):
	t0 = time.time()
	#global step
	#print len(y_input)
	#y_cur = np.reshape(y_input, (m, len(y_input)/m))
	y_cur = np.reshape(y, (n,len(y)/n))
	y_cur = y_cur.T
	gram = Phi.dot(Phi.T)
	product = Phi.dot(y_cur.T)
	x_res = orthogonal_mp_gram(gram, product, n_nonzero_coefs=s_thread)
	#x_res, _, _= mp_process(Phi, y_cur, ncoef=s_thread, verbose=False)
	td = time.time()
	#print "time to complete concatenation: ", td - ti
	return x_res, s_thread, td-t0


#used to automatically place labels on the rectangular bar chart below
def autolabel(rects, ax, zeroth_value):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.3f' % (height+zeroth_value),
                ha='center', va='bottom')


if len(sys.argv) == 8:
    folder = sys.argv[1]
    start_index = int(sys.argv[2])
    Phi_file = sys.argv[3]
    y_fl = sys.argv[4]
    sparsity_chosen = int(sys.argv[5])
    repeats = int(sys.argv[6])
    output_file = sys.argv[7]
else:
	print "Invalid, please try again! \n(Args expected: [Folder of choice] [start index for checking] [Phi's file (\w .h5 subfix)] [y's file(\w .csv subfix)] [# sparsity used by the MP algo] [# repeats/testing to be done] [output_file (\w .h5 prefix)])"
	sys.exit(0)


TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
#Phi, m, n, counter = take_data("Phi_result_trained_test_5sps.h5")
Phi, m, n, counter = take_data(Phi_file)
print m,n

y_data, _, _ , _= take_data(y_fl)
print np.shape(y_data), type(y_data)
'''
#y_file = pd.read_csv("data.csv")
y_file = pd.read_csv(y_fl)

y_data = y_file[col_choice].as_matrix()
print type(y_data)
'''
step = 25600
#print_sizes(Phi, y_data)

y_list = [] 
x_list = []
sparsity_list = []
runtime_list = []
error_list = []
rms_list = []
tprev = 0
scale = 1

chosen_mp = 'omp'

mp_process = omp

#print "testing time: ", time.time()

#NOTE if 1e3 line is active, x created is for 10 * y values, not the actual original y itself!
print "time elapsed for each y_slice: "
for i in range(0, repeats):
	t0 = time.time()
	y_cur = y_data[(start_index + i*step):(start_index + (i+1)*step)]

	

	cur_s  = sparsity_chosen
	t0 = time.time()

	Phi_in = Phi
	#x_test, sparsity, mp_time = process_y(Phi_in,y_cur, cur_s)
	x_test, sparsity, mp_time = process_y(Phi_in, y_cur, cur_s)
	#np.savetxt("sparse_code/spscode_"+str(i)+".csv", x_test, fmt='%10.9f',delimiter=",")
	
	file_create(folder+"/sps_code"+str(i)+".h5", x_test, m, len(y_cur)/n)
	tf = time.time()

	'''
	y_list.append(y_cur)
	x_list.append(x_test)
	sparsity_list.append(np.count_nonzero(x_test))
	
	
	print "total time: " + str(tf-t0) + ' ' + str(cur_s), np.shape(x_test), np.count_nonzero(x_test), sys.getsizeof(x_test)
	print "" 
	'''
	runtime_list.append(tf-t0)
	print 'slice' + str(i), "| total time: ",  str(tf-t0) + ' ' + str(cur_s), np.shape(x_test), "| sparsity:",  np.count_nonzero(x_test)
runtime_list = np.reshape(runtime_list, (len(runtime_list), 1))
file_create(folder+"/sps_code_runtime.h5", runtime_list, np.shape(runtime_list)[0], np.shape(runtime_list)[1])

sys.exit(0)








