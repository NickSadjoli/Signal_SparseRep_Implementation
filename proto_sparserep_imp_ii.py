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


print_lock = threading.Lock()

def process_result(Phi, slc, x_object, x_index, s_cur):
	#global Phi
	#global slice_size
	#x_object[x_index] , _, _= mp_process(Phi, slc, ncoef=s_cur, verbose=False)
	x_object[:, x_index], _, _ = mp_process(Phi, slc, ncoef=s_cur, verbose=False)
	'''
	with print_lock:
		print threading.current_thread().name + "Done "
	'''
def process_multi(Phi, slc, x_object, x_index, s_cur):
	#global Phi
	#global slice_size
	#x_object[x_index] , _, _= mp_process(Phi, slc, ncoef=s_cur, verbose=False)
	x_object[:, x_index], _, _ = mp_process(Phi[x_index], slc, ncoef=s_cur, verbose=False)
	'''
	with print_lock:
		print threading.current_thread().name + "Done "
	'''


def process_fit(process, Phi, y):
	process.fit(Phi,y)
	x = process.coef_
	return x

def process_y(Phi, y, s_thread):
	t0 = time.time()
	#global step
	#print len(y_input)
	#y_cur = np.reshape(y_input, (m, len(y_input)/m))
	y_cur = np.reshape(y, (n,len(y)/n))
	y_cur = y_cur.T
	gram = Phi.dot(Phi.T)
	product = Phi.dot(y_cur.T)
	x_res = orthogonal_mp_gram(gram, product, n_nonzero_coefs=cur_s)
	#x_res, _, _= mp_process(Phi, y_cur, ncoef=s_thread, verbose=False)
	td = time.time()
	#print "time to complete concatenation: ", td - ti
	return x_res, cur_s, td-t0


#used to automatically place labels on the rectangular bar chart below
def autolabel(rects, ax, zeroth_value):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % (height+zeroth_value),
                ha='center', va='bottom')


if len(sys.argv) == 7:
    col_choice= sys.argv[1]
    start_index = int(sys.argv[2])
    Phi_file = sys.argv[3]
    y_fl = sys.argv[4]
    sparsity_chosen = int(sys.argv[5])
    repeats = int(sys.argv[6])
else:
	print "Invalid, please try again! \n(Args expected: [Column of choice in y] [start index for checking] [Phi's file (\w .h5 subfix)] [y's file(\w .csv subfix)] [# sparsity used by the MP algo] [# repeats/testing to be done])"
	sys.exit(0)



#Phi, m, n, counter = take_data("Phi_result_trained_test_5sps.h5")
Phi, m, n, counter = take_data(Phi_file)
print m,n


#y_file = pd.read_csv("data.csv")
y_file = pd.read_csv(y_fl)

'''
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True)
'''
'''

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
count = 0
step = 25600 
y_data = y_file[col_choice].as_matrix()
x_data = range(0, step)
print np.shape(y_data[count*step:(count+1)*step])


def animate(i):
	global count
	ax1.clear()
	ax1.plot(x_data, y_data[count*step:(count+1)*step])
	#ax1.plot(x_data,y_data[count*step:(count+1)*step])
	count +=1

ani = FuncAnimation(fig, animate, interval=1000)#frames=np.linspace(0, 2*np.pi, 128), init_func=init, blit=True)
plt.show()
sys.exit(0)
'''


y_data = y_file[col_choice].as_matrix()
print type(y_data)
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

	y_cur = y_data[(start_index + i*step):(start_index + (i+1)*step)]
	#\y_cur = y_data[(i+12)*step: (i+12+1)*step]
	y_act = y_cur
	y_cur = y_cur * scale
	
	#y_cur = y
	#print 'slice '+ str(i/step)
	print 'slice' + str(i)

	cur_s  = sparsity_chosen
	t0 = time.time()

	Phi_in = Phi
	#x_test, sparsity, mp_time = process_y(Phi_in,y_cur, cur_s)
	x_test, sparsity, mp_time = process_y(Phi_in, y_cur, cur_s)
	#tf = time.time()
	#y_list.append(y_cur)
	y_list.append(y_act)
	x_list.append(x_test)
	sparsity_list.append(sparsity)
	runtime_list.append(mp_time)
	tf = time.time()
	print "total time: " + str(tf-t0) + ' ' + str(cur_s)
	print "" 

x_testing = x_list[0] #used for looking at the non-zero elements of

y_test_list = [None] * len(y_list)


#print len(y_list), len(x_list)
print "Errors experienced by each y slice: "
for j in range(0, len(y_list)):
	#Phi_tst = None
	Phi_tst = Phi
	'''
	if j*step >=128000:
		Phi_tst = Phi_ii
	else:
		Phi_tst = Phi
	'''
	y_test = None

	if counter > 1:
		for k in range(0,np.shape(x_list[j])[1]):
			slc = np.dot(Phi_tst[k], x_list[j][:, k])
			if y_test is None:
				y_test = slc
			else:
				y_test = np.concatenate((y_test, slc))

	else:
		y_test = np.dot(Phi_tst.T, x_list[j])
	
	y_test = np.reshape(y_test, (len(y_list[j]), 1) )
	
	
	
	print "y_test produced (Init):", y_test[0]
	y_test = y_test/scale
	print "Y_test produced (now and stored): ", y_test[0]


	y_ori = y_list[j]
	y_ori = np.reshape(y_ori, (len(y_ori), 1)) #to ensure consistency

	y_test_list[j] = y_test
	#rms = RMS(y_list[j], y_test)
	rms = RMS(y_ori, y_test)
	#rms = ((y_list[j] - y_test) ** 2).mean()
	#r_error = Recovery_Error(y_list[j],y_test)
	r_error = Recovery_Error(y_ori, y_test)
	#print Recovery_Error(y_list[j],y_test) , rms
	print r_error, rms
	#error_list.append(Recovery_Error(y_list[j],y_test))
	error_list.append(r_error)
	rms_list.append(rms)

'''
for i in y_test_list:
	print i
sys.exit(0)
'''	

file_create("x_trained_tester.h5", x_list, np.shape(x_list[0])[0], np.shape(x_list[0])[1])
save_y = None
while save_y not in ['Y', 'y', 'N', 'n']: #pythonic manner, vs:
#while not ((save_y != 'Y') and (save_y != 'y') and (save_y != 'N') and (save_y != 'n')): #non pythonic manner
	save_y = raw_input("Do you want to save the y version of the results as well? [Y/N] ")

if save_y == 'Y' or save_y == 'y':
	y_fname = raw_input("Please state the file name (w/ .h5 prefix. File will be saved in HDF5 format) => ")
	#print np.shape(y_test_list[0])
	file_create(y_fname, y_test_list, np.shape(y_test_list[0])[0], np.shape(y_test_list[0])[1])


'''
y_len = y_list[0][100:300]
y_tst = y_test_list[0][100:300]
print np.shape(y_tst), np.shape(y_len)
jdex = range(100,300)

plt.figure(1)
plt.subplot2grid((1,1), (0,0), colspan=1)
plt.plot(jdex, y_len, 'ro')
trend1 = trendline_fit(jdex, y_len)
plt.plot(jdex, trend1(y_len), 'r--')
plt.plot(jdex, y_tst, 'go')
trend2 = trendline_fit(jdex, y_tst)
plt.plot(jdex, trend2(y_tst), 'g--')
plt.ylabel('original vs tested y')
plt.xlabel('idx')

plt.show()
'''

#print RE, RMS and runtime relative to sparsity values
#original raw version
f, axgrid = plt.subplots(3)

ind = np.arange(len(sparsity_list))
width = 0.5

rect0 = axgrid[0].bar(ind, error_list, color='r')
axgrid[0].set_ylabel('RE values')
axgrid[0].set_title('Recovery_Error of {} vs Sparsity'.format(chosen_mp))
axgrid[0].set_xticks(ind+width/2)
axgrid[0].set_xticklabels(tuple(ind))
#axgrid[0].legend(error_list, ('RE for different sparsity values'))


rect1 = axgrid[1].bar(ind, rms_list, width, color='b')
axgrid[1].set_ylabel('RMS_Values')
axgrid[1].set_title('RMS for {} vs Sparsity'.format(chosen_mp))
axgrid[1].set_xticks(ind+width/2)
axgrid[1].set_xticklabels(tuple(ind)) #cannot directly take an array
#axgrid[1].legend(rms_list, ('RMS values for each sparsity_values')) 	

rect2 = axgrid[2].bar(ind, runtime_list, width, color='b')
axgrid[2].set_ylabel('Runtimes (s)')
axgrid[2].set_title('Runtimes for {} vs Sparsity'.format(chosen_mp))
axgrid[2].set_xticks(ind+width/2)
axgrid[2].set_xticklabels(tuple(ind)) #cannot directly take an array
#axgrid[2].legend(runtime_list, ('Runtime values'))


autolabel(rect0, axgrid[0],0)
autolabel(rect1, axgrid[1],0)
autolabel(rect2, axgrid[2],0)
plt.show()
###############

#relative difference version

f,axgrid = plt.subplots(3)
ind = np.arange(len(sparsity_list))
width = 0.5
#err_relative = np.zeros(np.shape(error_list))
err_relative = np.array(error_list)
err_relative *= 1000
err_relative = np.around(err_relative, decimals=2)
err_rel_val = err_relative[0]
err_relative -= err_rel_val

y_tick_1 = np.arange(err_relative.min(), err_relative.max(), 5)
y_tick_1_actual = y_tick_1 + err_rel_val

#print y_tick_1, y_tick_1_actual
rect0 = axgrid[0].bar(ind, err_relative, width, color='r')
axgrid[0].set_ylabel('RE values (*1000)')
#y_ticks = axgrid[0].get_yticks()
#y_ticks_actual = y_ticks + err_rel_val
#axgrid[0].set_yticklabels(y_ticks_actual)

axgrid[0].set_yticks(y_tick_1)
axgrid[0].set_yticklabels(tuple(y_tick_1_actual))

axgrid[0].set_title('Recovery_Error of {} vs Sparsity (relative to training set)'.format(chosen_mp))
axgrid[0].set_xticks(ind+width/2)
axgrid[0].set_xticklabels(tuple(ind))
#axgrid[0].legend(error_list, ('RE for different sparsity values'))

rms_relative = np.array(rms_list)
rms_relative *= 1000
rms_relative = np.around(rms_relative, decimals=2)
rms_rel_val = rms_relative[0]
rms_relative -= rms_rel_val
y_tick_2 = np.arange(rms_relative.min(), rms_relative.max(), 0.05)
y_tick_2_actual = y_tick_2 + rms_rel_val
#print y_tick_1, rms_relative
rect1 = axgrid[1].bar(ind, rms_relative, width, color='b')
axgrid[1].set_ylabel('RMS_Values (*1000)')
#y_ticks = axgrid[1].get_yticks()
#y_ticks_actual = y_ticks + err_rel_val
#axgrid[1].set_yticklabels(y_ticks_actual)

axgrid[1].set_yticks(y_tick_2)
axgrid[1].set_yticklabels(tuple(y_tick_2_actual))

axgrid[1].set_title('RMS for {} vs Sparsity (relative to training set)'.format(chosen_mp))
axgrid[1].set_xticks(ind+width/2)
axgrid[1].set_xticklabels(tuple(ind)) #cannot directly take an array
#axgrid[1].legend(rms_list, ('RMS values for each sparsity_values')) 	

runtime_relative = np.array(runtime_list)
runtime_relative *= 10
runtime_relative = np.around(runtime_relative, decimals=2)
runtime_rel_val = runtime_relative[0]
runtime_relative -= runtime_rel_val
y_tick_3 = np.arange(runtime_relative.min(), runtime_relative.max(), 0.05)
y_tick_3_actual = y_tick_3 + runtime_rel_val
rect2 = axgrid[2].bar(ind, runtime_relative, width, color='b')
axgrid[2].set_ylabel('Runtimes (*10 s)')
#y_ticks = axgrid[2].get_yticks()
#y_ticks_actual = y_ticks + err_rel_val
#axgrid[2].set_yticklabels(y_ticks_actual)

axgrid[2].set_yticks(y_tick_3)
axgrid[2].set_yticklabels(tuple(y_tick_3_actual))

axgrid[2].set_title('Runtimes for {} vs Sparsity (relative to training set)'.format(chosen_mp))
axgrid[2].set_xticks(ind+width/2)
axgrid[2].set_xticklabels(tuple(ind)) #cannot directly take an array
#axgrid[2].legend(runtime_list, ('Runtime values'))

autolabel(rect0, axgrid[0], err_rel_val)
autolabel(rect1, axgrid[1], rms_rel_val)
autolabel(rect2, axgrid[2], runtime_rel_val)

plt.show()
######################

#percentage version

f,axgrid = plt.subplots(3)
ind = np.arange(len(sparsity_list))
width = 0.5
#err_relative = np.zeros(np.shape(error_list))
err_relative = np.array(error_list)
err_relative = err_relative / err_relative[0]
err_relative = np.around(err_relative, decimals=2)
err_rel_val = err_relative[0]
err_relative -= err_rel_val

y_tick_1 = np.arange(err_relative.min(), err_relative.max(), 0.03)
y_tick_1_actual = y_tick_1 + err_rel_val# err_rel_val


#print y_tick_1, y_tick_1_actual
rect0 = axgrid[0].bar(ind, err_relative, width, color='r')
axgrid[0].set_ylabel('RE values (% vs training)')
#y_ticks = axgrid[0].get_yticks()
#y_ticks_actual = y_ticks + err_rel_val
#axgrid[0].set_yticklabels(y_ticks_actual)

axgrid[0].set_yticks(y_tick_1)
axgrid[0].set_yticklabels(tuple(y_tick_1_actual))


axgrid[0].set_title('Recovery_Error of {} vs Sparsity (relative to training set)'.format(chosen_mp))
axgrid[0].set_xticks(ind+width/2)
axgrid[0].set_xticklabels(tuple(ind))
#axgrid[0].legend(error_list, ('RE for different sparsity values'))

rms_relative = np.array(rms_list)
rms_relative *= 1000
rms_relative = np.around(rms_relative, decimals=2)
rms_rel_val = rms_relative[0]
rms_relative -= rms_rel_val
y_tick_2 = np.arange(rms_relative.min(), rms_relative.max(), 0.05)
y_tick_2_actual = y_tick_2 + rms_rel_val
#print y_tick_1, rms_relative
rect1 = axgrid[1].bar(ind, rms_relative, width, color='b')
axgrid[1].set_ylabel('RMS_Values (*1000)')
#y_ticks = axgrid[1].get_yticks()
#y_ticks_actual = y_ticks + err_rel_val
#axgrid[1].set_yticklabels(y_ticks_actual)

axgrid[1].set_yticks(y_tick_2)
axgrid[1].set_yticklabels(tuple(y_tick_2_actual))

axgrid[1].set_title('RMS for {} vs Sparsity (relative to training set)'.format(chosen_mp))
axgrid[1].set_xticks(ind+width/2)
axgrid[1].set_xticklabels(tuple(ind)) #cannot directly take an array
#axgrid[1].legend(rms_list, ('RMS values for each sparsity_values')) 	

runtime_relative = np.array(runtime_list)
runtime_relative *= 10
runtime_relative = np.around(runtime_relative, decimals=2)
runtime_rel_val = runtime_relative[0]
runtime_relative -= runtime_rel_val
y_tick_3 = np.arange(runtime_relative.min(), runtime_relative.max(), 0.05)
y_tick_3_actual = y_tick_3 + runtime_rel_val
rect2 = axgrid[2].bar(ind, runtime_relative, width, color='b')
axgrid[2].set_ylabel('Runtimes (*10 s)')
#y_ticks = axgrid[2].get_yticks()
#y_ticks_actual = y_ticks + err_rel_val
#axgrid[2].set_yticklabels(y_ticks_actual)

axgrid[2].set_yticks(y_tick_3)
axgrid[2].set_yticklabels(tuple(y_tick_3_actual))

axgrid[2].set_title('Runtimes for {} vs Sparsity (relative to training set)'.format(chosen_mp))
axgrid[2].set_xticks(ind+width/2)
axgrid[2].set_xticklabels(tuple(ind)) #cannot directly take an array
#axgrid[2].legend(runtime_list, ('Runtime values'))

autolabel(rect0, axgrid[0], err_rel_val)
autolabel(rect1, axgrid[1], rms_rel_val)
autolabel(rect2, axgrid[2], runtime_rel_val)

plt.show()

sys.exit(0)









