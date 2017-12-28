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


def process_y(Phi, y_input, s_thread):
	t0 = time.time()
	#global step
	#print len(y_input)
	#y_cur = np.reshape(y_input, (m, len(y_input)/m))
	y_cur = np.reshape(y_input, (len(y_input)/m, m))
	y_cur = y_cur.T
	k = np.shape(y_cur)[1]

	#do sanity check for amount of Phi available for use for all slices of y in threads later
	if k != counter:
		print "Amount of available Phi doesn't match the amount of y slices formed! Please ensure correct Phi was selected!"
		sys.exit(0)

	#print_sizes(y_cur, y_cur)
	#k = np.shape(y_cur)[0]
	#global slice_size
	threads = [None] * (k)
	#x_test = [None] * (k)
	x_test = np.zeros((n,k))
	
	#print_sizes(x_test[:, 0],y_cur)
	sparsity_tot = s_thread * (k)

	for i in range(len(threads)):
		#threads[i]= threading.Thread(target=process_result, args=(y_input[i*slice_size : (i+1)*slice_size], x_test, i, s_thread) ) 
		#threads[i] = threading.Thread(target=process_result, args=(y_cur[i], x_test, i, s_thread) )
		if counter > 1: #i.e. more than one Phi available
			threads[i] = threading.Thread(target=process_multi, args=(Phi, y_cur[:, i], x_test, i, s_thread))
		else:
			threads[i] = threading.Thread(target=process_result, args=(Phi, y_cur[:, i], x_test, i, s_thread) )
		threads[i].daemon = True
		threads[i].start()
		#threads.append(tr)

	for j in range(len(threads)):
		threads[j].join()

	x_res = x_test
	td = time.time()
	#print "time to complete concatenation: ", td - ti
	return x_res, sparsity_tot, td-t0

#used to automatically place labels on the rectangular bar chart below
def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % height,
                ha='center', va='bottom')


if len(sys.argv) == 3:
    col_choice= sys.argv[1]
    start_index = int(sys.argv[2])
else:
	print "Not matching arguments, please try again! (Args expected: [Column of choice in y] [start index for checking])"
	sys.exit(0)


repeats = 33
'''
#take the dictionary from specified file
file = tb.open_file("Phi_result_ori.h5", 'r')

#sanity check to determine type of Phi stored (whether it is a singular matrix or multiples of it)
counter = count_nodes(file)
print counter
if counter > 1:
	m,n = (0,0)
	Phi = [None] * counter
	c = 0
	for node in file:
		if c!= 0:
			n_index = node.name.lstrip('data_')

			if n_index == '':
				n_index = 0
			else:
				n_index = int(n_index)
			Phi[n_index] = node[:]
		c+=1
	m,n = np.shape(Phi[0])

else:
	Phi = file.root.data[:]
	m,n = np.shape(Phi)


file.close()
'''
Phi, m, n, counter = take_data("Phi_result_trained_test_5sps.h5")
print m,n

'''
#take the dictionary from specified file
file = tb.open_file("Phi_result_trained.h5", 'r')

#sanity check to determine type of Phi stored (whether it is a singular matrix or multiples of it)
counter_ii = count_nodes(file)
print counter_ii
if counter_ii > 1:
	m,n = (0,0)
	Phi_ii = [None] * counter_ii
	c = 0
	for node in file:
		if c!= 0:
			n_index = node.name.lstrip('data_')

			if n_index == '':
				n_index = 0
			else:
				n_index = int(n_index)
			Phi_ii[n_index] = node[:]
		c+=1
	m,n = np.shape(Phi_ii[0])

else:
	Phi_ii = file.root.data[:]
	m,n = np.shape(Phi_ii)


file.close()
'''

'''

y_file = tb.open_file("y_large.h5", 'r')
y = y_file.root.data[:]
step = len(y)
#y = np.repeat(y, repeats, axis=0)		
y_file.close()
print len(y)
'''

y_file = pd.read_csv("data.csv")

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
y_data = y_file[col_choice]
x_data = range(0, step)
print np.shape(y_data[count*step:(count+1)*step])


def animate(i):
	global count
	ax1.clear()
	ax1.plot(x_data, y_data[count*step:(count+1)*step])
	#ax1.plot(x_data,y_data[count*step:(count+1)*step])
	count +=1

ani = FuncAnimation(fig, animate, interval=2000)#frames=np.linspace(0, 2*np.pi, 128), init_func=init, blit=True)
plt.show()
sys.exit(0)
'''

y_data = y_file[col_choice].as_matrix()
print type(y_data)
step = m * counter


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
	#y_cur = y[i:i+step]
	'''
	if i == 3:
		print y_data[92115:92140]
		#y_data[92115:92140] = 0.03
		
		y_data[92135] = 0.03
		y_data[92136] = 0.03
		y_data[92137] = 0.03
		#y_data = np.abs(y_data)
	'''
	
	y_cur = y_data[(start_index + i*step):(start_index + (i+1)*step)]
	#\y_cur = y_data[(i+12)*step: (i+12+1)*step]
	print y_cur[0], i
	y_act = y_cur
	y_cur = y_cur * scale
	
	#y_cur = y
	#print 'slice '+ str(i/step)
	print 'slice' + str(i)
	print y_cur[0]
	#cur_s = ((i/step)+1)* 5
	#cur_s = ((i/step)+1)* 1 #use this for the large Phi one, since this is gonna be done per thread
	cur_s =  10
	t0 = time.time()
	#Phi_in = None
	'''
	if i*step >= 128000:
		Phi_in = Phi_ii
	else:
		Phi_in = Phi
	'''
	Phi_in = Phi
	x_test, sparsity, mp_time = process_y(Phi_in,y_cur, cur_s)
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

'''
y_tst = None

if counter > 1:
	for k in range(0,np.shape(x_list[0])[1]):
		slc = np.dot(Phi[k], x_list[0][:, k])/scale
		if y_tst is None:
			y_tst = slc
		else:
			y_tst = np.concatenate((y_tst, slc))

else:
	y_tst = np.dot(Phi, x_list[0])


y_len = y_list[0]
y_tst = y_tst[0:25600]
y_tst = np.reshape(y_tst, (len(y_tst), ))
print np.shape(y_tst), np.shape(y_len)

jdex = range(0, 25600)

y_T = (y_len - y_tst)**2 
print np.shape(y_T)

plt.figure(1)
plt.subplot2grid((1,1), (0,0), colspan=1)
plt.plot(jdex, y_len, 'ro')
#trend1 = trendline_fit(jdex, y_len)
#plt.plot(jdex, trend1(y_len), 'r--')
#plt.plot(jdex, y_tst, 'go')
plt.plot(jdex, y_T, 'bo')
#trend2 = trendline_fit(jdex, y_tst)
#plt.plot(jdex, trend2(y_tst), 'g--')
plt.ylabel('original vs tested y')
plt.xlabel('idx')

plt.show()

print np.sum(np.abs(y_len - y_tst)**2)
#print l2_norm(y_len- y_tst)
sys.exit(0)
'''

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
		y_test = np.dot(Phi_tst, x_list[j])
	
	y_test = np.reshape(y_test, (len(y_test), 1) )
	
	
	
	print "y_test produced (Init):", y_test[0]
	y_test = y_test/scale
	print "Y_test produced (now and stored): ", y_test[0]
	#print y_test


	'''
	y_len = y_list[j]
	y_tst = y_test
	#y_tst = np.reshape(y_tst, (len(y_tst), ))
	print np.shape(y_tst), np.shape(y_len)
	jdex = np.array(range(0,len(y_test)))
	jdex = j * 25600 + jdex

	plt.figure(1)
	plt.subplot2grid((1,1), (0,0), colspan=1)
	plt.plot(jdex, y_len, 'ro')
	plt.plot(jdex, y_tst, 'go')
	#trend1 = trendline_fit(jdex, y_len)
	#trend2 = trendline_fit(jdex, y_tst)
	
	plt.ylabel('original vs tested y')
	plt.xlabel('idx')

	plt.show()
	'''

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

file_create("x_trained_result.h5", x_list, np.shape(x_list[0])[0], np.shape(x_list[0])[1])
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
f, axgrid = plt.subplots(3)

ind = np.arange(len(sparsity_list))
width = 0.5
rect0 = axgrid[0].bar(ind, error_list, color='r')
axgrid[0].set_ylabel('RE values')
axgrid[0].set_title('Recovery_Error of {} vs Sparsity'.format(chosen_mp))
axgrid[0].set_xticks(ind+width/2)
axgrid[0].set_xticklabels(tuple(sparsity_list))
#axgrid[0].legend(error_list, ('RE for different sparsity values'))


rect1 = axgrid[1].bar(ind, rms_list, width, color='b')
axgrid[1].set_ylabel('RMS_Values')
axgrid[1].set_title('RMS for {} vs Sparsity'.format(chosen_mp))
axgrid[1].set_xticks(ind+width/2)
axgrid[1].set_xticklabels(tuple(sparsity_list)) #cannot directly take an array
#axgrid[1].legend(rms_list, ('RMS values for each sparsity_values')) 	

rect2 = axgrid[2].bar(ind, runtime_list, width, color='b')
axgrid[2].set_ylabel('Runtimes (s)')
axgrid[2].set_title('Runtimes for {} vs Sparsity'.format(chosen_mp))
axgrid[2].set_xticks(ind+width/2)
axgrid[2].set_xticklabels(tuple(sparsity_list)) #cannot directly take an array
#axgrid[2].legend(runtime_list, ('Runtime values'))


autolabel(rect0, axgrid[0])
autolabel(rect1, axgrid[1])
autolabel(rect2, axgrid[2])


plt.show()

sys.exit(0)









