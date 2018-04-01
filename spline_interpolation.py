"""-*- coding: utf-8; -*-

(c) 2016 Carlos Toro, catoro@gmail.com

This file is part of the MIDAS project

MIDAS is registered software; you cannot redistribute it and/or modify
without express knowledge of vicomtech, parts of this software are
distributed WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the LGPL License
for more details.

You should have received a copy of the LGPL License along with this software.
If not, see <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html>.

"""

import sys
import csv
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
from scipy.fftpack import rfft
from pympler import asizeof
import time
#import numpy.alen as alen

from utils import *

__author__ = "Carlos Toro"
__date__ = "Thu Dec 14 10:48:49 2017"
__copyright__ = "Copyright Carlos Toro"
__license__ = "LGPL"
__version__ = "1.0.1"
__maintainer__ = "catoro"
__email__ = "catoro@gmail.com"
__status__ = "Development"

tittle_pic = "Spline interpolation"
tittle_pic2 = "FFT model"
spline_degree = 3
sliced = 25600
pts_spline_approx = 25600
#FILE = "/Users/catoro/Code/VibrationAnalysis/Datasets/29_02_2016 3_34_58_ pm.txt"
TOOLS = "pan,wheel_zoom,box_zoom,reset,save"


def bspleval(x, knots, coeffs, order, debug=False):
    '''
    Evaluate a B-spline at a set of points.

    Parameters
    ----------
    x : list or ndarray
        The set of points at which to evaluate the spline.
    knots : list or ndarray
        The set of knots used to define the spline.
    coeffs : list of ndarray
        The set of spline coefficients.
    order : int
        The order of the spline.

    Returns
    -------
    y : ndarray
        The value of the spline at each point in x.
    '''

    k = order
    t = knots
    m = np.alen(t)
    npts = np.alen(x)
    print m-1, k+1, npts, type()
    B = np.zeros((m-1,k+1,npts))

    if debug:
        print('k=%i, m=%i, npts=%i' % (k, m, npts))
        print('t=', t)
        print('coeffs=', coeffs)

    ## Create the zero-order B-spline basis functions.
    for i in range(m-1):
        B[i,0,:] = float64(logical_and(x >= t[i], x < t[i+1]))

    if (k == 0):
        B[m-2,0,-1] = 1.0

    ## Next iteratively define the higher-order basis functions, working from lower order to higher.
    for j in range(1,k+1):
        for i in range(m-j-1):
            if (t[i+j] - t[i] == 0.0):
                first_term = 0.0
            else:
                first_term = ((x - t[i]) / (t[i+j] - t[i])) * B[i,j-1,:]

            if (t[i+j+1] - t[i+1] == 0.0):
                second_term = 0.0
            else:
                second_term = ((t[i+j+1] - x) / (t[i+j+1] - t[i+1])) * B[i+1,j-1,:]

            B[i,j,:] = first_term + second_term
        B[m-j-2,j,-1] = 1.0

    if debug:
        plt.figure()
        for i in range(m-1):
            plt.plot(x, B[i,k,:])
        plt.title('B-spline basis functions')

    ## Evaluate the spline by multiplying the coefficients with the highest-order basis functions.
    y = zeros(npts)
    for i in range(m-k-1):
        y += coeffs[i] * B[i,k,:]

    if debug:
        plt.figure()
        plt.plot(x, y)
        plt.title('spline curve')
        plt.show()

    return(y)


if len(sys.argv) == 5:
    y_fl = sys.argv[1]
    y_col = sys.argv[2]
    start_index = int(sys.argv[3])
    repeats = int(sys.argv[4])

else:
    print "Please give arguments with the form of:  [y_file] [chosen_column] [starting_index] [repeats]" 
    sys.exit(0)
'''
P1 = pd.read_csv(y_fl)
YAxisVib_1 = np.array(P1[y_col])
'''
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




print "time elapsed for each y_slice: "
for i in range(0, repeats):
	print 'slice' + str(i)
	t0 = time.time()
	start = start_index + i * sliced
	end = start_index + (i+1) * sliced
	print start,end
	#y_cur = YAxisVib_1[start:end]
	y_cur = y_data[start:end]
	#y_cur = YAxisVib_1[(start_index + i*sliced):(start_index + (i+1)*sliced)]
	#x = np.arange(start_index + i*sliced,start_index + (i+1)*sliced)
	#x = range(start_index + i*sliced,start_index + (i+1)*sliced)
	x  = range(start,end)
	tck = interpolate.splrep(x, y_cur, s=0)
	print type(tck[0][0]),type(tck[1][0])
	knot = np.reshape(tck[0], (len(tck[0]), 1))
	coef = np.reshape(tck[1], (len(tck[1]), 1))
	result = np.concatenate((knot , coef), axis=1)
	#evaluation = bspleval(x, tck[0], tck[1], 3, debug=False)
	#print evaluation

	#print tck[0], start_index + i*sliced, start_index + (i+1)*sliced, asizeof.asized(tck)
	'''
	print np.shape(result), np.shape(knot), np.shape(coef), tck[0]
	sys.exit(0)
	'''
	file_create("spline_inter/spl_inter_"+str(i)+".h5", result, np.shape(result)[0],np.shape(result)[1])
	#print tck[0], x.min(), x.max(), asizeof.asized(tck)

	#np.savetxt("spline_inter/spl_inter_"+str(i)+".csv", (tck[0].T,tck[1].T),delimiter=",")

	#y_cur = y
	#print 'slice '+ str(i/step)
	

	#tf = time.time()
	#y_list.append(y_cur)
	y_list.append(y_cur)
	x_list.append(tck)
	tf = time.time()
	runtime_list.append(tf-t0)
	#sparsity_list.append(sparsity)	
	print "total time: " + str(tf-t0), np.shape(tck), np.count_nonzero(tck), sys.getsizeof(tck)
	print "" 

runtime_list = np.reshape(runtime_list, (len(runtime_list), 1))
file_create(output_file, x_list, np.shape(x_list[0])[0], np.shape(x_list[0])[1])
file_create("spline_inter/spl_inter_runtime.h5", runtime_list, np.shape(runtime_list)[0], np.shape(runtime_list)[1])
save_y = None
while save_y not in ['Y', 'y', 'N', 'n']: #pythonic manner, vs:
#while not ((save_y != 'Y') and (save_y != 'y') and (save_y != 'N') and (save_y != 'n')): #non pythonic manner
	save_y = raw_input("Do you want to save the y version of the results as well? [Y/N] ")

if save_y == 'Y' or save_y == 'y':
	y_fname = raw_input("Please state the file name (w/ .h5 prefix. File will be saved in HDF5 format) => ")
	#print np.shape(y_test_list[0])
	file_create(y_fname, y_test_list, np.shape(y_test_list[0])[0], np.shape(y_test_list[0])[1])

'''

# tck = interpolate.splrep(x, y, k=spline_degree)
xnew = np.linspace(x.min(), x.max(), pts_spline_approx)
ynew = interpolate.splev(xnew, tck, der=0)


fft_y = np.abs(rfft(y))
fft_ynew = np.abs(rfft(ynew))

output_file("legend.html", title="legend.py example")
p1 = figure(title=tittle_pic, tools=TOOLS, plot_width=1200, plot_height=400)
p2 = figure(title=tittle_pic2, tools=TOOLS, plot_width=1200, plot_height=400)

p1.circle(x, y, legend="Control points", color="red", alpha=0.5)
p1.line(xnew, ynew, legend="interpolation", color="blue", alpha=0.8)

p2.line(x, fft_y, legend="Control points", color="red", alpha=0.5)
p2.line(xnew, fft_ynew, legend="interpolation", color="blue", alpha=0.8)

show(column(p1, p2))
'''