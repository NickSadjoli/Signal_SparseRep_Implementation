import numpy as np
import pandas as pd
from utils import *
from scipy.fftpack import dct, idct
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
from numpy.linalg import norm
import time


if len(sys.argv) == 5:
    start_index = int(sys.argv[1])
    y_fl = sys.argv[2]
    compressed_percentage = float(sys.argv[3])
    repeats = int(sys.argv[4])
else:
    print "Invalid, please try again! \n(Args expected: [start index for checking] [y's file(\w .h5 subfix)] [# compression ratio(%)] [# repeats/testing to be done])"
    sys.exit(0)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

	


y_whole, _, _, _ = take_data(y_fl)
step = 25600
runtime_list = []
for i in range(0,repeats):
    y_data = y_whole[i*step: (i+1)*step]
    t0 = time.time()
    '''
    y_init = np.zeros((np.shape(y_data)[0], 1))
    y_init[:] = y_data[:]
    file_create("dct/initial_y.h5", y_init, np.shape(y_init)[0], np.shape(y_init)[1])
    '''
    y_data = np.reshape(y_data, (len(y_data), ))
    x = np.arange(0, len(y_data))
    #print np.shape(y_data), type(y_data), np.count_nonzero(y_data)

    #transform y using DCT into freq domain, and get dct representation(coeff)
    y_dct= dct(y_data,norm='ortho')

    #print y_dct, len(y_dct)
    #x_compress = np.arange(0, len(y_dct))

    #sort dct coeff index
    dct_sort_id = np.argsort(y_dct)[::-1]
    #print y_dct_sort, dct_sort_id
    #print dct_sort_id

    #find coeffs of y_dct that constitutes compressed_percentage% of whole y 
    need = 1
    while(norm(y_dct[dct_sort_id[0:need]]) / norm(y_dct) < (compressed_percentage/100)):
        need = need + 1
    print need, compressed_percentage/100

    #zero the coeff that is not really doing contribution to compressed_percentage% of y (thresholding)
    y_dct[dct_sort_id[need+1:]] = 0
    #y_cmp = y_dct

    #get compressed signal by inverse dct the finalized coeffs
    y_cmp = idct(y_dct, norm='ortho')
    print np.shape(y_cmp)

    '''
    output_file("legend.html", title="legend.py example")
    p1 = figure(title="Original", tools=TOOLS, plot_width=800, plot_height=400)
    p2 = figure(title="After DCT", tools=TOOLS, plot_width=800, plot_height=400)

    #p1.circle(x, y, legend="Control points", color="red", alpha=0.5)
    p1.line(x, y_data, legend="Control Points", color="blue", alpha=0.8)

    #p2.line(x, fft_y, legend="Control points", color="red", alpha=0.5)
    p2.line(x_compress, y_dct, legend="After DCT", color="blue", alpha=0.8)

    show(column(p1,p2))
    '''
    y_cmp = np.reshape(y_cmp, (len(y_cmp), 1))
    file_create("dct/dct_"+str(i)+".h5", y_cmp, np.shape(y_cmp)[0], np.shape(y_cmp)[1])
    tf = time.time()
    print "compressed slice number: ", i, "kept_ coefficients: ", need, "||compressed nonzero elements:", np.count_nonzero(y_cmp), "detected data type: ", type(y_cmp[0]), tf-t0
    runtime_list.append(tf-t0)

runtime_list = np.reshape(runtime_list, (len(runtime_list), 1))
file_create("dct/dct_runtime.h5", runtime_list, np.shape(runtime_list)[0], np.shape(runtime_list)[1])
