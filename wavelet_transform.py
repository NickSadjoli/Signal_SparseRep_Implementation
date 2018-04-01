import numpy as np
from pywt import dwt, idwt
import pywt
import pandas as pd
from utils import *
#from scipy.fftpack import dwt, idwt
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
from numpy.linalg import norm
import time
from statsmodels.robust import mad 


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
    file_create("dwt/initial_y.h5", y_init, np.shape(y_init)[0], np.shape(y_init)[1])
    '''
    y_data = np.reshape(y_data, (len(y_data), ))
    
    x = np.arange(0, len(y_data))
    #print np.shape(y_data), type(y_data), np.count_nonzero(y_data)	
    #transform y using dwt into freq domain, and get dwt representation(approximate and detailed coeff)
    #y_dwtA, y_dwtD = dwt(y_data, 'db4')
    coeff = pywt.wavedec(y_data, "db20", mode="per")

    '''
    #print y_dwtA, y_dwtD, np.shape(y_dwtA), np.shape(y_dwtD)


    #sys.exit(0)

    #print y_dwt, len(y_dwt)
    #x_compress = np.arange(0, len(y_dwt))

    #sort dwt coeff index
    dwtA_sort_id = np.argsort(y_dwtA)[::-1]

    #print y_dwt_sort, dwt_sort_id
    #print dwt_sort_id

    #find coeffs of y_dwt that constitutes compressed_percentage% of whole y 
    needA = 1
    while(norm(y_dwtA[dwtA_sort_id[0:needA]]) / norm(y_dwtA) < (compressed_percentage/100)):
        needA = needA + 1
        #sprint needA
    print needA, compressed_percentage/100
    #zero the coeff that is not really doing contribution to compressed_percentage% of y (thresholding)
    y_dwtA[dwtA_sort_id[needA+1:]] = 0
    #y_dwtA = np.reshape(y_dwtA, (len(y_dwtA), 1))

    dwtD_sort_id = np.argsort(y_dwtD)[::-1]

    needD = 1

    while(norm(y_dwtD[dwtD_sort_id[0:needD]]) / norm(y_dwtD) < (compressed_percentage/100)):
        needD = needD + 1
    print needD, compressed_percentage/100
    #zero the coeff that is not really doing contribution to compressed_percentage% of y (thresholding)
    y_dwtD[dwtD_sort_id[needD+1:]] = 0
    #y_dwtD = np.reshape(y_dwtD, (len(y_dwtD), 1))
    
    #y_cmp = np.concatenate((y_dwtA, y_dwtD),axis=1)
    #print np.shape(y_cmp)
    #get compressed signal by inverse dwt the finalized coeffs
    y_cmp = idwt(y_dwtA, y_dwtD, 'db4')
    '''


    sigma = mad(coeff[-1])
    threshold = sigma * np.sqrt(2*np.log(len(y_data)))
    #coeff[1:] = (pywt.threshold(i, value=threshold, mode="soft") for i in coeff[1:])
    coeff[1: ] = (pywt.threshold(i, value=threshold, mode="hard") for i in coeff[1:])
    y_cmp = pywt.waverec(coeff, "db20", mode="per")
    #print np.shape(y_cmp)

    '''
    output_file("legend.html", title="legend.py example")
    p1 = figure(title="Original", tools=TOOLS, plot_width=800, plot_height=400)
    p2 = figure(title="After dwt", tools=TOOLS, plot_width=800, plot_height=400)

    #p1.circle(x, y, legend="Control points", color="red", alpha=0.5)
    p1.line(x, y_data, legend="Control Points", color="blue", alpha=0.8)

    #p2.line(x, fft_y, legend="Control points", color="red", alpha=0.5)
    p2.line(x_compress, y_dwt, legend="After dwt", color="blue", alpha=0.8)

    show(column(p1,p2))
    '''
    y_cmp = np.reshape(y_cmp, (len(y_cmp), 1))
    file_create("dwt/dwt_"+str(i)+".h5", y_cmp, np.shape(y_cmp)[0], np.shape(y_cmp)[1])
    tf = time.time()
    #print "compressed slice number: ", i, "kept_ coefficients: ", needA, needD,  "||compressed nonzero elements:", np.count_nonzero(y_cmp), "detected data type: ", type(y_cmp[0]), tf-t0
    print "compressed slice number: ", i,  "||compressed nonzero elements:", np.count_nonzero(y_cmp), "detected data type: ", type(y_cmp[0]), tf-t0
    runtime_list.append(tf-t0)

runtime_list = np.reshape(runtime_list, (len(runtime_list), 1))
file_create("dwt/dwt_runtime.h5", runtime_list, np.shape(runtime_list)[0], np.shape(runtime_list)[1])
