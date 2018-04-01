'''
#code is based on paper: http://www6.cityu.edu.hk/seam/Papers/A%20novel%20signal%20compression%20method%20based%20on%20optimal%20ensemble%20empirical%20mode%20decomposition%20for%20bearing%20vibration%20signals.pdf
'''

from PyEMD import EEMD, EMD
#from EEMD_Object import EEMD

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

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"


if len(sys.argv) == 5:
    start_index = int(sys.argv[1])
    y_fl = sys.argv[2]
    compressed_percentage = float(sys.argv[3])
    repeats = int(sys.argv[4])
else:
    print "Invalid, please try again! \n(Args expected: [start index for checking] [y's file(\w .h5 subfix)] [# compression ratio(%)] [# repeats/testing to be done])"
    sys.exit(0)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

	

emd = EMD()
#emd.extrema_detection
y_whole, _, _, _ = take_data(y_fl)

y_data = y_whole[0:25600]
y_data = np.reshape(y_data, (len(y_data), ))

x = np.arange(0,len(y_data))
t0 = time.time()
print np.shape(y_data), np.shape(x)

IMFs = emd.emd(y_data, max_imf=1)
print np.shape(IMFs)

# Plot results
output_file("legend.html", title="legend.py example")
figures = []
tf = time.time()
print tf-t0
#p1 = figure(title="Original vs Sparse, DCT, and DWT", tools=TOOLS, plot_width=800, plot_height=400)
for i in range(IMFs):
    fig = figure(title="IMF_"+str(i), tools=TOOLS, plot_width=800, plot_height=400)
    fig.line(x, IMFs[i], legend="DWT", color="green", alpha = 0.7)
    figures.append(fig)


show(column(figures[0],figures[1]))



