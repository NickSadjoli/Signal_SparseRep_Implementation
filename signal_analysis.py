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
from scipy.fftpack import fft, fftfreq, fftshift
from scipy import signal


if len(sys.argv) == 3:
    y_fl = sys.argv[1]
    y_col = sys.argv[2]

else:
    print "Please give arguments with the form of:  [y_file] [chosen_column]"
    sys.exit(0)


fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
count = 0
step = 25600 
y_file = pd.read_csv(y_fl)
y_data = y_file[y_col].as_matrix()
#x_data = range(0, step)
print np.shape(y_data[count*step:(count+1)*step])
y_data_slice = []
x_data = []


def animate(i):
    global count
    global y_data_slice
    global x_data

    y_data_slice = y_data[count*step:(count+1)*step]
    x_data = range(count*step, (count+1)*step)
    '''
    y_new = y_data[count*step:(count+1)*step]
    x_new = range(count*step, (count+1)*step)
    y_data_slice = np.concatenate((y_data_slice, y_new))
    x_data = np.concatenate((x_data, x_new))
    '''
    ax1.clear()
    ax1.plot(x_data, y_data_slice, 'r-')
    y_data_fft = fft( np.reshape(y_data_slice, (len(y_data_slice), )) )
    #y_data_fft = fft(np.reshape(y_new, (len(y_new), )) )

    N = (np.shape(y_data_fft))[0]

    #frequency of y (or rather frequency of sensor)
    f = float(step) / 2 
    #period obtained from frequency of sensor
    T = 1/f
    
    idx = fftfreq(N,T)
    idx = fftshift(idx)
    yt = fftshift(y_data_fft)
    '''
    idx = np.linspace(0, 1.0/(2.0*T), N/2)
    yt = y_data_fft
    '''
    ax2.clear()
    ax2.plot(idx, 1.0/N * np.abs(yt))
    #ax2.plot(idx, 2.0/N * yt[0:N/2])
    #ax1.plot(x_data,y_data[count*step:(count+1)*step])
   
    f, Psd = signal.periodogram(y_data_slice, len(y_data_slice))

    ax3.clear()
    ax3.semilogy(f,Psd)
    count +=1

ani = FuncAnimation(fig, animate, interval=2000)#frames=np.linspace(0, 2*np.pi, 128), init_func=init, blit=True)
plt.show()
sys.exit(0)
