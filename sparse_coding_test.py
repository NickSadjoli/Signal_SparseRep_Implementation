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
'''
file = pd.read_csv("sparse_code/spscode_0.csv",header=None)
data = file.as_matrix()
'''
data, _, _, _ = take_data("sparse_code/sps_code0.h5")
print np.shape(data)


output_file("legend.html", title="legend.py example")
p1 = figure(title=tittle_pic, tools=TOOLS, plot_width=1200, plot_height=400)
p2 = figure(title=tittle_pic2, tools=TOOLS, plot_width=1200, plot_height=400)

p1.circle(x, y, legend="Control points", color="red", alpha=0.5)
p1.line(xnew, ynew, legend="sparse representation", color="blue", alpha=0.8)

p2.line(x, fft_y, legend="Control points", color="red", alpha=0.5)
p2.line(xnew, fft_ynew, legend="sparse representation", color="blue", alpha=0.8)

show(column(p1, p2))