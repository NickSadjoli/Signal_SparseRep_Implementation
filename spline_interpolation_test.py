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

tittle_pic = "Spline interpolation"
tittle_pic2 = "FFT model"
spline_degree = 3
sliced = 25600
pts_spline_approx = 25600
#FILE = "/Users/catoro/Code/VibrationAnalysis/Datasets/29_02_2016 3_34_58_ pm.txt"
TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

file = pd.read_csv("spline_inter/spl_inter_0.csv", header=None)
data = file.as_matrix()
print data[0]
#print np.shape(data[0]), np.shape(data[1]), np.shape(data)
# tck = interpolate.splrep(x, y, k=spline_degree)
tck = (data[0], data[1], 3)
print sys.getsizeof(data)
xnew = np.linspace(0, 25599, pts_spline_approx)
ynew = interpolate.splev(xnew, tck, der=0)


#fft_y = np.abs(rfft(y))
fft_ynew = np.abs(rfft(ynew))

output_file("legend.html", title="legend.py example")
p1 = figure(title=tittle_pic, tools=TOOLS, plot_width=1200, plot_height=400)
p2 = figure(title=tittle_pic2, tools=TOOLS, plot_width=1200, plot_height=400)

#p1.circle(x, y, legend="Control points", color="red", alpha=0.5)
p1.line(xnew, ynew, legend="interpolation", color="blue", alpha=0.8)

#p2.line(x, fft_y, legend="Control points", color="red", alpha=0.5)
p2.line(xnew, fft_ynew, legend="interpolation", color="blue", alpha=0.8)

show(column(p1, p2))