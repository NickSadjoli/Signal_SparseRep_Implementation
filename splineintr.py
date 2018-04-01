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

import numpy as np
import pandas as pd
from scipy import interpolate
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
from scipy.fftpack import rfft

__author__ = "Carlos Toro"
__date__ = "Thu Dec 14 10:48:49 2017"
__copyright__ = "Copyright Carlos Toro"
__license__ = "LGPL"
__version__ = "1.0.1"
__maintainer__ = "catoro"
__email__ = "catoro@gmail.com"
__status__ = "Development"

tittle_pic = "25.6 kHz vibration signal sample from Spindle"
tittle_pic2 = "FFT model"
spline_degree = 3
sliced = 25600
pts_spline_approx = 25600
FILE = "data.csv"
#FILE = "/Users/catoro/Code/VibrationAnalysis/Datasets/29_02_2016 3_34_58_ pm.txt"
TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

P1 = pd.read_csv(FILE)
YAxisVib_1 = np.array(P1['VibSpindle_X'])
y = YAxisVib_1[:sliced]
x = np.arange(0, y.size)


tck = interpolate.splrep(x, y, k=spline_degree)
xnew = np.linspace(x.min(), x.max(), pts_spline_approx)
ynew = interpolate.splev(xnew, tck, der=0)
print np.shape(ynew)


fft_y = np.abs(rfft(y))
fft_ynew = np.abs(rfft(ynew))

output_file("legend.html", title="legend.py example")
p1 = figure(title=tittle_pic, tools=TOOLS, plot_width=1200, plot_height=400)
p2 = figure(title=tittle_pic2, tools=TOOLS, plot_width=1200, plot_height=400)

#p1.circle(x, y, legend="Control points", color="red", alpha=0.5)
p1.line(xnew, ynew, legend="spindle signal", color="red", alpha=0.8)

p2.line(x, fft_y, legend="Control points", color="red", alpha=0.5)
p2.line(xnew, fft_ynew, legend="interpolation", color="blue", alpha=0.8)

show(column(p1, p2))
