import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from pywt import dwt, idwt
from utils import *

from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

y_original, _, _, _ = take_data("test_data.h5")
#Phi_spscode, _, _, _  = take_data("Phi_result_nelksvd_spindle_xidle_1-5.h5")
Phi_spscode, _, _, _  = take_data("Phi_result_demo.h5")

#spscode_runtime, _, _, _ = take_data("sparse_code1-5/sps_code_runtime.h5")
spscode_runtime, _, _, _ = take_data("sparse_code_demo/sps_code_runtime.h5")
dct_runtime, _, _, _ = take_data("dct/dct_runtime.h5")
dwt_runtime, _, _, _ = take_data("dwt/dwt_runtime.h5")
#print np.shape(y_original), np.shape(Phi_spscode), np.shape(spscode_runtime), np.shape(dct_runtime)

step = 25600

spscode_error_list = []
spscode_rms_list = []
dct_error_list = []
dct_rms_list = []
dwt_error_list = []
dwt_rms_list = []

for i in range(0,40):
    y_ori = y_original[i*step:(i+1)*step]
    #spscode_x, _, _, _  = take_data("sparse_code1-5/sps_code"+str(i)+".h5")
    spscode_x, _, _, _  = take_data("sparse_code_demo/sps_code"+str(i)+".h5")
    #print np.shape(Phi_spscode.T), np.shape(spscode_x)
    spscode_test = np.dot(Phi_spscode.T, spscode_x)
    spscode_test = np.reshape(spscode_test, (len(y_ori), 1))
    #print y_ori, spscode_test
    spscode_error = Recovery_Error(y_ori, spscode_test) * 100
    spscode_rms = RMS(y_ori, spscode_test) * 100
    spscode_error_list.append(spscode_error)
    spscode_rms_list.append(spscode_rms)

    dct_test, _, _, _  = take_data("dct/dct_"+str(i)+".h5")
    #dct_test = idct(dct_test, norm='ortho')
    dct_error = Recovery_Error(y_ori,dct_test) * 100
    dct_rms = RMS(y_ori, dct_test) * 100
    dct_error_list.append(dct_error)
    dct_rms_list.append(dct_rms)

    dwt_test, _, _, _ = take_data("dwt/dwt_"+str(i)+".h5")
    dwt_error = Recovery_Error(y_ori,dwt_test) * 100
    dwt_rms = RMS(y_ori, dwt_test) * 100
    dwt_error_list.append(dwt_error)
    dwt_rms_list.append(dwt_rms)

    if i == 0:
        y_ori = np.reshape(y_ori, (len(y_ori), ))
        dct_test = np.reshape(dct_test, (len(dct_test), ))
        dwt_test = np.reshape(dwt_test, (len(dwt_test), ))
        spscode_test = np.reshape(spscode_test, (len(spscode_test), ))
        x = np.arange(0, len(y_ori))
        output_file("legend.html", title="legend.py example")
        p1 = figure(title="Original vs Sparse, DCT, and DWT", tools=TOOLS, plot_width=800, plot_height=400)
        p1.circle(x, y_ori, legend="Control points", color="red", alpha=0.6)
        p1.line(x, dwt_test, legend="DWT", color="green", alpha = 0.7)
        p1.line(x, dct_test, legend="DCT", color="orange", alpha = 0.6)
        p1.line(x, spscode_test, legend="sparse_code", color="blue", alpha=0.3)
        show(p1)


    print spscode_error, spscode_rms, dct_error, dct_rms, dwt_error, dwt_rms, spscode_runtime[i], dct_runtime[i], dwt_runtime[i]

print "Averages:\n"
print "Methods |   Sparse_Code    ||         DCT        ||         DWT        ||\n"
print "RE      | ", np.average(spscode_error_list),"  || " , np.average(dct_error_list), "    || ", np.average(dwt_error_list), "     ||"
print "RMS     | ", np.average(spscode_rms_list)," || " , np.average(dct_rms_list), "  || ", np.average(dwt_rms_list), "  ||"
print "Runtime | ", np.average(spscode_runtime)," || " , np.average(dct_runtime), " || ", np.average(dwt_runtime), " ||"

# create plot
fig, ax = plt.subplots()
index = np.arange(len(spscode_error_list[:20]))
bar_width = 0.3
opacity = 0.8
 
rects1 = plt.bar(index, spscode_error_list[:20], bar_width,
                 alpha=opacity,
                 color='blue',
                 ecolor='black',
                 label='Sparse_Code',
                 hatch="/")
 
rects2 = plt.bar(index + bar_width, dct_error_list[:20], bar_width,
                 alpha=opacity,
                 color='red',
                 ecolor='black',
                 label='DCT',
                 hatch="o")

rects3 = plt.bar(index + (2*bar_width), dwt_error_list[:20], bar_width,
                 alpha=opacity,
                 color='yellow',
                 ecolor='black',
                 label='DWT',
                 hatch="\\")
 
plt.xlabel('n-th sample', horizontalalignment='right')
plt.ylabel('Percentage of error (in %)')
plt.title('Comparison of Recovery_Error between different methods tested', loc='left')
plt.xticks(index + bar_width, tuple(index))
plt.legend(loc=2, prop={'size': 22})
 
plt.tight_layout()
plt.show()

# create plot
fig, ax = plt.subplots()
index = np.arange(len(spscode_rms_list[:20]))
bar_width = 0.3
opacity = 0.8
 
rects1 = plt.bar(index, spscode_rms_list[:20], bar_width,
                 alpha=opacity,
                 color='blue',
                 ecolor='black',
                 label='Sparse_Code',
                 hatch="/")
 
rects2 = plt.bar(index + bar_width, dct_rms_list[:20], bar_width,
                 alpha=opacity,
                 color='red',
                 ecolor='black',
                 label='DCT',
                 hatch="o")

rects3 = plt.bar(index + (2*bar_width), dwt_rms_list[:20], bar_width,
                 alpha=opacity,
                 color='yellow',
                 ecolor='black',
                 label='DWT',
                 hatch="\\")
 
plt.xlabel('n-th sample', horizontalalignment='right')
plt.ylabel('Percentage of error (in %)')
plt.title('Comparison of RMS between different methods tested', loc='left')
plt.xticks(index + bar_width, tuple(index))
plt.legend(loc=2, prop={'size': 22})
 
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
index = np.arange(len(spscode_runtime[:20]))
bar_width = 0.3
opacity = 0.8
 
rects1 = plt.bar(index, spscode_runtime[:20], bar_width,
                 alpha=opacity,
                 color='b',
                 label='Sparse_Code')
 
rects2 = plt.bar(index + bar_width, dct_runtime[:20], bar_width,
                 alpha=opacity,
                 color='r',
                 label='DCT')

rects3 = plt.bar(index + (2*bar_width), dwt_runtime[:20], bar_width,
                 alpha=opacity,
                 color='y',
                 label='DWT')
 
plt.xlabel('n-th sample')
plt.ylabel('Runtime (in s)')
plt.title('Comparison of Runtime between different methods tested')
plt.xticks(index + bar_width, tuple(index))
plt.legend()
 
plt.tight_layout()
plt.show()