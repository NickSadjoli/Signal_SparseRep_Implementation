import numpy as np
import tables as tb
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, fftshift

from utils import *

from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column

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


'''
if len(sys.argv) == 3:
    col_choice= sys.argv[1]
    start_index = int(sys.argv[2])
else:
	print "Not matching arguments, please try again! (Args expected: [Column of choice in y] [start index for checking])"
	sys.exit(0)
'''


if len(sys.argv) == 6:
    y_test_file = sys.argv[1]
    y_file_ori = sys.argv[2]
    col_choice= sys.argv[3]
    repeats= int(sys.argv[4])
    start_index = int(sys.argv[5])
else:
    print "Not matching arguments, please try again!"
    sys.exit(0)

y_tested, m, n, _ = take_data(y_test_file)


y_file = pd.read_csv(y_file_ori)
y_data = y_file[col_choice]

#FILE = "/Users/catoro/Code/VibrationAnalysis/Datasets/29_02_2016 3_34_58_ pm.txt"
tittle_pic = "Recovery from Sparse Representation"
tittle_pic2 = "FFT model"
TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
#print_sizes(y_tested[0], y_data[0:m])
'''
for i in y_tested:
	print i 
sys.exit(0)
'''

'''
### TO BE USED LATER: #####
# Below is a block used to ANIMATE a set of data tp create a graph that updates dynamically.
# Should be able to be put to good use for showcasing the followig fft results.


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
count = 0
step = 25600 
y_data = y_file[col_choice].as_matrix()
x_data = range(0, step)
print np.shape(y_data[count*step:(count+1)*step])


def animate(i):
    global count
    ax1.clear()
    ax1.plot(x_data, y_data[count*step:(count+1)*step])
    #ax1.plot(x_data,y_data[count*step:(count+1)*step])
    count +=1

ani = FuncAnimation(fig, animate, interval=1000)#frames=np.linspace(0, 2*np.pi, 128), init_func=init, blit=True)
plt.show()
sys.exit(0)
''' 

for i in range(0, repeats):
    #print y_tested[i]

    #take test sample of y and reshape to one sequence for fft to work effectively
    y_tst = y_tested[i]
    y_tst = np.reshape(y_tst, (len(y_tst), ))
    y_tst_fft = fft(y_tst)

    #y_tst_fft = np.fft.fft(y_tst)

    '''
    y_tst_fft = np.fft.fft(y_tested[i])
    y_tst_fft = np.reshape(y_tst_fft, (len(y_tst_fft), 1))
    '''
    #now take the original dataset and do the same 
    y_data_smp = y_data[start_index + i*m: start_index + (i+1)*m]
    y_data_smp = np.reshape(y_data_smp, (len(y_data_smp), ))
    y_data_fft = fft(y_data_smp)
    #y_data_fft = np.fft.fft(y_data_smp)
    '''
    y_data_fft = np.fft.fft(y_data[i*m:(i+1)*m])
    y_data_fft = np.reshape(y_data_fft, (len(y_data_fft), 1))
    '''
    #number of data points in y_test and y_data_smp
    N = (np.shape(y_data_fft))[0]

    #frequency of y (or rather frequency of sensor)
    f = float(25600) / 2 
    #period obtained from frequency of sensor
    T = 1/f

    #get suitables fft frequency for display
    idx = fftfreq(N, T)
    idx = fftshift(idx)
    y1 = fftshift(y_tst_fft)
    y2 = fftshift(y_data_fft)

    '''
    #create an index range starting from 0 till 0.5 of y's frequency (1/2 of f), with 
    idx = np.linspace(0.0, 0.5 * f, )
    '''
    #print_sizes(y_data_fft, y_tst_fft)
    
    '''
    print y_data[i*m:(i+1)*m]
    print "---"
    print y_tested[i]
    '''
    '''
    print y_data_fft
    print '---'
    print y_tst_fft
    '''

    #idx = np.arange(np.shape(y_tst_fft)[0])
    #if i >= 10:
    
    #dex = np.arange(np.shape(y_tst_fft)[0])

    # tck = interpolate.splrep(x, y, k=spline_degree)
    dex = range(start_index + i*m, start_index + (i+1)*m)
    '''
    xnew = np.linspace(x.min(), x.max(), pts_spline_approx)
    ynew = interpolate.splev(xnew, tck, der=0)


    fft_y = np.abs(rfft(y))
    fft_ynew = np.abs(rfft(ynew))
    '''
    output_file("legend.html", title="legend.py example")
    p1 = figure(title=tittle_pic, tools=TOOLS, plot_width=1200, plot_height=400)
    p2 = figure(title=tittle_pic2, tools=TOOLS, plot_width=1200, plot_height=400)

    p1.circle(dex, y_data_smp, legend="Control points", color="red", alpha=0.5)
    p1.line(dex, y_tst, legend="interpolation", color="blue", alpha=0.8)

    p2.line(idx, 1.0/N * np.abs(y2), legend="Control points", color="red", alpha=0.5)
    p2.line(idx, 1.0/N * np.abs(y1), legend="interpolation", color="blue", alpha=0.8)

    show(column(p1, p2))

    '''
    plt.figure(1)
    plt.plot(dex, y_data_smp, 'r-')
    plt.plot(dex, y_tst, 'g-')
    plt.ylabel('Original vs tested')
    plt.xlabel('data points')
    plt.show()

    plt.figure(1)
    plt.subplot2grid((1,1), (0,0), colspan=1)
    plt.plot(idx, 1.0/N * np.abs(y2), 'ro')
    plt.plot(idx, 1.0/N * np.abs(y1), 'g-')
    plt.ylabel('FFT of original vs tested')
    plt.xlabel('Frequency')
    plt.show()

    '''



    
    #print l2_norm(y_data_fft - y_tst_fft)
    #print "######"



#print y_tested[0]
