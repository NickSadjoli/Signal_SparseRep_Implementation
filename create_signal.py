import numpy as np
import tables as tb
import sys
from utils import file_create

'''
Script requires installation of tables (PyTables) first!
Script is modification of script found here: http://www.philippsinger.info/?p=464
'''

f_name = None
m = None
n = None

if len(sys.argv) == 4:
	f_name = sys.argv[1]
	m = int(sys.argv[2])
	n = int(sys.argv[3])
else:
	print "Please input following arguments: file_name(without '.h5' prefix) #measurements(m) #samples(n)"
	sys.exit(0)

'''
def file_create(file_name, m, n):
	file_name = f_name + '.h5'

	#f = tb.open_file('Phi.h5', 'w')
	f = tb.open_file(file_name, 'w')
	filters = tb.Filters(complevel=5, complib='blosc')

	#because by default numpy (in other scripts) operates using float64, this is necessary
	out = f.create_carray(f.root, 'data', tb.Float64Atom(), shape=(m, n), filters=filters) 

	print "h5 file created, now putting Phi from memory to file..."
	  
	step = 1000 #this is the number of rows we calculate each loop (example was using bl)
	#this may not the most efficient value
	#look into buffersize usage in PyTables and adopt the buffersite of the
	#carray accordingly to improve specifically fetching performance
	 
	#b = b.tocsc() #we slice b on columns, csc improves performance #not necessary in this case
	 
	#this can also be changed to slice on rows instead of columns (which is what will be done for Phi)
	for i in range(0, n, step):
	  out[:,i:min(i+step, n)] = Phi[:, i:min(i+step, n)] # initially, example was using this => (a.dot(b[:,i:min(i+bl, l)])).toarray()
	  print i
	print "Phi saving done, closing file..."
	 
	f.close()
'''

#Phi = np.random.normal(0,0.5,[6250,13000])

Phi = np.random.normal(0,1e-5,[m,n])

'''
a = rand(2000,2000, format='csr') #imagine that many values are stored in this matrix and that sparsity is low
b = a.T
'''
print "generated Phi on memory, creating h5 file pointer..."

#m, n = Phi.shape[0], Phi.shape[1]

#file_name = f_name + '.h5'
file_create(f_name, Phi, m, n)
