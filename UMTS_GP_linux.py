
# coding: utf-8

# In[1]:

import pylab as pb
pb.ion()
import numpy as np
import matplotlib.pyplot
import GPy
import random
import get_loc_linux2  # Need to change to _linux
import getwcdma_linux
import time


# In[2]:

#Test the UMTS GP processing on Dummy Data
# Create some dummy location data
#Match an index and perform the operations



date =name =[]
name = 'cellid'
lat =[]
lon=[]
loc_length = 20
#build and id list for the cell names We will make the location vector a little shorter than the data vector
name =[name + str(row) for row in range(loc_length)]
# After we build the data we don't want to be in original order
random.shuffle(name)


#build some random coordinates
lat  =[42]*loc_length 
lon = [-73]*loc_length
#clat = np.zeros(loc_length)
# Create a random offset from the center location
clat = np.random.rand(loc_length)
random.shuffle(clat)
lat += clat
lon += -clat
loc =  zip(name,lat,lon)
#############################
#now create some dummy data
######
# Let's retrieve some real location data

name2, lat, lon = get_loc_linux2.get_loc()
 
if False:
    lat = lat.tolist()
    lon = lon.tolist()
    name2 = name2.tolist()
 
loc = zip(name2,lat,lon)

date ='20140100'
length_data = 10
date =[date + str(row) for row in range(length_data)]
 
dbdata = np.random.randint(0,1500,length_data)

# Randomly shuffle the columns, first transpose since the shuffle works on rows
# this helps verify that we are finding the right matches.
random.shuffle(name)

#loc = np.transpose(loc)
#These Statements just create a random set of data in the range of 0 to 750.  We want the data item in 
#position 3.  the actual data is formatted this way so we have to hard code it.

dbdata2 = np.random.randint(0,1750,length_data)
dbdata1 = np.random.randint(0,350,length_data)
dbdata3 = zip(date,name2,dbdata1,dbdata2)

print "How long is our data set?  ", "Location=", len(loc),"dbdata =", len(dbdata)
 
#print loc
#print dbdata


# In[3]:

# Get real data
# Call getwcdma.py
start = time.time()
 
cellid, dbdata2 = getwcdma_linux.getwcdma()
end = time.time()
print end - start, " seconds to get WCDMA data from the dbase"

#dbdata2 =tuple(map(tuple,dbdata2))
#Now I have to convert the dictionary item to a list?
#dbdata2 = dbdata2.
dbdata = zip(cellid,dbdata2)

print len(dbdata), dbdata2[4], cellid[4]
print dbdata2[4] * dbdata2[2], dbdata[4][0], loc[4][0]
 
if loc[4][0] == dbdata[4][0]:
    print "name = ", loc[9]
else:
    print "cell ", cellid[4], loc[4][0], loc[4][1]


# In[5]:

#Test call of cell_mm
#Inputs needed are the location data and dbdata. 
import cell_mm_linux  #need to change to _linux
import time

smean =[]
ind =[]

# Get a mean and a list of the matching cells
start = time.time()
ind, smean = cell_mm_linux.cell_mm(loc,dbdata)
end = time.time()
print end - start, " seconds to calculate the mean and correlate cells"

# Test output
# The index rows indicate where the location cell id matched data from the date, so the rows will
# match the correct location indices.
import numpy
 
mXa = map(lambda row: loc[row][1],ind[0])
mXb = map(lambda row: loc[row][2],ind[0])
cd1 = map(lambda row: smean[row],ind[0])
if True:
    cc = cd1/np.linalg.norm(cd1)
else:
    cc =cd1
a = np.array(mXa)
b = np.array(mXb)
if True:
   
    mXa = a/np.linalg.norm(a)
   
    mXb = b/np.linalg.norm(b)
else:
    mXa =a
    mXb =b

mX = np.column_stack((mXa,mXb))
length_data = len(cc)
print len(mX), mX
#print mX.reshape(10,2)
#print "a =\n", a, "length ? =", len(a)
print "cc = \n", cc, "length ? = ", len(cc)
# Now we have our input data for the GP.


# In[6]:

# sample inputs and outputs
#X = np.random.uniform(-3.,3.,(length_data,2))
# Let's change X to [a b]
X =mX
#Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(length_data,1)*0.05


mYc = np.array(cc)
Y = mYc.reshape(length_data,1)

#print "Y =", Y, "\n", mYc.reshape(length_data,1)

start = time.time()
# define kernel
ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.white(2)
#ker = GPy.kern.Matern52(2,ARD=True)  
end = time.time()
print end - start, " seconds to build the kernel."

start = time.time()
# create simple GP model
m = GPy.models.GPRegression(X,Y,ker)
end = time.time()
print end - start, " seconds to perform the regression"

# contrain all parameters to be positive
m.constrain_positive('')
# Add statement to plot inline
get_ipython().magic(u'pylab --no-import-all inline')
# optimize and plot
m.optimize('tnc', max_f_eval = 50)
m.plot()

print len(Y), ' x ', len(X), "Data points"
print(m)


# In[ ]:



