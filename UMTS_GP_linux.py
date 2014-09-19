
# coding: utf-8

# In[30]:

import pylab as pb
pb.ion()
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import GPy
import random
import get_loc_linux  
import getwcdma_linux
import time
#%pylab --no-import-all inline


# In[2]:

#Test the UMTS GP processing on Dummy Data
# Create some dummy location data
#Match an index and perform the operations
REAL_DATA =False
######
# Let's retrieve some real location data

name2, lat, lon = get_loc_linux.get_loc()

# Following Fragment is leftover debug code for issues with data types.
if False:
    lat = lat.tolist()
    lon = lon.tolist()
    name2 = name2.tolist()
#Creates a container for the information so that we can randomly manipulate complete rows
loc = zip(name2,lat,lon)
if REAL_DATA:
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

    #now create some dummy data For test purposes. This code is NULL if real data is retrieved.
    #############################
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
    #############################
# After testing, these randomly generating data sets are not used and currently the data sets are
# selected in the external module getwcdma_linux()
#############################
print "How long is our data set?  ", "Location=", len(loc), name2[81]


# In[3]:

# Get real data
# Call getwcdma.py
# The tables are "hardcoded" currently and have to be selected within the function call itself. 
# This can be easily changed so that the user selects which data set (an hour ) to look at.
# Separate scripts on the dbase server have already taken data from over a three month period and
# placed into new tables separated by hour and sector.
# Future modifications can be used to segment with these function calls, but currently Python 
# dbase manipulations from remote calls are much slower than direct database 
#function calls on the server
start = time.time()
 
cellid, dbdata2 = getwcdma_linux.getwcdma()
end = time.time()
print end - start, " seconds to get WCDMA data from the dbase"

#Create container for cellid and data
dbdata = zip(cellid,dbdata2)
if False:
    # Test a few sample points.  Sometimes when libraries change e.g. Numpy data type manipulation 
    #corrupted.
    print len(dbdata), dbdata2[4], cellid[42]
    print dbdata2[4] * dbdata2[2], dbdata[4][0], loc[4][0]
 
 


# In[5]:

#Call of cell_mm_linux
#This function will take each of the data samples and calculate a mean over the entire range.
# Currently the inputs are segmented into hours from the choice of table being read.
#Inputs needed are the location data and dbdata. 
import cell_mm_linux  #need to change to _linux
import time

smean =[]
ind =[]

# Get a mean and a list of the matching cells
start = time.time()
# Note the function call will return a list indices from the location data matching cells
# found in the data set and the mean for the hour (averaged over all the samples)
# We time this function because it can be optimized later perhaps with Pyspark or cython
ind, smean = cell_mm_linux.cell_mm(loc,dbdata)
end = time.time()
print end - start, " seconds to calculate the mean and correlate cells"

# Test output
# The index rows indicate where the location cell id matched data from the date, so the rows will
# match the correct location indices.

# Now map the lat lon and predictor data
mXa = map(lambda row: loc[row][1],ind[0])
mXb = map(lambda row: loc[row][2],ind[0])
cd1 = map(lambda row: smean[row],ind[0])

# The GPs tend to work better on normalized data vs raw inputs.
if True:
    cc = cd1/np.linalg.norm(cd1)
else:
    cc =cd1
    
# We could also normalize the geocoordinates, but if we do we need to remember the values and
# relative array locations to generate a map later.
a = np.array(mXa)
b = np.array(mXb)
if False:
   
    mXa = a/np.linalg.norm(a)
   
    mXb = b/np.linalg.norm(b)
else:
    mXa =a
    mXb =b
#Resulting input coordinates from training set.
mX = np.column_stack((mXa,mXb))
#cc = the mean of the observed data.
length_data = len(cc)

if False:
    print len(mX), mX
# Now we have our input data for the GP.
print "Input data for GPs are processed"


# In[27]:

# sample inputs and outputs

#Obselete code
##########################################
# Don't need to copy mX here X =copy(mX)
# mYc = np.array(cc)
# Y = mYc.reshape(length_data,1)
##########################################

# Given ta regression problem, providing ALL the inputs as training sets is 
# sometimes performed.  Option here to take random sample for the training set
# or the entire sample.

#Decide a fraction of the data to select
sample_fraction = 4
if True:
    sample_size = int(len(mX)/sample_fraction)
    #data = random.sample(zip(mX,cc),sample_size)
    data = zip(mX,cc)
    random.shuffle(data)
     
    #Unzip the random sample
    coords, predictor = zip(*data)
    Xsample = np.array(coords[:sample_size])
    xTest =   np.array(coords[sample_size:])
    if False:
        print 'sample and test size = ', Xsample.shape, xTest.shape

    Ysample = np.array(predictor[:sample_size])
    Ysample = Ysample.reshape(sample_size,1)
    if False:
        print 'Predictor =', shape(Ysample), sample_size
    
    yTest =   np.array(predictor[sample_size:])
    yTest = yTest.reshape(len(yTest+1),1)
    if False:
        print 'Rest of Predictor =', yTest.shape, len(yTest)
        print 'Shape and type of predictor =',type(Ysample),Ysample.shape, Ysample[5]
 
else:
    Xsample =copy(X)
    Ysample =copy(Y)
if True:
    print type(Xsample), type(Ysample),'Shape =',Xsample.shape, Ysample.shape


# In[11]:

start = time.time()
# define kernel  This is where a lot more variation/experiments are needed
ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.white(2)
ker = GPy.kern.Matern52(2,ARD=True)  
end = time.time()
print end - start, " seconds to build the kernel."

start = time.time()
# create simple GP model
m = GPy.models.GPRegression(Xsample,Ysample,ker)
end = time.time()
print end - start, " seconds to perform the regression"

# contrain all parameters to be positive
m.constrain_positive('')

# optimize and plot
m.optimize('tnc', max_f_eval = 200)
# The plot function for a GPy model data type is built in but HIDES the resulting prediction variables
m.plot(plot_raw=True,levels=100)

print len(Ysample), ' x ', len(Xsample), "Data points"
print(m)


# In[29]:

# For experimentation here is a different kernel and different regression routine using 
# Radial Basis Functions on the same data
rbf = GPy.kern.rbf(2)
# create simple GP Model
m = GPy.models.SparseGPRegression(Xsample, Ysample, kernel=rbf, num_inducing=8)
m.checkgrad(verbose=1)

m.optimize('tnc', messages=1, max_iters=25)
m.plot()


# In[31]:

#Now to evaluate the data
# Create the test set. note xTest are all the samples NOT used in the random selection
# of sample data used to train the model.

X=np.copy(xTest)
min_a = np.min(X[:,0])
max_a = np.max(X[:,0])
min_b = np.min(X[:,1])
max_b = np.max(X[:,1])
resolution = int(25)

#Grid for input later into the mean function, derived from input coordinates
X1, X2 = np.mgrid[min_a:max_a:25j,min_b:max_b:25j]

# Grid
kb = pb.plot(X1,X2,'b+')
#The sample points for the GP
kb = pb.plot(Xsample[:,0],Xsample[:,1],'gd',markersize=12)
# Test  points for the GP
kb = pb.plot(xTest[:,0],xTest[:,1],'mo')
#
print 'Shape of X1, X2 which is the grid size =',X1.shape,X2.shape
print '\nSample size vs Full size', Xsample.shape,xTest.shape


# In[29]:

# Create input vector from grid bounded by geocoordinate inputs
Y1 = np.reshape(X1,(len(X1)*len(X1),1))
Y2  = np.reshape(X2,(len(X2)*len(X2),1))
Y = np.column_stack((Y1,Y2))
# Generate the prediction from the test variables
xPredict, _, _, _ = m.predict(Y)
# A second method for the prediction..currently showing no considerable difference 
xPredict, _= m._raw_predict(Y, which_parts='ALL')
print 'Shape of Y which is the predictor grid input =', Y.shape, Y1.shape, Y2.shape, xPredict.shape


# In[23]:

#An attemp to plot just the results and values
colors =  [float(10*item/255.) for item in xPredict]
 
db = plt.scatter(xTest[:,0],xTest[:,1], cmap=plt.cm.jet,marker='+', s=55, linewidths=2)
db = plt.scatter(Y1[:,0],Y2[:,0],c=xPredict,cmap=plt.cm.jet, marker='+', s=55, linewidths=2)
 


# In[32]:

# In this section we look at self-generate contour plots from exposed output variables 
# versus relying on the GPy plot module
print len(xPredict), 'Length of Y1', len(Y1), 'Length Y[0,1]',len(Y[:,1])
m_d = xPredict.reshape(resolution,resolution)
#m_d = copy(xPredict)
print  'New Reshaped ',m_d.shape

#db = pb.contourf(X1,X2,m_d,175)
db = pb.contourf(X1,X2,m_d, cmap=pb.cm.jet, vmin=abs(m_d).min(), vmax=abs(m_d).max())

#db3 = pb.scatter(X1,X2)
db4 = pb.scatter(xTest[:,0],xTest[:,1])


# In[ ]:



