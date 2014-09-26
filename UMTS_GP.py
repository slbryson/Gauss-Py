
# coding: utf-8

# In[1]:

import pylab as pb
pb.ion()
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import GPy
import random
import get_loc_linux  
import getwcdma_linux
import cell_mm_linux
import saas_write_csv
import create_dataset
import time
import saas_util  # need this for my switch statement

#get_ipython().magic(u'pylab --no-import-all inline')



# In[27]:

#Test the UMTS GP processing on Dummy Data
# Create some dummy location data
#Match an index and perform the operations
# Let's retrieve some real location data
# Get real data
# Call getwcdma.py which selects the data.  Pick a table 0 to N

listTable =[]
smean =[]
ssum = []
ind =[]
normalize = True
sample_fraction = 4
resolution = int(150)  #This is the Grid resolution for sampling
max_iters = 5000
eval_max = 50
var = .9
dim = 2
d =.08
choice = 4

listTable =['wms_kpi.dbo.test', 'wms_kpi.dbo.carriera0830','wms_kpi.dbo.carrierb1630']
table_ = 0


# In[28]:

#######################################
cellid, lat, lon = get_loc_linux.get_loc()
loc = zip(cellid,lat,lon)
print "How long is our data set?  ", len(loc)
#######################################


# In[29]:

start = time.time()
cellid, dbdata2 = getwcdma_linux.getwcdma(listTable[table_])
end = time.time()
#Create container for cellid and data
dbdata = zip(cellid,dbdata2)
print end - start, " seconds to get WCDMA data from the dbase and ",len(cellid), 'records.'


# In[30]:

#Call of cell_mm_linux
#This function will take each of the data samples and calculate a mean over the entire range.
# Currently the inputs are segmented into hours from the choice of table being read.
#Inputs needed are the location data and dbdata. 
# Get a mean and a list of the matching cells

start = time.time()
# Note the function call will return a list indices from the location data matching cells
# found in the data set and the mean for the hour (averaged over all the samples)
# We time this function because it can be optimized later perhaps with Pyspark or cython
ind, smean = cell_mm_linux.cell_mm(loc,dbdata)
end = time.time()
print end - start, " seconds to calculate the mean and correlate cells"


start = time.time()
mX, cc, ssum, Xsample, Ysample, xTest,yTest = create_dataset.gp_saas_norm(ind,loc,smean,normalize,sample_fraction)
end = time.time()
print 'Took ', end-start, 'seconds to build the dataset'
if False:
    print mX[8], len(mX), cc[8]
    print 'Shape =',Xsample.shape, Ysample.shape


# In[31]:

############################################################
# define kernel  This is where a lot more variation/experiments are needed
# At a bare minimum we need
# ker or rbf as kernels 
# m as the model
# varaince  var
# lengthscale d
############################################################
 
for case in saas_util.switch(choice):
    if case(1):
        ker = GPy.kern.Matern52(dim,ARD=True) + GPy.kern.white(2)
        ker['.*var']= var
        ker['.*lengthscale']= d
        m = GPy.models.GPRegression(Xsample,Ysample,ker)
        m.constrain_positive('')
        start = time.time()
        # optimize and plot
        m.optimize('scg', max_iters = max_iters, messages =0)
        break
    if case(2):
        #Let's use the Sparse GP regression with an exponential kernel
        rbf =GPy.kern.exponential(dim,var,ARD=True)
        rbf = rbf *rbf
        rbf['.*var']= var
        rbf['.*lengthscale']= d
        m = GPy.models.SparseGPRegression(Xsample, Ysample, kernel=rbf, num_inducing=len(Ysample/2))
        m.checkgrad(verbose=0)
        start = time.time()
        # optimize      
        m.optimize('tnc', max_f_eval = eval_max)
        break
    if case(3):
        rbf =GPy.kern.rbf(dim)
        rbf = rbf *rbf
        rbf['.*var']= var
        rbf['.*lengthscale']= d
        m = GPy.models.SparseGPRegression(Xsample, Ysample, kernel=rbf, num_inducing=len(Ysample/2))
        m.checkgrad(verbose=0)
        start = time.time()
        # optimize      
        m.optimize('tnc', max_f_eval = eval_max)
        break
    if case(4):
        #Let's use the Sparse GP regression with an exponential kernel
        rbf =GPy.kern.exponential(dim,var,ARD=True)
        rbf = rbf *rbf
        rbf['.*var']= var
        rbf['.*lengthscale']= d
        m = GPy.models.SparseGPRegression(Xsample, Ysample, kernel=rbf, num_inducing=len(Ysample/2))
        m.checkgrad(verbose=0)
        start = time.time()
        # optimize      
        m.optimize('scg', max_iters = eval_max)
        break
    if case():
        print 'we found nothing'
end = time.time()
print end - start, " seconds to optimize for ",max_iters," max iterations"

m.plot(plot_raw=True,levels=100)


# In[26]:

# Write out m4 data to see how well the modified Kernel and regression performed
mPredict = m.predict(mX)[0]
 
if True:
    result=saas_write_csv.write_csv('sparseGP_ker_original.csv',mX[:,0],mX[:,1],mPredict*ssum)

# Create the test set. note xTest are all the samples NOT used in the random selection
# of sample data used to train the model.

X=np.copy(xTest)
min_a = np.min(X[:,0])
max_a = np.max(X[:,0])
min_b = np.min(X[:,1])
max_b = np.max(X[:,1])
#Grid for input later into the mean function, derived from input coordinates
X1, X2 = np.mgrid[min_a:max_a:150j,min_b:max_b:150j]
Y1  = np.reshape(X1,(len(X1)*len(X1),1))
Y2  = np.reshape(X2,(len(X2)*len(X2),1))
Y   = np.column_stack((Y1,Y2))


# In[52]:

# In this section we look at self-generate contour plots from exposed output variables 
# versus relying on the GPy plot module

print 'Resolution = ', resolution
# First redo prediction based on grid, but don't overwrite prediction based on actual test values
#Note Y = grid representation from X1, X2 which are grid points from the test vector.
###################################
xGridPredict, _, _, _ = m.predict(Y)
m_d = xGridPredict.reshape(resolution,resolution)
#m_d = copy(xPredict)
print  'New Reshaped ',m_d.shape

#################################
db = pb.contourf(X1,X2,m_d, cmap=pb.cm.coolwarm, vmin=abs(m_d).min(), vmax=abs(m_d).max())
pb.title('Contour plot of prediction of all points in the grid')
################################
db4 = pb.scatter(xTest[:,0],xTest[:,1])
if True:
    #write out the prediction at every point in the grid
    result = saas_write_csv.write_csv('grid_predictor.csv',(Y1),Y2,xGridPredict*ssum)


# In[ ]:



