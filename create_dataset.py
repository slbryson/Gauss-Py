# This routine will compact the front end data processing so it can be recalled 
# in a loop to process multiple tables. 
# The routine takes a list of indices that indicate where cell locations were matched
# to data in an array that contains data for each point and then normalizes the data
#  9/25/2014  Sidney Bryson
#####################
def gp_saas_norm(ind,loc,smean,normalize,sample_fraction):
	import numpy as np
	from scipy import linalg
	from scipy import optimize
	from scipy import misc
	import random

	if normalize is "":
		normalize = True
	# Now map the lat lon and predictor data
	mXa = map(lambda row: loc[row][1],ind[0])
	mXb = map(lambda row: loc[row][2],ind[0])
	cd1 = map(lambda row: smean[row],ind[0])

	# The GPs tend to work better on normalized data vs raw inputs.
	if normalize:
	    # Need to debug this later
	    #cc = cd1/np.linalg.norm(cd1, ord = inf)
	    #cc = cd1/sum(cd1,axis=0)
	    cc = cd1/sum(cd1)
	else:
	    cc =cd1
    
	# We could also normalize the geocoordinates, but if we do we need to 
	#remember the values and
	# relative array locations to generate a map later.
	a = np.array(mXa)
	b = np.array(mXb)
	if normalize:
	    mXa =a
	    mXb =b
	else:
	    mXa = a/np.linalg.norm(a, ord = 0)
	    mXa = a/sum(a,axis=0)  # Untested 9/24/2014
	    mXb = b/np.linalg.norm(b, ord = 0)
	    mXb = b/sum(b,axis=0) # Untested 9/24/2014
	#Resulting input coordinates from training set.
	mX = np.column_stack((mXa,mXb))
	#cc = the mean of the observed data.

	if False:
	    print len(mX), mX
	# Now we have our input data for the GP.
	print "Input data for GPs are processed"
	 	
	##########################################

	# Given ta regression problem, providing ALL the inputs as training sets is 
	# sometimes performed.  Option here to take random sample for the training set
	# or the entire sample.

	#Decide a fraction of the data to select
	if sample_fraction == "":
		sample_fraction = 4
	
	sample_size = int(len(mX)/sample_fraction)
	 
	data = zip(mX,cc)
	random.shuffle(data)

	#Unzip the random sample
	coords, predictor = zip(*data)
	Xsample = np.array(coords[:sample_size])
	xTest =   np.array(coords[sample_size:])
	
	Ysample = np.array(predictor[:sample_size])
	Ysample = Ysample.reshape(sample_size,1)
	
	yTest =   np.array(predictor[sample_size:])
	yTest = yTest.reshape(len(yTest+1),1)
	ssum = sum(cd1)
	
	return mX, cc, ssum, Xsample, Ysample, xTest, yTest
