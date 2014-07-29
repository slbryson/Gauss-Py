#Now create a short function or module in python that takes input of location matrix and input
#of a data matrix, matches location by cell name then computes the average of all matched
#cells and returns the mean and the matched location of the cells.
# loc should have columns nameid, lat, long; dbdata should have nameid, data1,data2
# we are interested in the data2 or third column of the data

def cell_mm(loc,dbdata):
    # In this case mm means match and mean.  This is a specific module used to match UMTS data
    # with UMTS location data and compute the mean.
    import numpy as np
     
    # Critical that try_this is initialized to be able to do the assignment correctly.
    s =(len(loc), len(dbdata))
    try_this = np.zeros(s)
    # We will match data with cell location. Note the non-zero elements and corresponding indices
    # denote the location of matches between cells. 
    # Note this will extract data that matches cellid in both sets and form a new matrix
    # is is assumed the matching index is in position 1 and the data is in position 2.
    for row in range(len(loc)):
        for col in range(len(dbdata)):
            #do something;
            if loc[row][0] == dbdata[col][0]:
             try_this[row][col] =dbdata[col][1]
    # Now get the indices
    i,j=try_this.nonzero() # this statement plus the zip below will save the non-zero indices
    c = zip(i,j)
    #print "zipped coordinates\n",c, "\n Shape of try_this\n", try_this.shape
    ind = zip(*c)
    # We will compute the mean of the matches
    #in the event we have coordinates of the match, only the rows correspond with the match between cell id in the location matrix.
    # So sub-sample the location matrix with matches.

    # lets try a better way to find the mean of the columns
    my_mean = try_this.sum(1) / (try_this != 0).sum(1)
    # We will return the mean vector and corresponding location of the matches
    
    return ind,my_mean