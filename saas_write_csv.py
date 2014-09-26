# This file will eventually contain all the utilities for this application
# Sidney Bryson 9/25/2014
################################
def write_csv(filename,lat,lon,buckets):
    # This utility write lat, lon and predictor value to a csv file. 
    # Arguments include a filename
    import os
    import csv
 
    #############################################
 
    # Get length of data set to write too bad if the arrays are not equal length
    nitems = len(buckets)
    #print nitems, len(lat), len(lon)
    
    #############################################
    with open(filename,'w') as csvfile:
        output = csv.writer(csvfile, delimiter="\t")
        output.writerow(['lat', 'lon', 'value'])
         
        for row in range(0,nitems-1):
            lat_el = lat[row]
            lon_el = lon[row]
            value = buckets[row]
            output.writerow([float(lat_el), float(lon_el), float(value)])
    return "Done writing"
