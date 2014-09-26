##python
## This module reads from the UMTS database in SaaS and returns a cell name, latitude and longitude
# ConfigParser allows the user to set up their own access creditials in a config file
#  For this function since the location data doesn't change for the cells 
# then we don't need offer a selection.
import datetime
import pyodbc
import numpy as np
import ConfigParser
#Constants
CONFIG_FILE = 'config.ini'
def get_loc ():
	
	#constr =r'DSN=sqlserverdatasource;DRIVER={FreeTDS}; DATABASE=wms_kpi;uid=slbryson; Pwd=!;'
	constr = getDBConnStringConfig()
	
	listTable =['null','wms_kpi.dbo.location']

	#We need to sort the results based on cell name. 
	print listTable[1]
	str = 'select * from ' + listTable[1]
	print str
	con = pyodbc.connect(constr)
	c = con.cursor()


	#cursor.execute("select cell from wms_kpi.dbo.test")
	how_many = c.execute(str)
	#print how_many.description, '\n',  

	 #get the results
	kk = c.fetchall()
	howmany = int(len(kk))

	#print howmany
	if False:
	 for rec in db:
	    print repr(rec)
    

	lat= np.zeros((howmany,1))
	lon= np.zeros((howmany,1))
	name = np.chararray((howmany,1), itemsize=50)
	if False:
	 lat = {}
	 lon ={}
	 name ={}

	for rows in range(int(len(kk))):
	    lat[rows] = kk[rows][0]
	    lon[rows] = kk[rows][1]
	    name[rows] = kk[rows][4]
	    #print  rec.latitude, rec.longitude
	    #print lat
	# Debug
	if False:
	    for rows in range(65,80):
	        print lat[rows], name[rows],lon[rows]
	c.close()
	return name, lat, lon

def getDBConnStringConfig():
	conStr = r''
	cnf = ConfigParser.ConfigParser()
	cnf.read(CONFIG_FILE)
	for option in cnf.items('DATABASE'):
		conStr += '%s=%s;' %  option
	return conStr.rstrip(';')

