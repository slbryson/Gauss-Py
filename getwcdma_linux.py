#function to get wcma data.  Currently only for test.  
# Will add the capability to pass the dbase name as argument
import ConfigParser
CONFIG_FILE = 'config.ini'

def getwcdma():
	import datetime
	import pyodbc
	import numpy as np
	
	#constr =r'DSN=sqlserverdatasource;DRIVER={FreeTDS}; DATABASE=wms_kpi;uid=slbryson; Pwd=!;'
	constr =getDBConnStringConfig()
	
	listTable =['wms_kpi.dbo.test','wms_kpi.dbo.location', 'wms_kpi.dbo.carriera0830','wms_kpi.dbo.carrierb1630']

	#We need to sort the results based on cell name. 
	 
	str = 'select * from ' + listTable[0]
	 
	con = pyodbc.connect(constr)
	c = con.cursor()


	#cursor.execute("select cell from wms_kpi.dbo.test")
	how_many = c.execute(str)
	#print how_many.description, '\n',  

	 #get the results
	kk = c.fetchall()
	howmany = int(len(kk))


	#Get RRC attempts
	 
	psrrc= np.zeros((howmany,1))
	csrrc= np.zeros((howmany,1))
	cellid = np.chararray((howmany,1), itemsize=20)
	if False:
	 psrrc ={}
	 csrrc = {}
	 cellid = {}
	for rows in range(len(kk)):
		tempPS = kk[rows][3]
		psrrc[rows] = float(tempPS)
		csrrc[rows]= kk[rows][2]
		cellid[rows] = kk[rows][1]
	#print them
	if False:
		for rec in db:
			print repr(rec)
	if False:
		print psrrc
	#psrrc =psrrc.tolist()
	#csrrc = csrrc.tolist()
	#cellid = cellid.tolist()
	if False:
		print psrrc
	c.close()
	con.close()
	return cellid,psrrc

def getDBConnStringConfig():
	conStr = r''
	cnf = ConfigParser.ConfigParser()
	cnf.read(CONFIG_FILE)
	for option in cnf.items('DATABASE'):
		conStr += '%s=%s;' %  option
	return conStr.rstrip(';')

	
