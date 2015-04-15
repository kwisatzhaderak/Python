#!/usr/bin/python


#======================Change Long==========================================================================================#
#   Date                    By                          Description															#
#   2014-06-17              Wolf Rendall                Initial Release	-- Work in Progress Still	                        #
#																															#   
#===========================================================================================================================#


import os, sys, getopt, time, shutil, stat, re, fnmatch, paramiko, gzip
from datetime import datetime
from subprocess import call
from os import listdir
from os.path import isfile, join


#Assign Dates and ScriptName
start = time.time()
startDateTime = datetime.fromtimestamp(start).strftime("%Y%m%d%H%M%S")
scriptname = os.path.basename(__file__)

user = os.getenv('USER', 'Unknown') #getpass.getuser()
n = int(os.getenv('N', 3)) #Number of runs; default 3
start = time.time()
startDateTime = datetime.fromtimestamp(start).strftime("%Y%m%d%H%M%S")

#/*=================================
#=            Variables            =s
#=================================*/
#


raw_database = 'raw'
raw_table = 'zestimate'

ftp_file_prefix = 'AuctionZestimates_'
ftp_file_ext = '.csv.gz'
zestimate_pattern = ftp_file_prefix + '*' + ftp_file_ext
newdir_base = '/tmp/zestimate/extracted/'
archivedir_base = '/tmp/zestimate/done/'
hdfs_dir_base = '/apps/raw/zestimate/'
hdfs_server = 'hdfs://hmas01lax01us:8020'
hdfs_zestimate_dir = hdfs_server + hdfs_dir_base

cmd_hadoop_mkdir = 'hadoop fs -mkdir '
hit_data_file = 'AuctionZestimates'
hcatCreatePartition = 'hcat -e "\n'


#/*=================================
#=    Get Files from FTP Client    =s
#=================================*/
#
import paramiko

host = "fdep.auction.com"                    #hard-coded
port = 22
transport = paramiko.Transport((host, port))

password = "password"                #hard-coded
username = "username"               #hard-coded
transport.connect(username = username, password = password)

sftp = paramiko.SFTPClient.from_transport(transport)

import sys
path = './' + sys.argv[1]    #hard-coded
localpath = sys.argv[1]
sftp.put(localpath, path)

sftp.close()
transport.close()
print 'Upload done.'


## List of the files

files = []
zestimate_files = [ files for files in listdir(ftp_directory) if isfile(join(ftp_directory,files)) ]
zestimate_files.sort()

for zestimate_file in zestimate_files:

	if fnmatch.fnmatch(zestimate_file, zestimate_pattern ):
		print zestimate_file
		## First, need to split the filname into its components
		parts = re.split('_', zestimate_file)
		date = parts[1]
		date_parts = re.split('-',date)
		year = date_parts[0]
		month = date_parts[1]
		day = date_parts[2]

		
		newdir = newdir_base + year + month + day
		archivedir = archivedir_base + year + month + day
		hdfs_dir = hdfs_dir_base + year + month + day
		# Create output directory
		try:
			os.stat(newdir)
		except:
			os.makedirs(newdir)
			os.chmod(newdir, 0777)
			call(cmd_hadoop_mkdir + hdfs_dir, shell=True)
			print 'Will execute ' + cmd_hadoop_mkdir + hdfs_dir
		extract_cmd =  'tar xvfO ' + ftp_directory + zestimate_file + ' | gzip > ' + newdir + '/' + zestimate_file
		hdfs_copy = 'hadoop fs -copyFromLocal '  + newdir + '/' + zestimate_file
		hdfs_chmod = 'hadoop fs -chmod -R 775 ' + hdfs_dir_base + year + month + day
		print 'Extraction command:' + extract_cmd
		print 'Will copy to HDFS: ' + hdfs_copy
		call(extract_cmd, shell=True)
		call(hdfs_copy, shell=True)
		##Create partitions
		hcatCreatePartition = hcatCreatePartition + 'use ' + raw_database + '; alter table ' + raw_table + ' add IF NOT EXISTS partition(datestamp=\''+ year+month+day +'\') LOCATION \'' + hdfs_zestimate_dir + year + month + day + '\';\n'

		try:
			os.stat(archivedir)
		except:
			os.makedirs(archivedir)
			os.chmod(archivedir, 0777)

			move_cmd = 'mv ' + newdir +'/'+ hit_data_file_gz + ' ' + archivedir + '/' + hit_data_file_gz
			move_cmd2 = 'mv ' + ftp_directory + zestimate_file + ' ' + newdir + '/'
			call(move_cmd, shell=True)
			call(move_cmd2, shell=True)
			#print move_cmd
			#print move_cmd2

hcatCreatePartition = hcatCreatePartition + '"'
print hcatCreatePartition
call (hcatCreatePartition, shell=True)

#
