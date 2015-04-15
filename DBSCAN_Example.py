#!/usr/bin/python
# Author : Wolf Rendall, 2014
# wrendall@auction.com
# Auction.com 

# This script contains the clustering implementation that takes all the geocoded user actions and returns their primary area of search
# It is an alternative version of beck_kmeans_with_weights.py, using the DBSCAN algorithm instead.

# Currently we cluser using:
# Registration ZIP
# State Notifications
# Alerts ZIP
# Bids
# Saved Searches
# PDP Views
# Vault Views
# We hope to add all searches ASAP

import sys
import numpy as np
#import datetime        # We use this for efficiency testing only
from sklearn.cluster import DBSCAN

# Define an error checking formula. This will handle missing or weird lat/lons
def checkfloat (value):
    try:
        float(value)
        return float(value)
    except ValueError:
        return "Missing Lat/Lon Value: %s" %(value)
def checkint (value):
    try:
        int(float(value))
        return int(float(value))
    except ValueError:
        return "Missing Wight Value: %s" %(value)

#t0 = datetime.datetime.utcnow()
# input comes from STDIN (standard input)
for line in sys.stdin:
    #print(line)
    try:
        # remove any leading and trailing whitespace
        line = line.strip()
        # split the line into the email address and the remainder part containing variable number of tuples
        user_id, raw_tuples = line.split(',',1)
        # the whole set of tuples is wrapped in curly braces so strip them off first
        raw_tuples = raw_tuples.strip('({})')
        tuples = raw_tuples.split('),')
        # process each tuple into an array of coords
        obs = []
        for tuple in tuples:
            user_id, action, lat, lon, wgt = tuple.strip('()').split(',',4)
            lat = checkfloat(lat)
            lon = checkfloat(lon)
            wgt = checkint(wgt)
            if type(lat) is float and type(lon) is float and type(wgt) is int:
                row = [lat, lon]
                for i in range(wgt):
                    obs.append(row)
            #else: print >> sys.stderr, (user_id,lat,lon)

        #Make that array into a numpy array of two columns
        user_actions = np.array(obs)
        if np.size(user_actions) > 3:
            #Optionally, we can standardize the variables, but in testing these results proved worse
            #trans_user_actions = StandardScaler().fit_transform(user_actions)
            db = DBSCAN(eps = 1.0, min_samples = 2).fit(user_actions)

            labels = db.labels_
            #print(labels)
            core_sample_indices = db.core_sample_indices_
            #print(core_sample_indices)
            #if all are noise, pick the first cluster
            if np.size(core_sample_indices) > 0:
                cluster_assignment = labels[core_sample_indices].tolist()
                #print(cluster_assignment)
                unique_labels = np.unique(cluster_assignment)
                #print(unique_labels)

                #find the size (member count) of each cluster
                cluster_sizes = []
                for label in unique_labels:
                    size = cluster_assignment.count(label)
                    cluster_sizes.append(size)
                #print(cluster_sizes)

                #pick the largest cluster as the winning, with the first index in case of a tie
                winning_cluster = np.argmax(cluster_sizes)
                #print(winning_cluster)
                #Get the selected members and find their averages, this will be the centroid center.
                index_range = range(len(labels))
                #print(index_range)
                winning_index = [x[1] for x in zip(labels,index_range) if x[0] == int(winning_cluster)]
                #print(winning_index)
                winning_actions = user_actions[winning_index]
                #print(winning_actions)
                centroid = np.mean(winning_actions, axis = 0)
                #Export
            else:
                centroid = [row[0], row[1]]
            print ('%s\t%s\t%s') % (user_id, centroid[0], centroid[1])
        elif 'row' in locals(): print ('%s\t%s\t%s') % (user_id, row[0], row[1])
    
    except ValueError:
        print >> sys.stderr, line
