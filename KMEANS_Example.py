#!/usr/bin/python
# Author : Wolf Rendall, 2014
# wrendall@auction.com
# Auction.com 

# This script contains the clustering implementation that takes all the geocoded user actions and returns their primary area of search
# This is meant to run after beck_user_data.pig and before beck_asset_sort.pig

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
from sklearn.cluster import KMeans

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
        #print  '%s \t %s \n %s' %(user_id, np.size(user_actions), user_actions)
        #Make sure we even need to cluster by requiring 2 or more lat/lon coords
        if np.size(user_actions) > 3:
            number_of_clusters = int(np.floor(min(max(np.sqrt(np.size(user_actions) *0.5),2), 5)))
            #print " KMeans with %s clusters" % (number_of_clusters)
            #Important note, it is possible to employ parallelization with the n_jobs arg
            kmeans = KMeans(init='k-means++', n_clusters=number_of_clusters, n_init=10, tol = 0.00001, verbose = 0, n_jobs = 1)
            results = kmeans.fit(user_actions)
            group_labels = results.labels_
            centroids = results.cluster_centers_
            unique_labels = np.unique(group_labels)

            #print(group_labels)
            #print(centroids)

            ##Get the Einstein Sum of all points in the centroids
            #distances = []
            #for centroid in centroids:
            #coded_actions = np.array(zip(user_actions, labels))
            within_centroid_distances = []
            centroid_size = []
            for label in unique_labels:
                distances = []
                my_centroid = centroids[label]
                members = user_actions[group_labels==label]
                member_count = np.size(members)/2
                centroid_size.append(member_count)
                for member in members:
                    distance = np.sqrt(np.square(member[0]-my_centroid[0])+np.square(member[1]-my_centroid[1]))
                    distances.append(distance)
                    #print('my_centroid:\t%s\n member: \t%s\n distance:\t%s') %(my_centroid, member, distance)
                within_centroid_distances.append(np.mean(distances))
            
            #print(within_centroid_distances)
            smallest_cluster_index = np.argmin(centroid_size)
            largest_cluster_index = np.argmax(centroid_size)

            #Remove the smallest cluster from consideration
            if len(within_centroid_distances) > 11:
                smallest_cluster_index = np.argmin(centroid_size)
                smallest_cluster_value = centroids[smallest_cluster_index]
                smallest_cluster_distance = within_centroid_distances[smallest_cluster_index]
                centroids = [x for x in centroids if (x[0]+x[1]) != (smallest_cluster_value[0]+smallest_cluster_value[1])]
                within_centroid_distances = [x for x in within_centroid_distances if x != smallest_cluster_distance]

            #Find the best cluster of the remaining clusters:
            best_cluster_index = np.argmin(within_centroid_distances)
            final_center = centroids[best_cluster_index]

            final_center_size = centroid_size[best_cluster_index]
            largest_cluster_index = np.argmax(centroid_size)
            largest_center_size = centroid_size[largest_cluster_index]
            largest_center = centroids[largest_cluster_index]

            #print ('User: %s \nCentroid: %s \nBased on %s geocodable actions') % (user_id, final_center, np.size(user_actions)/2)
            #note, the sorting pig script is expecting a file named beck_centroids.tsv
            if largest_center_size < 200:
                print ('%s\t%s\t%s') % (user_id, final_center[0], final_center[1])
            else:    
                print ('%s\t%s\t%s') % (user_id, largest_center[0], largest_center[1])
        #If not enough lat/lon coords, just print the registration geocode
        elif 'row' in locals(): print ('%s\t%s\t%s') % (user_id, row[0], row[1])
        #else: print ('User: %s \nCentroid: %s \nBased on %s geocodable actions') % (user_id,'Insufficient Data', np.size(user_actions)/2)
    #print 'finished in %s' % (datetime.datetime.utcnow() - t0)
    except ValueError:
        print >> sys.stderr, line
