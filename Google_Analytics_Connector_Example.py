#!/usr/bin/env python
# This python module is used to connect to Google Analytics after the user has authenticated with OAuth.
# One default dimension is provided, but it'll accept more
# This method should be called to grab aggregated metrics
# For hit-level data, see bigquery connector

import re
from google2pandas import *

def getGA(agg_level='hourly', start_date='1daysAgo', return_metadata=False, simplify=True, limit=100):

    print("Usage: getGA(agg_level <comma delimited string>, start_date <YYYY-MM-DD>, limit <integer>)")

    if agg_level == 'hourly':
        dimension = ['date', 'hour', 'medium']
    else:
        dimension = list(agg_level.split(','))



    conn = GoogleAnalyticsQuery(secrets='./ga-creds/client_secrets.json',
                                token_file_name='./ga-creds/analytics.dat')

    # Some definitions
    # ids reflects the view we wish to query with our authenticated secret. Don't change it
    # Metrics is the measure we want to summarize. Comma separated list. Change it frequently
    # Dimensions are aggregating fields.
    # start_date accepts YYYY-MM-DD or NdaysAgo format
    # More at https://developers.google.com/analytics/devguides/reporting/core/v3/reference#q_summary

    view_id = 'SCRUBBED'

    # Sess, PV, Reg, SLR QS, PL QS, SLR Sub, PL Sub
    metrics = 'ga:sessions' \
              ',ga:pageviews' \
              ',ga:goal2Completions' \
              ',ga:goal4Completions' \
              ',ga:goal9Completions' \
              ',ga:goal5Completions' \
              ',ga:goal10Completions'

    approved_channels = ['(none)', 'organic', 'cpc', 'sem', 'ppc', 'paidsearch']
    filter_string = ['ga:medium==' + channel for channel in approved_channels]



    query = {\
        'ids' : view_id,
        'metrics' : metrics,
        'dimensions' : dimension,
        'start_date' : start_date,
        'max_results' : limit,
        'filters' : ",".join(filter_string),
        'all_results' : True
    }

    df, metadata = conn.execute_query(**query)

    if simplify:
        #turn (none) medium into direct channel
        diregex = re.compile(pattern="\(none\)")
        renaming = df.replace(to_replace=diregex, value="direct")
        levels = renaming['medium'].unique()
        print("renamed channels. current levels: {0}".format(levels))

        #turn non-direct non-organic channels into paid
        to_simplify = [channel for channel in approved_channels if channel not in ['direct', 'organic']]
        regex = re.compile(pattern="|".join(to_simplify))
        simplifying = renaming.replace(to_replace=regex, value="paid")
        levels = simplifying['medium'].unique()
        print("renamed channels. current levels: {0}".format(levels))

        simplified = simplifying.groupby(by=dimension, as_index=False)[simplifying.columns.values].sum()

        simplified.columns = ['date', 'hits_hour', 'type', 'sessions', 'pageviews',
                              'registrations', 'SLR_QualStart', 'PL_QualStart', 'SLR_Submit', 'PL_Submit']

        if return_metadata:
            return simplified, metadata
            print("Got a dateframe of size {0}".format(simplified.shape))
        else:
            print("Got a dateframe of size {0}".format(simplified.shape))
            return simplified

    if return_metadata:
        print("Got a dateframe of size {0}".format(df.shape))
        return df, metadata
    else:
        print("Got a dateframe of size {0}".format(df.shape))
        return df
