#!/usr/bin/env python
# This python module is used to estimate individual TV and Radio Broadcasts' impact on business-critical metrics
# We have to estimate indirectly since there is no tracking link between TV and Radio and online behavior
# The final outputs are per-program impacts. To analyze the super group, for instance Sean Hannity, use an average
# The module uses Elastic Net Regression to implement a time series prediction of traffic for a period from trends
# Traffic above the baseline prediction for any period is credited to an unaddressable campaign during the period
# For future use, use cross validation chain to arrive at optimal alpha. For now, select a middle-of-the-road ridge regression

import pandas as pd
import numpy as np
#import os
#import time
import datetime
import math
from sklearn import linear_model
from sklearn import metrics
import re
import scipy.stats
from sklearn.preprocessing import PolynomialFeatures
from getGABigQueryData import getGAData
from googleDocsConnector import getScheduleData
from getLoanApps import getAppData
from GAReader import getGA
import itertools

#
# Import direct, orgsearch, and paidsearch
#

#Set some Hyperparameters
SinusoidModel = False

shows = [
    'Sean Hannity - 1st', 'Sean Hannity - 2nd', 'Sean Hannity - BONUS',
    'Sean Hannity - MM', 'Sean Hannity - Encore', 'Dan Patrick',
    'Dan Patrick - BONUS', 'Dan Patrick - AN', 'Colin Cowherd',
    'Colin Cowherd - BONUS', 'Rich Eisen', 'Rich Eisen - BONUS',
    'Jay Mohr - BONUS 1', 'Jay Mohr - BONUS 2', 'Jay Mohr - BONUS 3',
    'Big Boy', 'Ben Maller - BONUS 1', 'Ben Maller - BONUS 2',
    'Ben Maller - BONUS 3', 'Ben Maller - BONUS 4',
    'Ben Maller - BONUS 5', 'Ben Maller - BONUS 6', 'Rush Limbaugh',
    'Brooke & Jubal', 'Bill Cunningham', 'America Now',
    'Ground Zero with Clyde Lewis', 'Metropolitan News/Talk Network',
    'A&E', 'Bravo', 'CNN', 'DIRECTV Live Sports', 'DIY', 'Esquire',
    'E!', 'ESPN', 'ESPN University', 'Fox News Channel (FNC)', 'FYI',
    'Headline News', 'ID', 'MSNBC', 'NFL Network', 'SEC Network', 'TBS',
    'TLC', 'USA', 'AT&T NFL ST - Monday Night', 'AT&T NFL ST - In-Game',
    'AT&T NFL ST - Pre-Kick', 'AT&T NFL ST - Halftime',
    'AT&T NFL ST - Interactive Sponsorship', 'CBS - NFL Football',
    'CBS - NFL Today', 'CBS - NFL Post Game',
    'CBS - NFL Regional Post Gun', 'AT&T - College Football - ESPN',
    'AT&T - College Football - ESPN2', 'CBS - College Football',
    'TNT - NBA Playoffs', 'ESPN - NBA Playoffs', 'ESPN SportsCenter',
    'Addressable TV', 'NBCSN - Olympics :30', 'NBCSN - Olympics :60',
    'USA - Olympics :30', 'USA - Olympics :60', 'MSNBC - Olympics :30',
    'MSNBC - Olympics :60', 'CNBC - Olympics :30',
    'CNBC - Olympics :60', 'Golf Channel - Olympics :30',
    'Golf Channel - Olympics :60', 'Bravo - Olympics :30',
    'Bravo - Olympics :60', 'CBS NFL', 'Today Show', 'Comedy',
    'Discovery Science', 'FX Movie Channel', 'FXX',
    'Independent Film Channel', 'Investigation Discovery', 'Ovation',
    'PBS Sprout', 'Sundance Channel', 'MTV Classic', 'MTV Live', 'MTV2',
    'TNT']

traffic_data = getGA(agg_level='hourly', start_date='2016-02-01', return_metadata=False, simplify=True, limit=24500)

schedule_data = getScheduleData()

schedule_data['date_str'] = schedule_data['broadcast_date'].apply(lambda x: datetime.datetime.strftime(x, "%Y-%m-%d"))

# Note, due to data irregularities From 4/10 - 5/14, we will replace them with the following 35 days for PL
traffic_data['true_date'] = pd.to_datetime(traffic_data['date'])
traffic_data['date'] = traffic_data['true_date'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))

repl = traffic_data[(traffic_data['true_date'] > datetime.datetime(2016, 5, 14)) & (traffic_data['true_date'] < datetime.datetime(2016, 6, 19))]

repl_cols = repl[['true_date', 'hits_hour', 'type', 'PL_Submit']]
repl_cols['date_to_replace'] = pd.Series(repl_cols['true_date'] - datetime.timedelta(days=35)).apply(lambda d: datetime.datetime.strftime(d, "%Y-%m-%d"))

repl_1 = pd.merge(left=traffic_data,
                  right=repl_cols,
                  left_on=['date', 'hits_hour', 'type'],
                  right_on=['date_to_replace', 'hits_hour', 'type'],
                  how='left')

repl_1['PL_Submit'] = repl_1['PL_Submit_y'].fillna(repl_1['PL_Submit_x'])

traffic_data_substituted = repl_1[['type', 'date', 'hits_hour', 'sessions',
       'registrations', 'SLR_QualStart', 'SLR_Submit', 'PL_QualStart', 'PL_Submit']]

traffic_data_substituted.loc[:,'hits_hour'] = traffic_data_substituted['hits_hour'].apply(lambda x: int(x))

#
# Format Data
#
print ("data acquired, formatting")

integrated_data = pd.merge(left=traffic_data_substituted,
                           right=schedule_data,
                           left_on=['date', 'hits_hour'],
                           right_on=['date_str', 'broadcast_hour'],
                           sort=False,
                           how='left'
                           )

#
#  Fill in broadcast duration in times when there was no broadcast
#
def filler(float):
    if math.isnan(float):
        return 0.0
    else:
        return float

to_fill = [column for column in schedule_data.columns.values if column not in ['broadcast_date','broadcast_hour', 'date_str']]
for column in to_fill:
    print("filling up {0}\tcurrent type: {1}".format(column, integrated_data[column][0]))
    integrated_data.loc[:, column] = integrated_data[column].apply(filler)
    #integrated_data.loc[:, 'tv_duration_secs'] = integrated_data['tv_duration_secs'].apply(filler)

integrated_data.drop(['broadcast_date','broadcast_hour','date_str'], axis=1, inplace=True)

# ENFORCE A DATE RANGE
integrated_data =integrated_data[integrated_data['date'].apply(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d")
                                                               ) < datetime.date(2016, 10, 1)]

def fit_formula(x_vars=None, y=None):
    model = linear_model.ElasticNet(alpha=0.5,
                                    l1_ratio=0.3,
                                    fit_intercept=True,
                                    normalize=False,
                                    positive=True)

    if x_vars is None:
        x_vars = input_df

    return model.fit(X=x_vars, y=y)

def model_it(mode, training_fraction, polynomial_degree, interactions_only):
    base_set = integrated_data[integrated_data.type == mode].sort_values(by=['date', 'hits_hour'],ascending=True)
    base_set['row_num'] = pd.Series(range(0, base_set.shape[0]), index=base_set.index)

    base_set['cumsum_7_tv_dur'] = pd.Series(base_set['tv_duration_secs'].rolling(window=168, center=False).sum())
    base_set['cumsum_7_radio_dur'] = pd.Series(base_set['radio_duration_secs'].rolling(window=168, center=False).sum())

    base_set['cumsum_14_tv_dur'] = pd.Series(base_set['tv_duration_secs'].rolling(window=2*168, center=False).sum())
    base_set['cumsum_14_radio_dur'] = pd.Series(base_set['radio_duration_secs'].rolling(window=2*168, center=False).sum())

    name_list = list(base_set.columns.values.tolist())

    lag_names = ['sessions_lag', 'registrations_lag', 'PL_QualStart_lag', 'PL_Submit_lag', 'SLR_QualStart_lag',
                 'SLR_Submit_lag']

    lags = range(1, 7)
    lags += [24, 48, 72, 96, 120, 144, 168]

    names = [t + "_" + str(l) for t in lag_names for l in lags]

    name_list += names

    lag_vars = ['sessions', 'registrations', 'PL_QualStart', 'PL_Submit', 'SLR_QualStart', 'SLR_Submit']

    new_cols = pd.DataFrame()

    for var in lag_vars:
        for lag in lags:
            new_data = base_set[var].shift(lag)
            new_cols = pd.concat([new_cols, new_data], axis=1, )

    enhanced_set = pd.concat([base_set, new_cols], axis=1)

    enhanced_set.columns = name_list

    enhanced_set['tv_dur_cumsum'] = enhanced_set['tv_duration_secs'].cumsum()
    enhanced_set['radio_dur_cumsum'] = enhanced_set['radio_duration_secs'].cumsum()

    # Create some sine wave bounded from 0 to 1 for hourly predictions since we know traffic is at a low point at 12 AM
    Sines1 = [((np.sin(((int(x) + 0) / 12.0) * np.pi - np.pi / 2) + 1) / 2) for x in enhanced_set['hits_hour']]
    Sines2 = [((np.sin(((int(x) + 3) / 12.0) * np.pi - np.pi / 2) + 1) / 2) for x in enhanced_set['hits_hour']]
    Sines3 = [((np.sin(((int(x) + 6) / 12.0) * np.pi - np.pi / 2) + 1) / 2) for x in enhanced_set['hits_hour']]
    Sines4 = [((np.sin(((int(x) + 9) / 12.0) * np.pi - np.pi / 2) + 1) / 2) for x in enhanced_set['hits_hour']]
    enhanced_set['sines1'] = pd.Series(data=Sines1, index=enhanced_set.index)
    enhanced_set['sines2'] = pd.Series(data=Sines2, index=enhanced_set.index)
    enhanced_set['sines3'] = pd.Series(data=Sines3, index=enhanced_set.index)
    enhanced_set['sines4'] = pd.Series(data=Sines4, index=enhanced_set.index)

    model_data = enhanced_set.dropna()
    training_limit = int(math.ceil(model_data.shape[0] * training_fraction))


    if SinusoidModel:
        model_cols = names + ['sines1', 'sines2', 'sines3', 'sines4',
                              'tv_dur_cumsum', 'radio_dur_cumsum',
                              'cumsum_7_tv_dur', 'cumsum_14_tv_dur',
                              'cumsum_7_radio_dur', 'cumsum_14_radio_dur'] + shows
    else:
        model_cols = names + shows

    explanatory_vars = model_data[model_cols]

    explained_vars = model_data[['sessions',
                                 'registrations',
                                 'PL_QualStart',
                                 'PL_Submit',
                                 'SLR_QualStart',
                                 'SLR_Submit']]

    training_xvar = explanatory_vars[0:training_limit]
    training_yvar = explained_vars[0:training_limit]

    poly = PolynomialFeatures(degree=polynomial_degree, interaction_only=interactions_only, include_bias=False)

    transformed_xvar = poly.fit_transform(training_xvar)
    target_feature_names = ['x'.join(['{}^{}'.format(pair[0], pair[1]) for pair in tuple if pair[1] != 0]) for tuple in
                            [zip(training_xvar.columns, p) for p in poly.powers_]]

    transformed_xvar = pd.DataFrame(transformed_xvar, columns=target_feature_names)

    transformed_explanatory = poly.transform(explanatory_vars)
    target_feature_names = ['x'.join(['{}^{}'.format(pair[0], pair[1]) for pair in tuple if pair[1] != 0]) for tuple in
                            [zip(explanatory_vars.columns, p) for p in poly.powers_]]
    transformed_explanatory = pd.DataFrame(transformed_explanatory, columns=target_feature_names)

    print "data formed, training model using {0} observations and {1} features".format(training_xvar.shape[0],
                                                                                               training_xvar.shape[1])

    ############################
    #
    # Training
    #
    ############################

    loop_components = ['type', 'hour']

    session_vars = [col for col in transformed_xvar.columns.values if
                  not re.search(pattern=r'PL_QualStart|PL_Submit|SLR_QualStart|SLR_Submit|registrations', string=col)]
    session_df = transformed_xvar[session_vars]
    linreg_ARIMA_sessions = fit_formula(x_vars=session_df, y=training_yvar['sessions'])

    registration_vars = [col for col in transformed_xvar.columns.values if
                  not re.search(pattern=r'sessions|PL_Submit|SLR_QualStart|SLR_Submit|PL_QualStart', string=col)]
    registration_df = transformed_xvar[registration_vars]
    linreg_ARIMA_registrations = fit_formula(x_vars=registration_df, y=training_yvar['registrations'])

    plqs_vars = [col for col in transformed_xvar.columns.values if
                  not re.search(pattern=r'sessions|PL_Submit|SLR_QualStart|SLR_Submit|registrations', string=col)]
    plqs_df = transformed_xvar[plqs_vars]
    linreg_ARIMA_PL_QS = fit_formula(x_vars=plqs_df, y=training_yvar['PL_QualStart'])

    plsub_vars = [col for col in transformed_xvar.columns.values if
                  not re.search(pattern=r'sessions|PL_QualStart|SLR_QualStart|SLR_Submit|registrations', string=col)]
    plsub_df = transformed_xvar[plsub_vars]
    linreg_ARIMA_PL_Submit = fit_formula(x_vars=plsub_df, y=training_yvar['PL_Submit'])

    slrqs_vars = [col for col in transformed_xvar.columns.values if
                  not re.search(pattern=r'sessions|PL_Submit|PL_QualStart|SLR_Submit|registrations', string=col)]
    slrqs_df = transformed_xvar[slrqs_vars]
    linreg_ARIMA_SLR_QS = fit_formula(x_vars=slrqs_df, y=training_yvar['SLR_QualStart'])

    slrsub_vars = [col for col in transformed_xvar.columns.values if
                  not re.search(pattern=r'sessions|PL_Submit|SLR_QualStart|PL_QualStart|registrations', string=col)]
    slrsub_df = transformed_xvar[slrsub_vars]
    linreg_ARIMA_SLR_Submit = fit_formula(x_vars=slrsub_df, y=training_yvar['SLR_Submit'])

    print "{0} traffic ARIMA models trained".format(mode)

    session_df = transformed_explanatory[session_vars]
    model_data.loc[:, 'predicted_sessions'] = pd.Series(data=linreg_ARIMA_sessions.predict(session_df),
                                                        index=model_data.index)
    plqs_df = transformed_explanatory[plqs_vars]
    model_data.loc[:, 'predicted_PL_QualStart'] = pd.Series(data=linreg_ARIMA_PL_QS.predict(plqs_df),
                                                            index=model_data.index)
    plsub_df = transformed_explanatory[plsub_vars]
    model_data.loc[:, 'predicted_PL_Submit'] = pd.Series(data=linreg_ARIMA_PL_Submit.predict(plsub_df),
                                                         index=model_data.index)
    slrqs_df = transformed_explanatory[slrqs_vars]
    model_data.loc[:, 'predicted_SLR_QualStart'] = pd.Series(data=linreg_ARIMA_SLR_QS.predict(slrqs_df),
                                                             index=model_data.index)
    slrsub_df = transformed_explanatory[slrsub_vars]
    model_data.loc[:, 'predicted_SLR_Submit'] = pd.Series(data=linreg_ARIMA_SLR_Submit.predict(slrsub_df),
                                                          index=model_data.index)
    registration_df = transformed_explanatory[registration_vars]
    model_data.loc[:, 'predicted_registrations'] = pd.Series(data=linreg_ARIMA_registrations.predict(registration_df),
                                                             index=model_data.index)



    # Model statistics
    model_data.loc[:, 'error_ARIMA'] = model_data['predicted_sessions'] - model_data['sessions']

    print("{0} Data metrics".format(mode))
    print metrics.r2_score(y_true=model_data['sessions'], y_pred=model_data['predicted_sessions'])
    print metrics.r2_score(y_true=model_data['registrations'], y_pred=model_data['predicted_registrations'])
    print metrics.r2_score(y_true=model_data['PL_QualStart'], y_pred=model_data['predicted_PL_QualStart'])
    print metrics.r2_score(y_true=model_data['PL_Submit'], y_pred=model_data['predicted_PL_Submit'])
    print metrics.r2_score(y_true=model_data['SLR_QualStart'], y_pred=model_data['predicted_SLR_QualStart'])
    print metrics.r2_score(y_true=model_data['SLR_Submit'], y_pred=model_data['predicted_SLR_Submit'])

    # model_data.to_csv("direct_traffic_predictions_new.csv")
    #out_df = pd.concat(objs=[out_df, model_data], axis=0)

    return [model_data,
            (linreg_ARIMA_PL_QS, plqs_df),
            (linreg_ARIMA_PL_Submit, plsub_df),
            (linreg_ARIMA_registrations, registration_df),
            (linreg_ARIMA_sessions, session_df),
            (linreg_ARIMA_SLR_QS, slrqs_df),
            (linreg_ARIMA_SLR_Submit, slrsub_df)
            ]

dir_model_data, dir_PL_QS, dir_PL_Submit, dir_registrations, dir_sessions, dir_SLR_QS, dir_SLR_Submit = model_it(mode="direct", training_fraction=1.0, polynomial_degree=1, interactions_only=True)
paid_model_data, paid_PL_QS, paid_PL_Submit, paid_registrations, paid_sessions, paid_SLR_QS, paid_SLR_Submit = model_it(mode="paid", training_fraction=1.0, polynomial_degree=1, interactions_only=True)
org_model_data, org_PL_QS, org_PL_Submit, org_registrations, org_sessions, org_SLR_QS, org_SLR_Submit = model_it(mode="organic", training_fraction=1.0, polynomial_degree=1, interactions_only=True)

out_df = pd.concat(objs=[dir_model_data, paid_model_data, org_model_data], axis=0)
out_df.to_csv("out_df_20161021.csv")


def coef_ranking(model_output):
    model_values = zip(model_output[1].columns.values, model_output[0].coef_)
    key_function = lambda x: x[0].split("^")[0]
    i = 0
    result_df = pd.DataFrame(columns=('show', 'contribution'))
    for key, values in itertools.groupby(model_values, key=key_function):
        contrib = sum(float(v[1]) for v in values)
        print ("adding {0} with an overall contribution of {1}".format(key, contrib))
        row = [key, contrib]
        result_df.loc[i] = row
        i += 1

    print("summed {0} individual shows".format(i))

    grouped = result_df.groupby(by=['show'], as_index=False)['contribution'].sum()

    sorted = grouped.sort(columns='contribution',ascending=False)

    return sorted

dir_slr = coef_ranking(dir_SLR_Submit)
dir_slr.columns = ['show', 'dir_SLR_Sub_Per_Bcast_Second']
paid_slr = coef_ranking(paid_SLR_Submit)
paid_slr.columns = ['show', 'paid_SLR_Sub_Per_Bcast_Second']
org_slr = coef_ranking(org_SLR_Submit)
org_slr.columns = ['show', 'org_SLR_Sub_Per_Bcast_Second']

dir_pl = coef_ranking(dir_PL_Submit)
dir_pl.columns = ['show', 'dir_PL_Sub_Per_Bcast_Second']
paid_pl = coef_ranking(paid_PL_Submit)
paid_pl.columns = ['show', 'paid_PL_Sub_Per_Bcast_Second']
org_pl = coef_ranking(org_PL_Submit)
org_pl.columns = ['show', 'org_PL_Sub_Per_Bcast_Second']

all_time_df = pd.DataFrame(columns=('show', 'bcast_seconds_all'))
i = 0
for show in shows:
    # We can take any model data to sum broadcasts
    time = schedule_data[show].sum()
    all_time_df.loc[i] = [show, time]
    i += 1

q3_time_df = pd.DataFrame(columns=('show', 'bcast_seconds_q3'))
i = 0
q3_sched = schedule_data[schedule_data['broadcast_date'] > datetime.datetime(2016, 6, 30)]
for show in shows:
    # We can take any model data to sum broadcasts
    time = q3_sched[show].sum()
    q3_time_df.loc[i] = [show, time]
    i += 1

slr = pd.merge(left=dir_slr, right=paid_slr, how='left', on='show')
slr = pd.merge(left=slr, right=org_slr, how='left', on='show')
slr = pd.merge(left=slr, right = all_time_df, how='inner', on='show')
slr = pd.merge(left=slr, right = q3_time_df, how='left', on='show')

slr.to_csv("SLR_relative_show_contribution.csv")

pl = pd.merge(left=dir_pl, right=paid_pl, how='left', on='show')
pl = pd.merge(left=pl, right=org_pl, how='left', on='show')
pl = pd.merge(left=pl, right = all_time_df, how='inner', on='show')
pl = pd.merge(left=pl, right = q3_time_df, how='left', on='show')

pl.to_csv("PL_relative_show_contribution.csv")

slr['larger_group'] = slr['show'].apply(lambda x: x.split("-")[0].strip())
pl['larger_group'] = pl['show'].apply(lambda x: x.split("-")[0].strip())

dir_sess = coef_ranking(dir_sessions)
dir_sess.columns = ['show', 'hourly_dir_sess_per_bcast_second']
paid_sess = coef_ranking(paid_sessions)
paid_sess.columns = ['show', 'hourly_paid_sess_per_bcast_second']
org_sess = coef_ranking(org_sessions)
org_sess.columns = ['show', 'hourly_org_sess_per_bcast_second']

sess = pd.merge(left=dir_sess, right=paid_sess, how='left', on='show')
sess = pd.merge(left=sess, right=org_sess, how='left', on='show')
sess = pd.merge(left=sess, right = all_time_df, how='inner', on='show')
sess = pd.merge(left=sess, right = q3_time_df, how='left', on='show')

sess['larger_group'] = sess['show'].apply(lambda x: x.split("-")[0].strip())
sess.to_csv("sess_relative_show_contribution.csv")