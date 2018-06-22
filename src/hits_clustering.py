import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
import multiprocessing as mp
import threading as thr

from sklearn.preprocessing import StandardScaler
#import hdbscan
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import argparse
import seeds as sd
import collections as coll
import math
from extension import extend_submission, extend_labels
import cone_slicing as cone
import merge as merge

# Nicole's first model, no z2
# SCALED_DISTANCE = [1,       1,       0.5, 0.01, 0.01, 0.001, 0.001]
# FEATURE_MATRIX = ['sina1', 'cosa1', 'z1', 'xd', 'yd', 'px', 'py']

SCALED_DISTANCE = [1,       1,       0.5, 0.125, 0.01, 0.01, 0.001, 0.001]
FEATURE_MATRIX = ['sina1', 'cosa1', 'z1', 'z2',  'xd', 'yd', 'px', 'py']

SCALED_DISTANCE_2 = [1,       1,       0.5, 0.01, 0.01, 0.001, 0.001]
FEATURE_MATRIX_2 = ['sina1', 'cosa1', 'z3', 'xd', 'yd', 'px', 'py']

SCALED_DISTANCE_3 = [1,       1,       0.5, 0.125, 0.01, 0.01, 0.001, 0.001]
FEATURE_MATRIX_3= ['sina1', 'cosa1', 'z4', 'z1', 'xd', 'yd', 'px', 'py']

DBSCAN_LOCAL_EPS = 0.0033
DBSCAN_LOCAL_EPS_1 = 0.0041
DBSCAN_LOCAL_EPS_2 = 0.0037
DBSCAN_LOCAL_EPS_3 = 0.0045

STEPRR = 0.03

STEPEPS = 0.0000015
STEPS = 120
EXTENSION_ATTEMPT = 8
# 0.06-0.07 is the most ideal value in most cases, so we want it to be the final processing parameter

EXTENSION_LIMIT_START = 0.03
EXTENSION_LIMIT_INTERVAL = 0.005

MIN_CONE_TRACK_LENGTH = 2
MAX_CONE_TRACK_LENGTH = 30
THRESHOLD_MIN = 5
THRESHOLD_MAX = 30

print('Feature matrix: ' + str(FEATURE_MATRIX))
print('scaled distance: ' + str(SCALED_DISTANCE))

print('Feature matrix 2nd pass: ' + str(FEATURE_MATRIX_2))
print('scaled distance 2nd pass: ' + str(SCALED_DISTANCE_2))

print('Feature matrix 3rd pass: ' + str(FEATURE_MATRIX_3))
print('scaled distance 3rd pass: ' + str(SCALED_DISTANCE_3))

print('stepeps: ' + str(STEPEPS))

print('steprr: ' + str(STEPRR))
print('local eps: ' + str(DBSCAN_LOCAL_EPS))
print('local eps 1: ' + str(DBSCAN_LOCAL_EPS_1))
print('local eps 2: ' + str(DBSCAN_LOCAL_EPS_2))
print('local eps 3: ' + str(DBSCAN_LOCAL_EPS_3))

print('steps: ' + str(STEPS))
print('threshold min: ' + str(THRESHOLD_MIN))
print('threshold max: ' + str(THRESHOLD_MAX))
print('extension attempt: ' + str(EXTENSION_ATTEMPT))
print('extension range from  ' + str(EXTENSION_LIMIT_START) + ' to ' + str(EXTENSION_LIMIT_START + EXTENSION_LIMIT_INTERVAL*(EXTENSION_ATTEMPT-1)))


class Clusterer(object):
    def __init__(self, model_parameters):                        
        self.model_parameters = model_parameters

    def _eliminate_outliers(self,labels):
        indices=np.zeros((len(labels)),np.float32)
 
        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if cluster == 0:
                continue
            index = np.argwhere(self.clusters==cluster)
            index = np.reshape(index,(index.shape[0]))
            indices[i] = len(index)
          
        for i, cluster in enumerate(labels):
            if indices[i] > THRESHOLD_MAX or indices[i] < THRESHOLD_MIN:   
                self.clusters[self.clusters==cluster]=0  


    def _dbscan(self, dfh, label_file_root):
        label_file1 = label_file_root + '_dbscan1.csv'
        label_file2 = label_file_root + '_dbscan2.csv'
        label_file3 = label_file_root + '_dbscan3.csv'
        label_file4 = label_file_root + '_dbscan4.csv'

        dfh['d'] = np.sqrt(dfh.x**2+dfh.y**2+dfh.z**2)
        dfh['r'] = np.sqrt(dfh.x**2+dfh.y**2)
        dfh['z1'] = dfh['z']/dfh['r'] 
        dfh['z2'] = dfh['z']/dfh['d']
        dfh['z3'] = np.log1p(np.absolute(dfh.z/dfh.r))*np.sign(dfh.z)
        dfh['z4'] = np.arccos(dfh.z/dfh.d)
        dfh['z5'] = (dfh.r - np.absolute(dfh.z))/dfh.r
        #theta
        dfh['rz'] = np.arctan2(dfh.r, dfh.z)
        rr = dfh['r']/1000      

        dfh['a0'] = np.arctan2(dfh.y,dfh.x)
        dfh['xd'] = dfh.x/dfh['d']
        dfh['yd'] = dfh.y/dfh['d']

        if os.path.exists(label_file1):
            print('Loading dbscan loop 1 file: ' + label_file1)
            labels_loop1 = pd.read_csv(label_file1).label.values
        else:
            for ii in tqdm(np.arange(-STEPS, STEPS, 1)):
                print ('\r steps: %d '%ii, end='',flush=True)
                dfh['a1'] = dfh['a0'] + (rr + self.model_parameters[2][0]*rr**2)*ii/180*np.pi + (0.00001*ii)*dfh.z*np.sign(dfh.z)/180*np.pi
                dfh['za1'] = dfh.z/dfh['a1']
                dfh['z1a1'] = dfh['z1']/dfh['a1']
                dfh['rcos'] = dfh.r/np.cos(dfh.a1 - dfh.a0)

                dfh['x2'] = 1/dfh['z1']
                dfh['cur'] = np.absolute(dfh.r) / (dfh.r**2 + (dfh.z/dfh.a1)**2)
                # parameter space
                dfh['px'] = -dfh.r*np.cos(dfh.a1)*np.cos(dfh.a0) - dfh.r*np.sin(dfh.a1)*np.sin(dfh.a0)
                dfh['py'] = -dfh.r*np.cos(dfh.a1)*np.sin(dfh.a0) + dfh.r*np.sin(dfh.a1)*np.cos(dfh.a0)
                
                dfh['sina1'] = np.sin(dfh['a1'])
                dfh['cosa1'] = np.cos(dfh['a1'])
                
                ss = StandardScaler()
            
                dfs = ss.fit_transform(dfh[self.model_parameters[0]].values)
                dfs = np.multiply(dfs, self.model_parameters[1])
        
                self.clusters = DBSCAN(eps=self.model_parameters[3][0] + ii*STEPEPS,min_samples=1, n_jobs=-1).fit(dfs).labels_

                if ii == -STEPS:
                    dfh['s1'] = self.clusters
                    dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
                else:
                    dfh['s2'] = self.clusters
                    dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
                    maxs1 = dfh['s1'].max()
                    cond = np.where( (dfh['N2'].values>dfh['N1'].values) & (dfh['N2'].values < 20) )
                    s1 = dfh['s1'].values
                    s1[cond] = dfh['s2'].values[cond]+maxs1
                    dfh['s1'] = s1
                    dfh['s1'] = dfh['s1'].astype('int64')
                    dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')

            labels_loop1 = np.copy(dfh['s1'].values)
            df = pd.DataFrame(labels_loop1)
            df.to_csv(label_file1, index=False, header=['label'])

        dfh['a0'] = np.arctan2(dfh.x,-dfh.y)
        dfh['xd'] = -dfh.y/dfh['d']
        dfh['yd'] = dfh.x/dfh['d']
        
        if os.path.exists(label_file2):
            print('Loading dbscan loop 2 file: ' + label_file2)
            labels_loop2 = pd.read_csv(label_file2).label.values
        else:
            for ii in tqdm(np.arange(-STEPS, STEPS, 1)):
                print ('\r steps: %d '%ii, end='',flush=True)

                dfh['a1'] = dfh['a0'] + (rr + self.model_parameters[2][0]*rr**2)*ii/180*np.pi + (0.00001*ii)*dfh.z*np.sign(dfh.z)/180*np.pi
        
                dfh['za1'] = dfh.z/dfh['a1']
                dfh['z1a1'] = dfh['z1']/dfh['a1']
                dfh['rcos'] = dfh.r/np.cos(dfh.a1 - dfh.a0)

                dfh['cur'] = np.absolute(dfh.r) / (dfh.r**2 + (dfh.z/dfh.a1)**2)
                # parameter space
                dfh['px'] = -dfh.r*np.cos(dfh.a1)*np.cos(dfh.a0) - dfh.r*np.sin(dfh.a1)*np.sin(dfh.a0)
                dfh['py'] = -dfh.r*np.cos(dfh.a1)*np.sin(dfh.a0) + dfh.r*np.sin(dfh.a1)*np.cos(dfh.a0)
        
                dfh['sina1'] = np.sin(dfh['a1'])
                dfh['cosa1'] = np.cos(dfh['a1'])
        
                ss = StandardScaler()
        
                dfs = ss.fit_transform(dfh[self.model_parameters[0]].values)
                dfs = np.multiply(dfs, self.model_parameters[1])

                self.clusters = DBSCAN(eps=self.model_parameters[3][1] + ii*STEPEPS,min_samples=1, n_jobs=-1).fit(dfs).labels_

                if ii == -STEPS:
                    dfh['s1'] = self.clusters
                    dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
                else:
                    dfh['s2'] = self.clusters
                    dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
                    maxs1 = dfh['s1'].max()
                    cond = np.where( (dfh['N2'].values>dfh['N1'].values) & (dfh['N2'].values < 20) )
                    s1 = dfh['s1'].values
                    s1[cond] = dfh['s2'].values[cond]+maxs1
                    dfh['s1'] = s1
                    dfh['s1'] = dfh['s1'].astype('int64')
                    dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')

            labels_loop2 = np.copy(dfh['s1'].values)
            df = pd.DataFrame(labels_loop2)
            df.to_csv(label_file2, index=False, header=['label'])
 
        dfh['a0'] = np.arctan2(-dfh.y,-dfh.x)
        dfh['xd'] = -dfh.x/dfh['d']
        dfh['yd'] = -dfh.y/dfh['d']

        if os.path.exists(label_file3):
            print('Loading dbscan loop 3 file: ' + label_file3)
            labels_loop3 = pd.read_csv(label_file3).label.values
        else:
            for ii in tqdm(np.arange(-STEPS, STEPS, 1)):
                print ('\r steps: %d '%ii, end='',flush=True)

                dfh['a1'] = dfh['a0'] + (rr + self.model_parameters[2][0]*rr**2)*ii/180*np.pi + (0.00001*ii)*dfh.z*np.sign(dfh.z)/180*np.pi
        
                dfh['za1'] = dfh.z/dfh['a1']
                dfh['z1a1'] = dfh['z1']/dfh['a1']
                dfh['rcos'] = dfh.r/np.cos(dfh.a1)


                dfh['cur'] = np.absolute(dfh.r) / (dfh.r**2 + (dfh.z/dfh.a1)**2)
                # parameter space
                dfh['px'] = -dfh.r*np.cos(dfh.a1)*np.cos(dfh.a0) - dfh.r*np.sin(dfh.a1)*np.sin(dfh.a0)
                dfh['py'] = -dfh.r*np.cos(dfh.a1)*np.sin(dfh.a0) + dfh.r*np.sin(dfh.a1)*np.cos(dfh.a0)
        
                dfh['sina1'] = np.sin(dfh['a1'])
                dfh['cosa1'] = np.cos(dfh['a1'])
        
                ss = StandardScaler()
        
                dfs = ss.fit_transform(dfh[self.model_parameters[0]].values)
                dfs = np.multiply(dfs, self.model_parameters[1])

                self.clusters = DBSCAN(eps=self.model_parameters[3][2] + ii*STEPEPS,min_samples=1, n_jobs=-1).fit(dfs).labels_

                if ii == -STEPS:
                    dfh['s1'] = self.clusters
                    dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
                else:
                    dfh['s2'] = self.clusters
                    dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
                    maxs1 = dfh['s1'].max()
                    cond = np.where( (dfh['N2'].values>dfh['N1'].values) & (dfh['N2'].values < 20) )
                    s1 = dfh['s1'].values
                    s1[cond] = dfh['s2'].values[cond]+maxs1
                    dfh['s1'] = s1
                    dfh['s1'] = dfh['s1'].astype('int64')
                    dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')

            labels_loop3 = np.copy(dfh['s1'].values)
            df = pd.DataFrame(labels_loop3)
            df.to_csv(label_file3, index=False, header=['label'])
 
        if os.path.exists(label_file4):
            print('Loading dbscan loop 4 file: ' + label_file4)
            labels_loop4 = pd.read_csv(label_file4).label.values
        else:
            for ii in tqdm(np.arange(-STEPS, STEPS, 1)):
                print ('\r steps: %d '%ii, end='',flush=True)

                dfh['a1'] = dfh['a0'] + (rr + self.model_parameters[2][0]*rr**2)*ii/180*np.pi + (0.00001*ii)*dfh.z*np.sign(dfh.z)/180*np.pi
        
                dfh['za1'] = dfh.z/dfh['a1']
                dfh['z1a1'] = dfh['z1']/dfh['a1']
                dfh['rcos'] = dfh.r/np.cos(dfh.a1)


                dfh['cur'] = np.absolute(dfh.r) / (dfh.r**2 + (dfh.z/dfh.a1)**2)
                # parameter space
                dfh['px'] = -dfh.r*np.cos(dfh.a1)*np.cos(dfh.a0) - dfh.r*np.sin(dfh.a1)*np.sin(dfh.a0)
                dfh['py'] = -dfh.r*np.cos(dfh.a1)*np.sin(dfh.a0) + dfh.r*np.sin(dfh.a1)*np.cos(dfh.a0)
        
                dfh['sina1'] = np.sin(dfh['a1'])
                dfh['cosa1'] = np.cos(dfh['a1'])
        
                ss = StandardScaler()
        
                dfs = ss.fit_transform(dfh[self.model_parameters[0]].values)
                dfs = np.multiply(dfs, self.model_parameters[1])

                self.clusters = DBSCAN(eps=self.model_parameters[3][3] + ii*STEPEPS,min_samples=1, n_jobs=-1).fit(dfs).labels_

                if ii == -STEPS:
                    dfh['s1'] = self.clusters
                    dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
                else:
                    dfh['s2'] = self.clusters
                    dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
                    maxs1 = dfh['s1'].max()
                    cond = np.where( (dfh['N2'].values>dfh['N1'].values) & (dfh['N2'].values < 20) )
                    s1 = dfh['s1'].values
                    s1[cond] = dfh['s2'].values[cond]+maxs1
                    dfh['s1'] = s1
                    dfh['s1'] = dfh['s1'].astype('int64')
                    dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')

            labels_loop4 = np.copy(dfh['s1'].values)
            df = pd.DataFrame(labels_loop4)
            df.to_csv(label_file4, index=False, header=['label'])

        return (labels_loop1, labels_loop2, labels_loop3, labels_loop4)

    def predict(self, hits, label_file_root): 
        (l1, l2, l3, l4)  = self._dbscan(hits, label_file_root)
        # Don't remove outliers, callers can implement better outlier removal.
        # dfh = self._dbscan(dfh, local_eps=DBSCAN_LOCAL_EPS_1, shift=True)
        # self.clusters = dfh['s1'].values
        #labels = np.unique(self.clusters)
        #self._eliminate_outliers(labels)
        return (l1, l2, l3, l4)

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

def hack_one_last_run(labels, labels2, hits2):
    labels2_x = np.copy(labels)
    labels2_x[labels2_x != 0] = 0

    # Expand our labels to include zero(0) for any hits that were removed.
    hits2_indexes = hits2.index.tolist()
    fix_ix = 0
    for hits2_ix in hits2_indexes:
        labels2_x[hits2_ix] = labels2[fix_ix]
        fix_ix = fix_ix + 1
    return labels2_x

def run_cone_slicing_predictions(event_id, all_hits, label_identifier):

    # Shortcut - if we've previously generated and saved labels, just use them
    # rather than re-generating.
    label_file = 'event_' + str(event_id)+'_labels_' + label_identifier + '.csv'
    if os.path.exists(label_file):
        labels_cone = pd.read_csv(label_file).label.values
        return labels_cone

    # Cone slicing.
    labels_cone1 = cone.slice_cones(all_hits, MIN_CONE_TRACK_LENGTH, MAX_CONE_TRACK_LENGTH, False)
    labels_cone1 = merge.renumber_labels(labels_cone1)

    labels_cone2 = cone.slice_cones(all_hits, MIN_CONE_TRACK_LENGTH, MAX_CONE_TRACK_LENGTH, True)
    labels_cone2 = merge.renumber_labels(labels_cone2)

    labels_cone = merge.heuristic_merge_tracks(labels_cone1, labels_cone2, print_summary=False)

    for i in range(EXTENSION_ATTEMPT):
        limit = EXTENSION_LIMIT_START + EXTENSION_LIMIT_INTERVAL*i
        labels_cone = extend_labels(i, labels_cone, all_hits, do_swap=i%2==1, limit=(limit))

    labels_cone = merge.renumber_labels(labels_cone)

    # Filter out any tracks that do not originate from volumes 7, 8, or 9
    #seed_length = 5
    #my_volumes = [7, 8, 9]
    #labels_cone = merge.filter_invalid_tracks(labels_cone, all_hits, my_volumes, 2)

    # Save the generated labels, can avoid re-generation next run.
    df = pd.DataFrame(labels_cone)
    df.to_csv(label_file, index=False, header=['label'])

    return labels_cone

def run_predictions(event_id, all_labels, all_hits, truth, model, label_file_root, unmatched_only=True, filter_hits=True, track_extension=True):
    """ Run a round of predictions on all or a subset of remaining hits.
    Parameters:
      all_labels: Input np array of labeled tracks, where the index in all_labels matches
        the index of the corresponding hit in the all_hits dataframe. Each value contains
        either 0 (if the corresponding hit is not associated with any track), or a unique
        track ID (all hits that form the same track will have the same track ID).
      all_hits: Dataframe containing all the hits to be predicted for an event.
      model: The model that predictions will be run on. This model must expose a
        'predict()' method that accepts the input hits to predict as the first parameter,
        and the input 'model_parameters' as the second input parameter.
      unmatched_only: True iff only unmatched hits should be predicted. Unmatched hits are
        determined from the all_labels input array, where an unmatched hit contains a
        track ID of 0. False for this parameter means that all hits in the all_hits
        dataframe will be used to make predictions.
      filter_hits: True iff the predicted hits should be filter to those known to be
        high quality, i.e. that have a specific minimum track length, and that
        contain hits in volumes 7, 8, or 9. False for this parameter means that no
        filtering will be performed.
      track_extension: True iff found tracks should be extended (both ways) to lengthen
        any found tracks. Track extension is performed from the full list of tracks, i.e.
        after mergeing (if mergeing was performed).

    Returns: The new np array of predicted labels/tracks, as well as the unfiltered version.
    """
    hits_to_predict = all_hits

    if unmatched_only:
        # Make a copy of the hits, removing all hits from valid_labels
        hits_to_predict = all_hits.copy(deep=True)
        drop_indices = np.where(all_labels != 0)[0]
        hits_to_predict = hits_to_predict.drop(hits_to_predict.index[drop_indices])

    # Run predictions on the input model
    (l1, l2, l3, l4) = model.predict(hits_to_predict, label_file_root)

    # Make sure max track ID is not larger than length of labels list.
    l1 = sd.renumber_labels(l1)
    l2 = sd.renumber_labels(l2)
    l3 = sd.renumber_labels(l3)
    l4 = sd.renumber_labels(l4)

    # If only predicting on unmatched hits, add any new predicted tracks directly
    # into the output labels. Otherwise, just return the newly predicted output labels.
    if unmatched_only:
        l1a = np.copy(all_labels)
        l1[l1 == 0] = 0 - len(all_labels) - 1
        l1 = l1 + len(all_labels) + 1
        l1a[l1a == 0] = l1
        l2a = np.copy(all_labels)
        l2[l2 == 0] = 0 - len(all_labels) - 1
        l2 = l2 + len(all_labels) + 1
        l2a[l2a == 0] = l2
        l3a = np.copy(all_labels)
        l3[l3 == 0] = 0 - len(all_labels) - 1
        l3 = l3 + len(all_labels) + 1
        l3a[l3a == 0] = l3
        l4a = np.copy(all_labels)
        l4[l4 == 0] = 0 - len(all_labels) - 1
        l4 = l4 + len(all_labels) + 1
        l4a[l4a == 0] = l4
    else:
        l1a = l1
        l2a = l2
        l3a = l3
        l4a = l4

    one_submission = create_one_event_submission(event_id, all_hits, l1a)
    score = score_event(truth, one_submission)
    print("Unfiltered dbscan loop 1 score for event %d: %.8f" % (event_id, score))
    one_submission = create_one_event_submission(event_id, all_hits, l2a)
    score = score_event(truth, one_submission)
    print("Unfiltered dbscan loop 2 score for event %d: %.8f" % (event_id, score))
    one_submission = create_one_event_submission(event_id, all_hits, l3a)
    score = score_event(truth, one_submission)
    print("Unfiltered dbscan loop 3 score for event %d: %.8f" % (event_id, score))
    one_submission = create_one_event_submission(event_id, all_hits, l4a)
    score = score_event(truth, one_submission)
    print("Unfiltered dbscan loop 4 score for event %d: %.8f" % (event_id, score))

    # If desired, extend tracks
    if track_extension:
        for i in range(EXTENSION_ATTEMPT):
            limit = EXTENSION_LIMIT_START + EXTENSION_LIMIT_INTERVAL*i
            l1a = extend_labels(i, l1a, all_hits, do_swap=i%2==1, limit=(limit))
            l2a = extend_labels(i, l2a, all_hits, do_swap=i%2==1, limit=(limit))
            l3a = extend_labels(i, l3a, all_hits, do_swap=i%2==1, limit=(limit))
            l4a = extend_labels(i, l4a, all_hits, do_swap=i%2==1, limit=(limit))
    
        l1a = sd.renumber_labels(l1a)
        l2a = sd.renumber_labels(l2a)
        l3a = sd.renumber_labels(l3a)
        l4a = sd.renumber_labels(l4a)
        one_submission = create_one_event_submission(event_id, all_hits, l1a)
        score = score_event(truth, one_submission)
        print("Unfiltered extended dbscan loop 1 score for event %d: %.8f" % (event_id, score))
        one_submission = create_one_event_submission(event_id, all_hits, l2a)
        score = score_event(truth, one_submission)
        print("Unfiltered extended dbscan loop 2 score for event %d: %.8f" % (event_id, score))
        one_submission = create_one_event_submission(event_id, all_hits, l3a)
        score = score_event(truth, one_submission)
        print("Unfiltered extended dbscan loop 3 score for event %d: %.8f" % (event_id, score))
        one_submission = create_one_event_submission(event_id, all_hits, l4a)
        score = score_event(truth, one_submission)
        print("Unfiltered extended dbscan loop 4 score for event %d: %.8f" % (event_id, score))
    else:
        l1a = sd.renumber_labels(l1a)
        l2a = sd.renumber_labels(l2a)
        l3a = sd.renumber_labels(l3a)
        l4a = sd.renumber_labels(l4a)


    if filter_hits:
        # Filter out any tracks that do not originate from volumes 7, 8, or 9.
        # Then, re-number all tracks so they are densely packed.
        seed_length = 5
        my_volumes = [7, 8, 9]
        l1a = sd.filter_invalid_tracks(l1a, all_hits, my_volumes, seed_length)
        l1a = sd.renumber_labels(l1a)
        l2a = sd.filter_invalid_tracks(l2a, all_hits, my_volumes, seed_length)
        l2a = sd.renumber_labels(l2a)
        l3a = sd.filter_invalid_tracks(l3a, all_hits, my_volumes, seed_length)
        l3a = sd.renumber_labels(l3a)
        l4a = sd.filter_invalid_tracks(l4a, all_hits, my_volumes, seed_length)
        l4a = sd.renumber_labels(l4a)

        one_submission = create_one_event_submission(event_id, all_hits, l1a)
        score = score_event(truth, one_submission)
        print("Filtered dbscan loop 1 score for event %d: %.8f" % (event_id, score))
        one_submission = create_one_event_submission(event_id, all_hits, l2a)
        score = score_event(truth, one_submission)
        print("Filtered dbscan loop 2 score for event %d: %.8f" % (event_id, score))
        one_submission = create_one_event_submission(event_id, all_hits, l3a)
        score = score_event(truth, one_submission)
        print("Filtered dbscan loop 3 score for event %d: %.8f" % (event_id, score))
        one_submission = create_one_event_submission(event_id, all_hits, l4a)
        score = score_event(truth, one_submission)
        print("Filtered dbscan loop 4 score for event %d: %.8f" % (event_id, score))

        # Perform sophisticated outlier removal, duplicate-z removal, slope-based removal
        l1a = merge.remove_outliers(l1a, all_hits, print_counts=False)
        l2a = merge.remove_outliers(l2a, all_hits, print_counts=False)
        l3a = merge.remove_outliers(l3a, all_hits, print_counts=False)
        l4a = merge.remove_outliers(l4a, all_hits, print_counts=False)

        one_submission = create_one_event_submission(event_id, all_hits, l1a)
        score = score_event(truth, one_submission)
        print("Filtered non-outlier dbscan loop 1 score for event %d: %.8f" % (event_id, score))
        one_submission = create_one_event_submission(event_id, all_hits, l2a)
        score = score_event(truth, one_submission)
        print("Filtered non-outlier dbscan loop 2 score for event %d: %.8f" % (event_id, score))
        one_submission = create_one_event_submission(event_id, all_hits, l3a)
        score = score_event(truth, one_submission)
        print("Filtered non-outlier dbscan loop 3 score for event %d: %.8f" % (event_id, score))
        one_submission = create_one_event_submission(event_id, all_hits, l4a)
        score = score_event(truth, one_submission)
        print("Filtered non-outlier dbscan loop 4 score for event %d: %.8f" % (event_id, score))

    # Merge all dbscan loop labels together
    labels_merged = merge.heuristic_merge_tracks(l1a, l2a, print_summary=False)
    one_submission = create_one_event_submission(event_id, all_hits, labels_merged)
    score = score_event(truth, one_submission)
    print("Merged loop 1&2 score for event %d: %.8f" % (event_id, score))
    labels_merged = merge.heuristic_merge_tracks(labels_merged, l3a, print_summary=False)
    one_submission = create_one_event_submission(event_id, all_hits, labels_merged)
    score = score_event(truth, one_submission)
    print("Merged loop 1&2&3 score for event %d: %.8f" % (event_id, score))
    labels_merged = merge.heuristic_merge_tracks(labels_merged, l4a, print_summary=False)
    one_submission = create_one_event_submission(event_id, all_hits, labels_merged)
    score = score_event(truth, one_submission)
    print("Merged loop 1&2&3&4 score for event %d: %.8f" % (event_id, score))

    return (labels_merged)

def run_helix_unrolling_predictions(event_id, hits, truth, label_identifier, model_parameters):
    # Shortcut - if we've previously generated and saved labels, just use them
    # rather than re-generating.
    label_file_root = 'event_' + str(event_id)+'_labels_' + label_identifier
    label_file = label_file_root + '.csv'
    if os.path.exists(label_file):
        print(str(event_id) + ': load ' + label_file)
        labels = pd.read_csv(label_file).label.values
        ##FIXME
        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)
        print("Loaded score for event %d: %.8f" % (event_id, score))
        ##FIXME
        return labels

    print(str(event_id) + ': clustering on ' + label_identifier)

    model = Clusterer(model_parameters)
    
    # For the first run, we do not have an input array of labels/tracks.
    label_file_root1 = label_file_root + '_phase1'
    (labels) = run_predictions(event_id, None, hits, truth, model, label_file_root1, unmatched_only=False, filter_hits=True, track_extension=True)

    if truth is not None:
        # Score for the event
        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)
        print("Filtered 1st pass score for event %d: %.8f" % (event_id, score))

    label_file_root2 = label_file_root + '_phase2'
    model = Clusterer(model_parameters)
    (labels) = run_predictions(event_id, labels, hits, truth, model, label_file_root2, unmatched_only=True, filter_hits=False, track_extension=True)

    if truth is not None:
        # Score for the event
        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)
        print("2nd pass score for event %d: %.8f" % (event_id, score))

        # Un-comment this if you want to see the quality of the seeds generated.
        #seed_length = 5
        #my_volumes = [7, 8, 9]
        #labels = sd.filter_invalid_tracks(labels, hits, my_volumes, seed_length)
        #sd.count_truth_track_seed_hits(labels, truth, seed_length, print_results=True)

    df = pd.DataFrame(labels)
    df.to_csv(label_file, index=False, header=['label'])

    # label_file_root3 = label_file_root + '_phase3'
    # model = Clusterer(model_parameters)
    # (labels) = run_predictions(event_id, labels, hits, truth, model, label_file_root3, unmatched_only=True, merge_labels=True, filter_hits=False, track_extension=True)

    # # Save the generated labels, can avoid re-generation next run.


    # # Score for the event
    # if truth is not None:
    #     one_submission = create_one_event_submission(event_id, hits, labels)
    #     score = score_event(truth, one_submission)
    #     print("3rd pass score for event %d: %.8f" % (event_id, score))

    return labels

def predict_event(event_id, hits, train_or_test, truth):
    model_parameters = []
    model_parameters.append(FEATURE_MATRIX)
    model_parameters.append(SCALED_DISTANCE)
    model_parameters.append([STEPRR])           
    model_parameters.append([DBSCAN_LOCAL_EPS, DBSCAN_LOCAL_EPS_1, DBSCAN_LOCAL_EPS_2, DBSCAN_LOCAL_EPS_3])        
    labels_helix1 = run_helix_unrolling_predictions(event_id, hits, truth, train_or_test + '_helix1', model_parameters)

    model_parameters.clear()
    model_parameters.append(FEATURE_MATRIX_2)
    model_parameters.append(SCALED_DISTANCE_2)
    model_parameters.append([STEPRR])        
    model_parameters.append([DBSCAN_LOCAL_EPS, DBSCAN_LOCAL_EPS_1, DBSCAN_LOCAL_EPS_2, DBSCAN_LOCAL_EPS_3])        
    labels_helix2 = run_helix_unrolling_predictions(event_id, hits, truth, train_or_test + '_helix2', model_parameters)

    model_parameters.clear()
    model_parameters.append(FEATURE_MATRIX_3)
    model_parameters.append(SCALED_DISTANCE_3)
    model_parameters.append([STEPRR])        
    model_parameters.append([DBSCAN_LOCAL_EPS, DBSCAN_LOCAL_EPS_1, DBSCAN_LOCAL_EPS_2, DBSCAN_LOCAL_EPS_3])        
    labels_helix3 = run_helix_unrolling_predictions(event_id, hits, truth, train_or_test + '_helix3', model_parameters)

    # Do cone slicing, use heuristic merge to combine with helix unrolling
    #labels_cone = run_cone_slicing_predictions(event_id, hits, 'train_cone')
    #labels_cone = merge.remove_outliers(labels_cone, hits, print_counts=False)

    # Merge results from two sets of predictions, removing outliers first
    labels_helix1 = merge.remove_outliers(labels_helix1, hits, print_counts=False)
    labels_helix2 = merge.remove_outliers(labels_helix2, hits, print_counts=False)
    labels_helix3 = merge.remove_outliers(labels_helix3, hits, print_counts=False)

    if truth is not None:
        one_submission = create_one_event_submission(event_id, hits, labels_helix1)
        score = score_event(truth, one_submission)
        print("After outlier removal helix1 %d: %.8f" % (event_id, score))
        
        one_submission = create_one_event_submission(event_id, hits, labels_helix2)
        score = score_event(truth, one_submission)
        print("After outlier removal helix2 %d: %.8f" % (event_id, score))

        one_submission = create_one_event_submission(event_id, hits, labels_helix3)
        score = score_event(truth, one_submission)
        print("After outlier removal helix3 %d: %.8f" % (event_id, score))
    
    labels = merge.heuristic_merge_tracks(labels_helix1, labels_helix2, print_summary=False)
    if truth is not None:
        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)
        print("Merged helix1&2 unrolling for event %d: %.8f" % (event_id, score))

    labels = merge.heuristic_merge_tracks(labels, labels_helix3, print_summary=False)
    if truth is not None:
        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)
        print("Merged helix1&2&3 unrolling for event %d: %.8f" % (event_id, score))

    # labels = merge.heuristic_merge_tracks(labels, labels_cone, print_summary=False)
    # one_submission = create_one_event_submission(event_id, hits, labels)
    # score = score_event(truth, one_submission)
    # print("Merged All unrolling and cone slicing for event %d: %.8f" % (event_id, score))

    return labels


def run_single_threaded_training(skip, nevents):
    path_to_train = "../input/train_1"
    dataset_submissions = []
    dataset_scores = []

    for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=skip, nevents=nevents):

        labels = predict_event(event_id, hits, 'train', truth)

        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)

        # labels = merge.heuristic_merge_tracks(labels, labels_cone, print_summary=False)
        # one_submission = create_one_event_submission(event_id, hits, labels)
        # score = score_event(truth, one_submission)
        # print("Merged All unrolling and cone slicing for event %d: %.8f" % (event_id, score))

        # Un-comment this if you want to see the quality of the seeds generated.
        #seed_length = 5
        #my_volumes = [7, 8, 9]
        #labels = sd.filter_invalid_tracks(labels, hits, my_volumes, seed_length)
        #sd.count_truth_track_seed_hits(labels, truth, seed_length, print_results=True)

        # Append the final submission for this event, as well as the score.
        dataset_submissions.append(one_submission)
        dataset_scores.append(score)


    print('Mean score: %.8f' % (np.mean(dataset_scores)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', nargs=2, type=int)
    parser.add_argument('--training', nargs=2, type=int)
    args = parser.parse_args()
    test_skip = 0
    test_events = 0
    training_skip = 0
    training_events = 0

    if args.test is not None:
        test_skip = args.test[0]
        test_events = args.test[1]

    if args.training is not None:
        training_skip = args.training[0]
        training_events = args.training[1]

    if not training_events == 0:
        run_single_threaded_training(training_skip, training_events)

    path_to_test = "../input/test"
    test_dataset_submissions = []

    #create_submission = True # True for submission 
    if test_events > 0:
        print('Predicting test results, skip: %d, events: %d' % (test_skip, test_events))
        use_header = (test_skip == 0)
        for event_id, hits, cells in load_dataset(path_to_test, skip=test_skip, nevents=test_events, parts=['hits', 'cells']):

            print('Event ID: ', event_id)

            # Helix unrolling predictions
            labels = predict_event(event_id, hits, 'test', None)

            # Create our submission for this test event.
            one_submission = create_one_event_submission(event_id, hits, labels)
            test_dataset_submissions.append(one_submission)
            

        # Create submission file
        submission = pd.concat(test_dataset_submissions, axis=0)
        submission_file = 'submission_' + "{:03}".format(test_skip) + '_' + str(test_events) + '.csv'
        submission.to_csv(submission_file, index=False, header=use_header)

