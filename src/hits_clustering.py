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

RZ_SCALES = [0.4, 1.6, 0.5]
SCALED_DISTANCE = [1,       1,       0.5, 0.125, 0.01, 0.01, 0.001, 0.001]
FEATURE_MATRIX = ['sina1', 'cosa1', 'z1', 'z2', 'xd', 'yd', 'px', 'py']
SCALED_DISTANCE_2 = [1,       1,       0.5, 0.01]
FEATURE_MATRIX_2 = ['sina1', 'cosa1', 'z1', 'z1a1' ]


DBSCAN_GLOBAL_EPS = 0.0075
DBSCAN_LOCAL_EPS = 0.0033

STEPRR = 0.03
STEPEPS = 0.0000015
STEPS = 120
THRESHOLD_MIN = 5
THRESHOLD_MAX = 30
EXTENSION_ATTEMPT = 8
# 0.06-0.07 is the most ideal value in most cases, so we want it to be the final processing parameter

EXTENSION_LIMIT_START = 0.03
EXTENSION_LIMIT_INTERVAL = 0.005

MIN_CONE_TRACK_LENGTH = 2
MAX_CONE_TRACK_LENGTH = 30


print('Feature matrix: ' + str(FEATURE_MATRIX))
print('scaled distance: ' + str(SCALED_DISTANCE))

print('Feature matrix 2nd pass: ' + str(FEATURE_MATRIX_2))
print('scaled distance 2nd pass: ' + str(SCALED_DISTANCE_2))

#print('rz scales: ' + str(RZ_SCALES))

print('steprr: ' + str(STEPRR))
print('stepeps: ' + str(STEPEPS))
print('local eps: ' + str(DBSCAN_LOCAL_EPS))
print('steps: ' + str(STEPS))
print('threshold min: ' + str(THRESHOLD_MIN))
print('threshold max: ' + str(THRESHOLD_MAX))
print('extension attempt: ' + str(EXTENSION_ATTEMPT))
print('extension range from  ' + str(EXTENSION_LIMIT_START) + ' to ' + str(EXTENSION_LIMIT_START + EXTENSION_LIMIT_INTERVAL*(EXTENSION_ATTEMPT-1)))
print('minimum cone slice track length: ' + str(MIN_CONE_TRACK_LENGTH))
print('maximum cone slice track length: ' + str(MAX_CONE_TRACK_LENGTH))


class Clusterer(object):
    def __init__(self,rz_scales=RZ_SCALES):                        
        self.rz_scales=rz_scales

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

    def _preprocess(self, hits):
        
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r

        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        for i, rz_scale in enumerate(self.rz_scales):
            X[:,i] = X[:,i] * rz_scale
          
        return X

    def _init(self, dfh, secondpass=False):
        dfh['d'] = np.sqrt(dfh.x**2+dfh.y**2+dfh.z**2)
        dfh['r'] = np.sqrt(dfh.x**2+dfh.y**2)
        dfh['a0'] = np.arctan2(dfh.y,dfh.x)
        dfh['z1'] = dfh['z']/dfh['r'] 
        dfh['z2'] = dfh['z']/dfh['d']
        dfh['z3'] = np.log1p(np.absolute(dfh.z/dfh.r))*np.sign(dfh.z)
        dfh['za0'] = dfh.z/dfh.a0
        rr = dfh['r']/1000
        dfh['rd'] = dfh['r']/dfh['d']
        dfh['xy'] = dfh.x/dfh.y
        dfh['xd'] = dfh.x/dfh['d']
        dfh['yd'] = dfh.y/dfh['d']

        for ii in tqdm(np.arange(-STEPS, STEPS, 1)):
            print ('\r steps: %d '%ii, end='',flush=True)

            dfh['a1'] = dfh['a0'] + (rr + STEPRR*rr**2)*ii/180*np.pi
            dfh['za1'] = dfh.z/dfh['a1']
            dfh['z1a1'] = dfh['z1']/dfh['a1']

            dfh['x2'] = 1/dfh['z1']
            dfh['cur'] = np.absolute(dfh.r) / (dfh.r**2 + (dfh.z/dfh.a1)**2)
            # parameter space
            dfh['px'] = -dfh.r*np.cos(dfh.a1)*np.cos(dfh.a0) - dfh.r*np.sin(dfh.a1)*np.sin(dfh.a0)
            dfh['py'] = -dfh.r*np.cos(dfh.a1)*np.sin(dfh.a0) + dfh.r*np.sin(dfh.a1)*np.cos(dfh.a0)
            
            dfh['sina1'] = np.sin(dfh['a1'])
            dfh['cosa1'] = np.cos(dfh['a1'])
            
            ss = StandardScaler()
   
            if secondpass is True:
                dfs = ss.fit_transform(dfh[FEATURE_MATRIX_2].values)
                dfs = np.multiply(dfs, SCALED_DISTANCE_2)
            else:
                dfs = ss.fit_transform(dfh[FEATURE_MATRIX].values)
                dfs = np.multiply(dfs, SCALED_DISTANCE)

            self.clusters = DBSCAN(eps=DBSCAN_LOCAL_EPS + ii*STEPEPS,min_samples=1, n_jobs=-1).fit(dfs).labels_

            if ii==-STEPS:
                dfh['s1']=self.clusters
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
            else:
                dfh['s2'] = self.clusters
                dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
                maxs1 = dfh['s1'].max()
                cond = np.where(dfh['N2'].values>dfh['N1'].values )
                s1 = dfh['s1'].values
                s1[cond] = dfh['s2'].values[cond]+maxs1
                dfh['s1'] = s1
                dfh['s1'] = dfh['s1'].astype('int64')
                self.clusters = dfh['s1'].values
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')

        # for ii in tqdm(np.arange(0, STEPS, 1)):
        #     print ('\r steps: %d '%ii, end='',flush=True)            
        #     dfh['za0'] = dfh.z/dfh.a0
        #     dfh['a1'] = dfh['a0'] - (rr + STEPRR*rr**2)*ii/180*np.pi
            
        #     dfh['za1'] = dfh.z/dfh['a1']
        #     dfh['z1a1'] = dfh['z1']/dfh['a1']
        #     dfh['cur'] = np.absolute(dfh.r) / (dfh.r**2 + (dfh.z/dfh.a1)**2)
        #         # parameter space
        #     dfh['px'] = -dfh.r*np.cos(dfh.a1)*np.cos(dfh.a0) - dfh.r*np.sin(dfh.a1)*np.sin(dfh.a0)
        #     dfh['py'] = -dfh.r*np.cos(dfh.a1)*np.sin(dfh.a0) + dfh.r*np.sin(dfh.a1)*np.cos(dfh.a0)


        #     dfh['x2'] = 1/dfh['z1'] 
        #     dfh['sina1'] = np.sin(dfh['a1'])
        #     dfh['cosa1'] = np.cos(dfh['a1'])
        #     dfh['xd'] = -dfh.x/dfh['d']
        #     dfh['yd'] = -dfh.y/dfh['d']
            
        #     ss = StandardScaler()
            
        #     dfs = ss.fit_transform(dfh[FEATURE_MATRIX].values)

        #     dfs = np.multiply(dfs, SCALED_DISTANCE)
        #     clusters = DBSCAN(eps=DBSCAN_LOCAL_EPS - ii*STEPEPS,min_samples=1, n_jobs=-1).fit(dfs).labels_

        #     dfh['s2'] = clusters
        #     dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
        #     maxs1 = dfh['s1'].max()
        #     cond = np.where(dfh['N2'].values>dfh['N1'].values)
        #     s1 = dfh['s1'].values
        #     s1[cond] = dfh['s2'].values[cond]+maxs1
        #     dfh['s1'] = s1
        #     dfh['s1'] = dfh['s1'].astype('int64')
        #     dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        return dfh['s1'].values

    def predict(self, hits, secondpass=False): 
        self.clusters = self._init(hits, secondpass)        
            
        labels = np.unique(self.clusters)
        self._eliminate_outliers(labels)
        if secondpass is True:
            X = self._preprocess(hits) 
            max_len = np.max(self.clusters)
            self.clusters[self.clusters==0] = DBSCAN(eps=DBSCAN_GLOBAL_EPS,min_samples=1,algorithm='kd_tree', n_jobs=-1).fit(X[self.clusters==0]).labels_+max_len
        
        return self.clusters

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

def run_single_threaded_training(skip, nevents):
    path_to_train = "../input/train_1"
    dataset_submissions = []
    dataset_scores = []

    for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=skip, nevents=nevents):

        # Cone slicing.
        # labels_cone_1 = cone.slice_cones(hits, MIN_CONE_TRACK_LENGTH, MAX_CONE_TRACK_LENGTH, do_swap=False)
        # labels_cone_2 = cone.slice_cones(hits, MIN_CONE_TRACK_LENGTH, MAX_CONE_TRACK_LENGTH, do_swap=True)
        # labels_cone = sd.merge_tracks(labels_cone_1, labels_cone_2)
        
        # labels_cone = sd.renumber_labels(labels_cone)
        # one_submission = create_one_event_submission(event_id, hits, labels_cone)
        # score = score_event(truth, one_submission)
        # print("Cone slice score for event %d: %.8f" % (event_id, score))
        

        # Helix unrolling track pattern recognition
        model = Clusterer()
        #FIXME remove this code
        #labels = model.predict(hits)

        label_file = 'event_' + str(event_id)+'_labels.csv'
        if os.path.exists(label_file):
            labels = pd.read_csv(label_file).label.values
        else:
            labels = model.predict(hits)
            df = pd.DataFrame(labels)
            df.to_csv(label_file, index=False, header=['label'])
        
        # Score for the event
        one_submission = create_one_event_submission(event_id, hits, labels)
   
        score = score_event(truth, one_submission)
        print("Unroll helix score for event %d: %.8f" % (event_id, score))

        # Make sure max track ID is not larger than length of labels list.
        labels = sd.renumber_labels(labels)
         
       
        for i in range(EXTENSION_ATTEMPT):          
            limit = EXTENSION_LIMIT_START + EXTENSION_LIMIT_INTERVAL*i
            labels = extend_labels(labels, hits, do_swap=i%2==1, limit=(limit))
       
        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)
       
        print("First backfitting for helix score for event %d: %.8f" % (event_id, score))


        # Filter out any tracks that do not originate from volumes 7, 8, or 9
        seed_length = 5
        my_volumes = [7, 8, 9]
        #sd.count_truth_track_seed_hits(labels, truth, seed_length, print_results=True)
        valid_labels = sd.filter_invalid_tracks(labels, hits, my_volumes, seed_length)
        #sd.count_truth_track_seed_hits(valid_labels, truth, seed_length, print_results=True)
        one_submission = create_one_event_submission(event_id, hits, valid_labels)
        score = score_event(truth, one_submission)
        print("Filtered unroll helix score for event %d: %.8f" % (event_id, score))

        # Make a copy of the hits, removing all hits from valid_labels
        hits2 = hits.copy(deep=True)
        drop_indices = np.where(valid_labels != 0)[0]
        hits2 = hits2.drop(hits2.index[drop_indices])

        # Re-run our clustering algorithm on the remaining hits
        model2 = Clusterer()
        labels2 = model2.predict(hits2, True)
        labels2[labels2 == 0] = 0 - len(labels) - 1
        labels2 = labels2 + len(labels) + 1
        # Expand labels2 to include a zero(0) entry for all hits that were removed
        # labels2_x = hack_one_last_run(labels, labels2, hits2)
        # one_submission = create_one_event_submission(event_id, hits, labels2_x)
        # score = score_event(truth, one_submission)
        # print("Score for unroll 2: %.8f" % (score))

        # Create final track labels, merging those tracks found in the first and second passes
        labels3 = np.copy(valid_labels)
        labels3[labels3 == 0] = labels2

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels3)
        # Score for the event
        score = score_event(truth, one_submission)
        print("2-pass helix unroll score for event %d: %.8f" % (event_id, score))

        # Merge/ensemble cone slicing and helix unrolling tracks
        #labels4 = sd.merge_tracks(labels3, labels_cone)
        #FIXME NO MERGE
        one_submission = create_one_event_submission(event_id, hits, labels3)
        score = score_event(truth, one_submission)
        #print("Merged cone+helix score for event %d: %.8f" % (event_id, score))

        for i in range(EXTENSION_ATTEMPT):          
            limit = EXTENSION_LIMIT_START + EXTENSION_LIMIT_INTERVAL*i
            labels3 = extend_labels(labels3, hits, do_swap=i%2==1, limit=(limit))

        one_submission = create_one_event_submission(event_id, hits, labels3)


        # for i in range(EXTENSION_ATTEMPT): 
        #     limit = EXTENSION_LIMIT_START + EXTENSION_LIMIT_INTERVAL*i
        #     one_submission = extend_submission(one_submission, hits, do_swap=i%2==1, limit=limit)
        score = score_event(truth, one_submission)

        print("2nd backfitting for event %d: %.8f" % (event_id, score))
        sd.count_truth_track_seed_hits(labels, truth, seed_length, print_results=True)
       
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

            # Cone slicing.
            # labels_cone = cone.slice_cones(hits, MIN_CONE_TRACK_LENGTH, MAX_CONE_TRACK_LENGTH)
            # labels_cone = sd.renumber_labels(labels_cone)

            # Track pattern recognition 
            model = Clusterer()
            labels = model.predict(hits)

            # Make sure max track ID is not larger than length of labels list.
            labels = sd.renumber_labels(labels)
            for i in range(EXTENSION_ATTEMPT):      
                limit = EXTENSION_LIMIT_START + EXTENSION_LIMIT_INTERVAL*i
                labels = extend_labels(labels, hits, do_swap=i%2==1, limit=limit)

            # Filter out any tracks that do not originate from volumes 7, 8, or 9
            seed_length = 5
            my_volumes = [7, 8, 9]
            valid_labels = sd.filter_invalid_tracks(labels, hits, my_volumes, seed_length)

            # Make a copy of the hits, removing all hits from valid_labels
            hits2 = hits.copy(deep=True)
            drop_indices = np.where(valid_labels != 0)[0]
            hits2 = hits2.drop(hits2.index[drop_indices])

            # Re-run our clustering algorithm on the remaining hits
            model2 = Clusterer()
            labels2 = model2.predict(hits2, True)
            labels2 = labels2 + len(labels) + 1

            labels3 = np.copy(valid_labels)
            labels3[labels3 == 0] = labels2

            # Merge/ensemble cone slicing and helix unrolling tracks
            #labels4 = sd.merge_tracks(labels3, labels_cone)

            #FIXME
            # Prepare submission for an event
            #one_submission = create_one_event_submission(event_id, hits, labels4)
            one_submission = create_one_event_submission(event_id, hits, labels3)


            for i in range(EXTENSION_ATTEMPT): 
                limit = EXTENSION_LIMIT_START + EXTENSION_LIMIT_INTERVAL*i
                one_submission = extend_submission(one_submission, hits, do_swap=i%2==1, limit=limit)

            test_dataset_submissions.append(one_submission)
            

        # Create submission file
        submission = pd.concat(test_dataset_submissions, axis=0)
        submission_file = 'submission_' + "{:03}".format(test_skip) + '_' + str(test_events) + '.csv'
        submission.to_csv(submission_file, index=False, header=use_header)

