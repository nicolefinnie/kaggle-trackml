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
from extension import extend
import cone_slicing as cone

RZ_SCALES = [0.4, 1.6, 0.5]
SCALED_DISTANCE = [1, 1, 0.3, 0.3]
DBSCAN_GLOBAL_EPS = 0.0075

STEPRR = 0.03
STEPEPS = 0.0000015
STEPS = 100
THRESHOLD_MIN = 5
THRESHOLD_MAX = 30
EXTENSION_ATTEMPT = 8

print('rz scales: ' + str(RZ_SCALES))
print('scaled distance: ' + str(SCALED_DISTANCE))
print('steprr: ' + str(STEPRR))
print('stepeps: ' + str(STEPEPS))
print('steps: ' + str(STEPS))
print('threshold min: ' + str(THRESHOLD_MIN))
print('threshold max: ' + str(THRESHOLD_MAX))
print('extension attempt: ' + str(EXTENSION_ATTEMPT))


class DBScanClusterer(object):
    
    def __init__(self):
        self.eps = DBSCAN_GLOBAL_EPS
    
    def _preprocess(self, hits):
        ss = StandardScaler()
        X = ss.fit_transform(hits[['x', 'y', 'z']].values)
        return X
    
    def predict(self, hits):
        X = self._preprocess(hits)
        
        cl = DBSCAN(eps=self.eps, min_samples=3, algorithm='kd_tree')
        labels = cl.fit_predict(X)
        
        return labels

# class HDBScanClusterer(object):
    
#     def __init__(self):
#         self.rz_scales = RZ_SCALES
    
#     def _preprocess(self, hits):
#         ss = StandardScaler()
#         X = ss.fit_transform(hits[['x', 'y', 'z']].values)
#         return X
    
#     def predict(self, hits):
#         X = self._preprocess(hits)
#         LEAF_SIZE = 50
#         cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=5,cluster_selection_method='leaf',metric='braycurtis',leaf_size=LEAF_SIZE,approx_min_span_tree=False)
#         labels = cl.fit_predict(X) + 1
        
#         return labels


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

    def _init(self, dfh):
        dfh['r'] = np.sqrt(dfh.x**2+dfh.y**2+dfh.z**2)
        dfh['rt'] = np.sqrt(dfh.x**2+dfh.y**2)
        dfh['a0'] = np.arctan2(dfh.y,dfh.x)
        dfh['z1'] = dfh['z']/dfh['rt'] 
        dfh['z2'] = dfh['z']/dfh['r']    
        rr = dfh['rt']/1000
        
        for ii in tqdm(np.arange(-STEPS, STEPS, 1)):
            print ('\r steps: %d '%ii, end='',flush=True)

            dfh['a1'] = dfh['a0'] + (rr + STEPRR*rr**2)*ii/180*np.pi
            
            dfh['x1'] = dfh['a1']/dfh['z1']
            dfh['x2'] = 1/dfh['z1']
            dfh['sina1'] = np.sin(dfh['a1'])
            dfh['cosa1'] = np.cos(dfh['a1'])
            
            ss = StandardScaler()
            
            dfs = ss.fit_transform(dfh[['sina1','cosa1','z1', 'z2']].values)
            
            dfs = np.multiply(dfs, SCALED_DISTANCE)
            self.clusters = DBSCAN(eps=0.0033-ii*STEPEPS,min_samples=1,metric='euclidean', n_jobs=-1).fit(dfs).labels_

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
        return dfh['s1'].values

    def predict(self, hits, cartesian=False): 
        self.clusters = self._init(hits)        
            
        labels = np.unique(self.clusters)
        self._eliminate_outliers(labels)
        if cartesian is True:
            X = self._preprocess(hits) 
            max_len = np.max(self.clusters)
            self.clusters[self.clusters==0] = DBSCAN(eps=0.0075,min_samples=1,algorithm='kd_tree', n_jobs=-1).fit(X[self.clusters==0]).labels_+max_len
        
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

        if False:
            #model = Clusterer()
            #labels = model.predict(hits)
            #xseed_length = 5
            #xmy_volumes = [7, 8, 9]
            #labels = sd.filter_invalid_tracks(labels, hits, xmy_volumes, xseed_length)

            labels = cone.slice_cones(hits, delta_angle=1.0)
            labels = sd.renumber_labels(labels)
            one_submission = create_one_event_submission(event_id, hits, labels)
            score = score_event(truth, one_submission)
            print("Cone slice score for event %d: %.8f" % (event_id, score))

            hits1a = hits.copy(deep=True)
            drop_indices = np.where(labels != 0)[0]
            hits1a = hits1a.drop(hits1a.index[drop_indices])
            #hits1a = hits1a.reset_index(drop=True)

            # Use helix unrolling on the remaining tracks
            model = Clusterer()
            labels1a = model.predict(hits1a)
            #labels1a = cone.slice_cones(hits1a)
            labels1a_x = hack_one_last_run(labels, labels1a, hits1a)
            one_submission = create_one_event_submission(event_id, hits, labels1a_x)
            score = score_event(truth, one_submission)
            print("Score for unroll remainders: %.8f" % (score))
            labels1a[labels1a == 0] = 0 - len(labels) - 1
            labels1a = labels1a + len(labels) + 1

            labels1b = np.copy(labels)
            labels1b[labels1b == 0] = labels1a
            one_submission = create_one_event_submission(event_id, hits, labels1b)
            score = score_event(truth, one_submission)
            print("Merged score for event %d: %.8f" % (event_id, score))

            # ORIG CODE BELOW...

        # Track pattern recognition
        model = Clusterer()
        labels = model.predict(hits)

        # Score for the event
        one_submission = create_one_event_submission(event_id, hits, labels)
   
        score = score_event(truth, one_submission)
        print("Original score for event %d: %.8f" % (event_id, score))

        # Make sure max track ID is not larger than length of labels list.
        labels = sd.renumber_labels(labels)

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
        labels2[labels2 == 0] = 0 - len(labels) - 1
        labels2 = labels2 + len(labels) + 1
        # Expand labels2 to include a zero(0) entry for all hits that were removed
        # labels2_x = hack_one_last_run(labels, labels2, hits2)
        # one_submission = create_one_event_submission(event_id, hits, labels2_x)
        # score = score_event(truth, one_submission)
        # print("Score for unroll 2: %.8f" % (score))

        # # Re-run our clustering algorithm on the remaining hits
        # model2a = DBScanClusterer()
        # labels2a = model2a.predict(hits2)
        # labels2a = labels2a + len(labels) + 1
        # # Expand labels2 to include a zero(0) entry for all hits that were removed
        # labels2a_x = hack_one_last_run(labels, labels2a, hits2)
        # one_submission = create_one_event_submission(event_id, hits, labels2a_x)
        # score = score_event(truth, one_submission)
        # print("Score for dbscan 2: %.8f" % (score))

        # # Re-run our clustering algorithm on the remaining hits
        # model2b = HDBScanClusterer()
        # labels2b = model2b.predict(hits2)
        # labels2b = labels2b + len(labels) + 1
        # # Expand labels2 to include a zero(0) entry for all hits that were removed
        # labels2b_x = hack_one_last_run(labels, labels2b, hits2)
        # one_submission = create_one_event_submission(event_id, hits, labels2b_x)
        # score = score_event(truth, one_submission)
        # print("Score for hdbscan 2: %.8f" % (score))

        # Create final track labels, merging those tracks found in the first and second passes
        labels3 = np.copy(valid_labels)
        labels3[labels3 == 0] = labels2

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels3)
        # Score for the event
        score = score_event(truth, one_submission)
        print("2-pass Score for event %d: %.8f" % (event_id, score))

        for i in range(EXTENSION_ATTEMPT): 
            one_submission = extend(one_submission, hits)
        score = score_event(truth, one_submission)

        print("Add Extension score for event %d: %.8f" % (event_id, score))
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

            # Track pattern recognition 
            model = Clusterer()
            labels = model.predict(hits)

            # Make sure max track ID is not larger than length of labels list.
            labels = sd.renumber_labels(labels)

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
            labels2 = model2.predict(hits2)
            labels2 = labels2 + len(labels) + 1

            labels3 = np.copy(valid_labels)
            labels3[labels3 == 0] = labels2

            # Prepare submission for an event
            one_submission = create_one_event_submission(event_id, hits, labels3)

            for i in range(EXTENSION_ATTEMPT): 
                one_submission = extend(one_submission, hits)
        
            test_dataset_submissions.append(one_submission)
            

        # Create submission file
        submission = pd.concat(test_dataset_submissions, axis=0)
        submission_file = 'submission_' + str(test_skip) + '_' + str(test_events) + '.csv'
        submission.to_csv(submission_file, index=False, header=use_header)

