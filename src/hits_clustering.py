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

RZ_SCALES = [0.4, 1.6, 0.5]
#SCALED_DISTANCE = [1.0, 1.0, 0.7, 0.025, 0.025]
SCALED_DISTANCE = [1, 1, 0.4, 0.4]

DZ = 0.000015
STEPDZ = 0.0000002
STEPEPS = 0.000002  
STEPS = 200
THRESHOLD_MIN = 5
THRESHOLD_MAX = 30


print('rz scales: ' + str(RZ_SCALES))
print('scaled distance: ' + str(SCALED_DISTANCE))
print('dz: ' + str(DZ))
print('stepdz: ' + str(STEPDZ))
print('stepeps: ' + str(STEPEPS))
print('steps: ' + str(STEPS))
print('threshold min: ' + str(THRESHOLD_MIN))
print('threshold max: ' + str(THRESHOLD_MAX))


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
        dz = DZ
        
        for ii in tqdm(range(STEPS)):
            dz = dz + ii*STEPDZ 
            dfh['a1'] = dfh['a0']+dz*dfh['z']*np.sign(dfh['z'].values)
            dfh['x1'] = dfh['a1']/dfh['z1']
            dfh['x2'] = 1/dfh['z1']
            dfh['sina1'] = np.sin(dfh['a1'])
            dfh['cosa1'] = np.cos(dfh['a1'])
            
            ss = StandardScaler()
            
            #dfs = ss.fit_transform(dfh[['sina1','cosa1','z1','x1','x2']].values)
            dfs = ss.fit_transform(dfh[['sina1','cosa1','z1', 'z2']].values)
            
            dfs = np.multiply(dfs, SCALED_DISTANCE)
            self.clusters = DBSCAN(eps=0.0033-ii*STEPEPS,min_samples=1,metric='euclidean', n_jobs=-1).fit(dfs).labels_

            if ii==0:
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

        dz = DZ
        for ii in tqdm(range(STEPS)):
            dz = dz - ii*STEPDZ
            dfh['a1'] = dfh['a0']+dz*dfh['z']*np.sign(dfh['z'].values)
            dfh['x1'] = dfh['a1']/dfh['z1']
            dfh['x2'] = 1/dfh['z1']
            dfh['sina1'] = np.sin(dfh['a1'])
            dfh['cosa1'] = np.cos(dfh['a1'])
            
            ss = StandardScaler()
            #dfs = ss.fit_transform(dfh[['sina1','cosa1','z1','x1','x2']].values)
            dfs = ss.fit_transform(dfh[['sina1','cosa1','z1', 'z2']].values)
            dfs = np.multiply(dfs, SCALED_DISTANCE)
            self.clusters = DBSCAN(eps=0.0033-ii*STEPEPS,min_samples=1,metric='euclidean', n_jobs=-1).fit(dfs).labels_

            dfh['s2'] = self.clusters
            dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
            maxs1 = dfh['s1'].max()
     
            cond = np.where(dfh['N2'].values>dfh['N1'].values  )
            s1 = dfh['s1'].values
            s1[cond] = dfh['s2'].values[cond]+maxs1
            dfh['s1'] = s1
            dfh['s1'] = dfh['s1'].astype('int64')
            dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        return dfh['s1'].values

    def predict(self, hits): 
        self.clusters = self._init(hits)        
        X = self._preprocess(hits) 
            
        labels = np.unique(self.clusters)
        self._eliminate_outliers(labels)
        max_len = np.max(self.clusters)
        self.clusters[self.clusters==0] = DBSCAN(eps=0.0075,min_samples=1,algorithm='kd_tree', n_jobs=8).fit(X[self.clusters==0]).labels_+max_len
               
        return self.clusters

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission


def run_single_threaded_training(skip, nevents):
    path_to_train = "../input/train_1"
    dataset_submissions = []
    dataset_scores = []

    for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=skip, nevents=nevents):
        # Track pattern recognition
        model = Clusterer()
        labels = model.predict(hits)

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels)
        dataset_submissions.append(one_submission)

        # Score for the event
        score = score_event(truth, one_submission)
        dataset_scores.append(score)

        print("Score for event %d: %.8f" % (event_id, score))

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

            # Prepare submission for an event
            one_submission = create_one_event_submission(event_id, hits, labels)
            test_dataset_submissions.append(one_submission)
            

        # Create submission file
        submission = pd.concat(test_dataset_submissions, axis=0)
        submission_file = 'submission_' + str(test_skip) + '_' + str(test_events) + '.csv'
        submission.to_csv(submission_file, index=False, header=use_header)

