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

class Clusterer(object):
    def __init__(self,rz_scales=[0.4, 1.6, 0.5]):                        
        self.rz_scales=rz_scales

    def _eliminate_outliers(self,labels,M):
        #norms=np.zeros((len(labels)),np.float32)
        indices=np.zeros((len(labels)),np.float32)
 
        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if cluster == 0:
                continue
            index = np.argwhere(self.clusters==cluster)
            index = np.reshape(index,(index.shape[0]))
            indices[i] = len(index)
            #x = M[index]
            #norms[i] = self._test_quadric(x)
        #threshold1 = np.percentile(norms,90)*5
        threshold2 = 25
        threshold3 = 7
        for i, cluster in enumerate(labels):
            #if norms[i] > threshold1 or indices[i] > threshold2 or indices[i] < threshold3:
            if indices[i] > threshold2 or indices[i] < threshold3:
             
                self.clusters[self.clusters==cluster]=0  

    # def _test_quadric(self,x):
    #     Z = np.zeros((x.shape[0],10), np.float32)
    #     Z[:,0] = x[:,0]**2
    #     Z[:,1] = 2*x[:,0]*x[:,1]
    #     Z[:,2] = 2*x[:,0]*x[:,2]
    #     Z[:,3] = 2*x[:,0]
    #     Z[:,4] = x[:,1]**2
    #     Z[:,5] = 2*x[:,1]*x[:,2]
    #     Z[:,6] = 2*x[:,1]
    #     Z[:,7] = x[:,2]**2
    #     Z[:,8] = 2*x[:,2]
    #     Z[:,9] = 1
    #     v, s, t = np.linalg.svd(Z,full_matrices=False)        
    #     smallest_index = np.argmin(np.array(s))
    #     T = np.array(t)
    #     T = T[smallest_index,:]        
    #     norm = np.linalg.norm(np.dot(Z,T), ord=2)**2
    #     return norm

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
        dfh['rt'] = np.sqrt(dfh.x**2+dfh.y**2)
        dfh['a0'] = np.arctan2(dfh.y,dfh.x)
        dfh['z1'] = dfh['z']/dfh['rt']        
        dz = 0.000015
        stepdz = 0.00000025
        stepeps = 0.000002
        steps = 150
        
        for ii in tqdm(range(steps)):
            dz = dz + ii*stepdz 
            dfh['a1'] = dfh['a0']+dz*dfh['z']*np.sign(dfh['z'].values)
            dfh['x1'] = dfh['a1']/dfh['z1']
            dfh['x2'] = 1/dfh['z1']
            dfh['sina1'] = np.sin(dfh['a1'])
            dfh['cosa1'] = np.cos(dfh['a1'])
            
            ss = StandardScaler()
            c = [0.85, 0.85, 0.75, 0.5, 0.5]
            
            dfs = ss.fit_transform(dfh[['sina1','cosa1','z1','x1','x2']].values)
            dfs = np.multiply(dfs, c)
            self.clusters = DBSCAN(eps=0.0033-ii*stepeps,min_samples=1,metric='euclidean', n_jobs=-1).fit(dfs).labels_

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
        dz = 0.000015
        stepdz = -0.00000025
        stepeps = -0.000002
        for ii in tqdm(range(steps)):
            dz = dz + ii*stepdz
            dfh['a1'] = dfh['a0']+dz*dfh['z']*np.sign(dfh['z'].values)
            dfh['x1'] = dfh['a1']/dfh['z1']
            dfh['x2'] = 1/dfh['z1']
            dfh['sina1'] = np.sin(dfh['a1'])
            dfh['cosa1'] = np.cos(dfh['a1'])
            
            ss = StandardScaler()
            dfs = ss.fit_transform(dfh[['sina1','cosa1','z1','x1','x2']].values)
            dfs = np.multiply(dfs, c)
            self.clusters = DBSCAN(eps=0.0033+ii*stepeps,min_samples=1,metric='euclidean', n_jobs=-1).fit(dfs).labels_


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
               
        # cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,
        #                      metric='braycurtis',cluster_selection_method='leaf',algorithm='best', leaf_size=50, approx_min_span_tree=False)
        
        labels = np.unique(self.clusters)
        
        n_labels = 0
        while n_labels < len(labels):
            n_labels = len(labels)
            self._eliminate_outliers(labels,X)
            max_len = np.max(self.clusters)
            #self.clusters[self.clusters==0] = cl.fit_predict(X[self.clusters==0])+max_len
            self.clusters[self.clusters==0]  = DBSCAN(eps=0.0075,min_samples=1,algorithm='kd_tree', n_jobs=8).fit(X[self.clusters==0]).labels_+max_len

            labels = np.unique(self.clusters)
        return self.clusters

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

def predict_create_submission_in_batch(events_id_list, hits_list, truth_list):
    num_threads = len(events_id_list)
    print('num_threads: ' + str(num_threads))
    submissions = []
    scores = []
    result_submissions = [None]*num_threads
    result_scores = [None]*num_threads
    models = [None]*num_threads
    labels = [None]*num_threads
    print('Using ' + str(num_threads) + ' Threads')

    def parallel_thread(event_id, hits, truth, result_submissions, result_scores, models, labels, index):
        print('Starting the worker of the event ' + str(event_id) + ' with index ' + str(index))
        models[index] = Clusterer()
        # a list of labels 
        labels[index] = models[index].predict(hits)
        print('Finished the worker of the event ' + str(event_id) + ' with index ' + str(index))

        # Prepare submission for an event
        print('Creating submission of the event ' + str(event_id) + ' with index ' + str(index))
        result_submissions[index] = create_one_event_submission(event_id, hits, labels[index])

        # Score for the event
        if truth is not None:
            result_scores[index] = score_event(truth, result_submissions[index])

        return True
        
    threads = []

    for ii in range(num_threads):
        #event_id, hits, _, _, truth = load_dataset(path_to_train, skip=ii, nevents=1)
        thread = thr.Thread(target=parallel_thread, args=[events_id_list[ii], hits_list[ii], truth_list[ii], result_submissions, result_scores, models, labels, ii])
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    
    for ii in range(num_threads):
        submissions.extend(result_submissions[ii])
        scores.extend(result_scores[ii])
    return submissions, scores


def run_training_in_threads():
    event_id_list = []
    hits_list = []
    truth_list = []
    # Change this according to your directory preferred setting
    path_to_train = "../input/train_1"

    for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=0, nevents=5):
       event_id_list.append(event_id)
       hits_list.append(hits)
       truth_list.append(truth)

    submissions, scores = predict_create_submission_in_batch(event_id_list, hits_list, truth_list)
    for i in range(len(submissions)):
       #print(submissions[i])
       #print(scores[i])
       print("Score for event %d: %.8f" % (event_id_list[i], scores[i]))

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

