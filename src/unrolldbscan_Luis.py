import numpy as np

x = np.array([(62,397,103),(82,347,107),(93,288,120),
     (94,266,128),(65,163,169),(12,102,198),
     (48,138,180),(77,187,157),(85,209,149),(89,316,112)])

xm = np.mean(x)

X = (x-xm).T


v, s, t = np.linalg.svd(X,full_matrices=True)


sigma1 = s[0]
sigma2 = s[1]
sigma3 = s[2]
v1 = t[0]
v2 = t[1]
v3 = t[2]

Z = np.zeros((x.shape[0],10), np.float32)
Z[:,0] = x[:,0]**2
Z[:,1] = 2*x[:,0]*x[:,1]
Z[:,2] = 2*x[:,0]*x[:,2]
Z[:,3] = 2*x[:,0]
Z[:,4] = x[:,1]**2
Z[:,5] = 2*x[:,1]*x[:,2]
Z[:,6] = 2*x[:,1]
Z[:,7] = x[:,2]**2
Z[:,8] = 2*x[:,2]
Z[:,9] = 1

v, s, t = np.linalg.svd(Z,full_matrices=True)
smallest_value = np.min(np.array(s))
smallest_index = np.argmin(np.array(s))
T = np.array(t)
T = T[smallest_index,:]
S = np.zeros((4,4),np.float32)
S[0,0] = T[0]
S[0,1] = S[1,0] = T[1]
S[0,2] = S[2,0] = T[2]
S[0,3] = S[3,0] = T[3]
S[1,1] = T[4]
S[1,2] = S[2,1] = T[5]
S[1,3] = S[3,1] = T[6]
S[2,2] = T[7]
S[2,3] = S[3,2] = T[8]
S[3,3] = T[9]
norm = np.linalg.norm(np.dot(Z,T), ord=2)**2
print(norm)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event


# Change this according to your directory preferred setting
path_to_train = "../input/train_1"


# This event is in Train_1
event_prefix = "event000001000"

hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))


from sklearn.preprocessing import StandardScaler
import hdbscan
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN

class Clusterer(object):
    def __init__(self,rz_scales=[0.69, 0.965, 1.528]):                        
        self.rz_scales=rz_scales
    
    def _eliminate_outliers(self,labels,M):
        norms=np.zeros((len(labels)),np.float32)
        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if cluster == 0:
                continue
            index = np.argwhere(self.clusters==cluster)
            index = np.reshape(index,(index.shape[0]))
            if len(index)>20 or len(index)<7:
                self.clusters[self.clusters==cluster]=0
            x = M[index]
            norms[i] = self._test_quadric(x)
        threshold = np.percentile(norms,90)*5
        for i, cluster in enumerate(labels):
            if norms[i] > threshold:
                self.clusters[self.clusters==cluster]=0            
    def _test_quadric(self,x):
        Z = np.zeros((x.shape[0],10), np.float32)
        Z[:,0] = x[:,0]**2
        Z[:,1] = 2*x[:,0]*x[:,1]
        Z[:,2] = 2*x[:,0]*x[:,2]
        Z[:,3] = 2*x[:,0]
        Z[:,4] = x[:,1]**2
        Z[:,5] = 2*x[:,1]*x[:,2]
        Z[:,6] = 2*x[:,1]
        Z[:,7] = x[:,2]**2
        Z[:,8] = 2*x[:,2]
        Z[:,9] = 1
        v, s, t = np.linalg.svd(Z,full_matrices=False)        
        smallest_index = np.argmin(np.array(s))
        T = np.array(t)
        T = T[smallest_index,:]        
        norm = np.linalg.norm(np.dot(Z,T), ord=2)**2
        return norm

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
        dfh['r2'] = np.sqrt(dfh.x**2+dfh.y**2)
        dfh['z1'] = dfh['z']/dfh['rt']        
        dz = 0.00012
        stepdz = 0.000005
        for ii in tqdm(range(24)):
            dz = dz + ii*stepdz
            dfh['a1'] = dfh['a0']+dz*dfh['z']*np.sign(dfh['z'].values)
            dfh['x1'] = dfh['a1']/dfh['z1']
            dfh['x2'] = 1/dfh['z1']
            dfh['x3'] = dfh['x1']+dfh['x2']
            ss = StandardScaler()
            dfs = ss.fit_transform(dfh[['a1','z1','x1','x2','x3']].values)
            self.clusters = DBSCAN(eps=0.0035-dz,min_samples=1,metric='euclidean').fit(dfs).labels_
            if ii==0:
                dfh['s1']=self.clusters
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
            else:
                dfh['s2'] = self.clusters
                dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
                maxs1 = dfh['s1'].max()
                cond = np.where(dfh['N2'].values>dfh['N1'].values)
                s1 = dfh['s1'].values
                s1[cond] = dfh['s2'].values[cond]+maxs1
                dfh['s1'] = s1
                dfh['s1'] = dfh['s1'].astype('int64')
                self.clusters = dfh['s1'].values
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        dz = 0.00012
        stepdz = -0.000005
        for ii in tqdm(range(24)):
            dz = dz + ii*stepdz
            dfh['a1'] = dfh['a0']+dz*dfh['z']*np.sign(dfh['z'].values)
            dfh['x1'] = dfh['a1']/dfh['z1']
            dfh['x2'] = 1/dfh['z1']
            dfh['x3'] = dfh['x1']+dfh['x2']
            ss = StandardScaler()
            dfs = ss.fit_transform(dfh[['a1','z1','x1','x2','x3']].values)
            self.clusters = DBSCAN(eps=0.0035+dz,min_samples=1,metric='euclidean').fit(dfs).labels_
            dfh['s2'] = self.clusters
            dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
            maxs1 = dfh['s1'].max()
            cond = np.where(dfh['N2'].values>dfh['N1'].values)
            s1 = dfh['s1'].values
            s1[cond] = dfh['s2'].values[cond]+maxs1
            dfh['s1'] = s1
            dfh['s1'] = dfh['s1'].astype('int64')
            dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        return dfh['s1'].values
    def predict(self, hits): 
        self.clusters = self._init(hits)        
        X = self._preprocess(hits) 
        
        cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,
                             metric='braycurtis',cluster_selection_method='leaf',algorithm='best', leaf_size=50)
        
        labels = np.unique(self.clusters)
        n_labels = 0
        while n_labels < len(labels):
            n_labels = len(labels)
            self._eliminate_outliers(labels,X)
            max_len = np.max(self.clusters)
            self.clusters[self.clusters==0] = cl.fit_predict(X[self.clusters==0])+max_len
            labels = np.unique(self.clusters)
        return self.clusters

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

dataset_submissions = []
dataset_scores = []

for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=0, nevents=5):
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


path_to_test = "../input/test"
test_dataset_submissions = []

create_submission = False # True for submission 
if create_submission:
    for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

        # Track pattern recognition 
        model = Clusterer()
        labels = model.predict(hits)

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels)
        test_dataset_submissions.append(one_submission)
        
        print('Event ID: ', event_id)

    # Create submission file
    submission = pd.concat(test_dataset_submissions, axis=0)
    submission.to_csv('submission.csv', index=False)

