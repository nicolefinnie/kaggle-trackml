from trackml.dataset import load_event
from trackml.score import score_event
#from IPython.display import display
import tensorflow as tf
import pandas as pd
import numpy as np
import glob

train = np.unique([p.split('-')[0] for p in sorted(glob.glob('../input/train_1/**'))])
test = np.unique([p.split('-')[0] for p in sorted(glob.glob('../input/test/**'))])
#det = pd.read_csv('../input/detectors.csv')
sub = pd.read_csv('../input/sample_submission.csv')
#print(len(train), len(test), len(det), len(sub))

#for e in train:
#    hits, cells, particles, truth = load_event(e)
#    print(len(hits), len(cells), len(particles), len(truth))
#    display(hits.head(2))
#    display(cells.head(2))
#    display(particles.head(2))
#    display(truth.head(2))
#    break

#for e in test:
#    hits, cells = load_event(e, parts=['hits', 'cells'])
#    print(len(hits), len(cells))
#    display(hits.head(2))
#    display(cells.head(2))
#    break

from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.neighbors import NearestNeighbors
from scipy import stats
"""
updated - added self.rz_scale
"""
class Clusterer(object):
    
    def __init__(self,rz_scales=[0.65, 0.965, 1.428]):                        
        self.rz_scales=rz_scales
    
    def _eliminate_outliers(self,labels,M):
        norms=np.zeros((len(labels)),np.float32)
        indices=np.zeros((len(labels)),np.int32)
        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if cluster == 0:
                continue
            index = np.argwhere(self.clusters==cluster)
            x = M[index]
            indices[i] = len(index)
            local_mask = np.ones((len(index)),np.bool)
            norms[i] = self._test_quadric(x)
        threshold = np.percentile(norms,90)
        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if norms[i] > threshold:
                self.clusters[self.clusters==cluster]=0    
                
    def _test_quadric(self,x):
        if len(x.shape)==3:
            x = np.reshape(x,(x.shape[0],3))
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
    
    def _preprocess(self, hits, rz_scales):
        
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
        for i, rz_scale in enumerate(rz_scales):
            X[:,i] = X[:,i] * rz_scale
        
        return X   
    def predict(self, hits, rz_scales):        
        volumes = np.unique(hits['volume_id'].values)
        X = self._preprocess(hits, rz_scales)
        self.clusters = np.zeros((len(X),1),np.int32)
        max_len = 1
        cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,
                             metric='braycurtis',cluster_selection_method='leaf',algorithm='best', leaf_size=50)
        self.clusters = cl.fit_predict(X)+1
        labels = np.unique(self.clusters)
        n_labels = 0
        while n_labels < len(labels):
            n_labels = len(labels)
            self._eliminate_outliers(labels,X)
            max_len = np.max(self.clusters)
            self.clusters[self.clusters==0] = cl.fit_predict(X[self.clusters==0])+max_len
            labels = np.unique(self.clusters)
        return self.clusters

import xgboost as xgb
from sklearn import *
import hdbscan
from tqdm import tqdm

scl = preprocessing.StandardScaler()
#dbscan = cluster.DBSCAN(eps=0.007555, min_samples=1, algorithm='kd_tree', n_jobs=-1)
dbscan = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=7, cluster_selection_method='leaf', prediction_data=False, metric='braycurtis')

def norm_points(df, g=0.7):
    x = df.x.values
    y = df.y.values
    z = df.z.values
    r = np.sqrt(x**2 + y**2 + z**2)
    df['x2'] = x/r
    df['y2'] = y/r
    r = np.sqrt(x**2 + y**2)
    df['z2'] = (z/r) * g
    
    df['r'] = r
    #df['r2'] = np.sqrt(x**2 + y**2)
    #df['r3'] = np.sqrt(x**2 + z**2)
    #df['r4'] = np.sqrt(y**2 + z**2)
    #df['phi'] = np.arctan2(y, x)
    #df['phi2'] = np.arctan2(y, z)
    #df['phi3'] = np.arctan2(x, z)
    #df['hm'] = (2. * np.cos(df['phi'] - g) / df['r2']).values
    return df

limit = 0
df_train = []
df_val = []
for e in train:
    hits, cells, truth = load_event(e, parts=['hits', 'cells', 'truth'])
    hits['event_id'] = int(e[-9:])
    cells = cells.groupby(by=['hit_id'])['ch0', 'ch1', 'value'].agg(['mean','median','max','min','sum']).reset_index()
    cells.columns = ['hit_id'] + ['-'.join([c2,c1]) for c1 in ['mean','median','max','min','sum'] for c2 in ['ch0', 'ch1', 'value']]
    hits = pd.merge(hits, cells, how='left', on='hit_id')
    tcols = list(truth.columns)
    hits = pd.merge(hits, truth, how='left', on='hit_id')
    hits = norm_points(hits)
    cols = [c for c in hits.columns if c not in ['event_id', 'hit_id', 'particle_id']+tcols]

    #Classification
    hits['target'] = hits['particle_id'].map(lambda x: 0 if x==0 else 1)
    #tree = neighbors.KDTree(hits[['x2', 'y2', 'z2']].values)
    #hits['dist_kd'] = hits.apply(lambda r: tree.query([[r['x2'], r['y2'], r['z2']]], k=1)[0][0][0], axis=1)
    #hits['n_in_r_kd'] = hits.apply(lambda r: tree.query_radius([[r['x2'], r['y2'], r['z2']]], r=0.3, count_only=True)[0], axis=1)
    #tree = neighbors.BallTree(hits[['x2', 'y2', 'z2']].values)
    #hits['dist_ball'] = hits.apply(lambda r: tree.query([[r['x2'], r['y2'], r['z2']]], k=1)[0][0][0], axis=1)
    #hits['n_in_r_ball'] = hits.apply(lambda r: tree.query_radius([[r['x2'], r['y2'], r['z2']]], r=0.3, count_only=True)[0], axis=1)
    x1, x2, y1, y2 = model_selection.train_test_split(hits[cols], hits['target'].values, test_size=0.20, random_state=19)

    train_ = x1.copy()
    train_['target'] = y1
    df_train.append(train_.copy())
    
    val = x2.copy()
    val['target'] = y2
    df_val.append(val.copy())
    
    params = {'eta': 0.2, 'max_depth': 7, 'objective':'binary:logistic', 'seed': 19, 'silent': True}
    
    if limit == 0:
        watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
        model = xgb.train(params, xgb.DMatrix(x1, y1), 500,  watchlist, verbose_eval=10, early_stopping_rounds=20)
    else:
        train_ = pd.concat(df_train, ignore_index=True).reset_index(drop=True)
        val = pd.concat(df_val, ignore_index=True).reset_index(drop=True)
        watchlist = [(xgb.DMatrix(train_[cols], train_['target'].values), 'train'), (xgb.DMatrix(val[cols], val['target'].values), 'valid')]
        #model.load_model('tml_xgb_model.model')
        #model = xgb.train(params, xgb.DMatrix(train_[cols], train_['target'].values), 500,  watchlist, verbose_eval=10, early_stopping_rounds=20, xgb_model=model)
        model = xgb.train(params, xgb.DMatrix(train_[cols], train_['target'].values), 500,  watchlist, verbose_eval=10, early_stopping_rounds=20)
    #model.save_model('tml_xgb_model.model')
    
    preds = model.predict(xgb.DMatrix(hits[cols]), ntree_limit=model.best_ntree_limit)
    hits['track_id'] = [1 if p>0.2 else 0 for p in preds]

    #Add TensorFlow reinforcement learning here for N cluster optimizer...
    
    if len(hits[hits['track_id']==0])>0:
        hits0 = hits[hits['track_id']==0].reset_index(drop=True)
        hits1 = hits[hits['track_id']==1].reset_index(drop=True)
        print(len(hits0), len(hits1))
        hits1['track_id'] = dbscan.fit_predict(scl.fit_transform(hits1[['x2', 'y2', 'z2']].values)) + 1
        hits = pd.concat([hits0, hits1], ignore_index=True).fillna(0).reset_index(drop=True)
    else:
        hits['track_id'] = dbscan.fit_predict(scl.fit_transform(hits[['x2', 'y2', 'z2']].values)) + 1

    score = score_event(hits[tcols], hits[['event_id','hit_id','track_id']])
    print(e, len(hits), len(truth['particle_id'].unique()), len(hits['track_id'].unique()), score)
    if limit > 5:
        break
    limit += 1

RZ_SCALE = [0.65, 0.965, 1.428]
LEAF_SIZE = 50

df_test = []
for e in test:
    hits, cells = load_event(e, parts=['hits', 'cells'])
    hits['event_id'] = int(e[-9:])
    cells = cells.groupby(by=['hit_id'])['ch0', 'ch1', 'value'].agg(['mean','median','max','min','sum']).reset_index()
    cells.columns = ['hit_id'] + ['-'.join([c2,c1]) for c1 in ['mean','median','max','min','sum'] for c2 in ['ch0', 'ch1', 'value']]
    hits = pd.merge(hits, cells, how='left', on='hit_id')
    col = [c for c in hits.columns if c not in ['event_id', 'hit_id', 'particle_id']]
    hits = norm_points(hits)

    #Classification
    #tree = neighbors.KDTree(hits[['x2', 'y2', 'z2']].values)
    #hits['dist_kd'] = hits.apply(lambda r: tree.query([[r['x2'], r['y2'], r['z2']]], k=1)[0][0][0], axis=1)
    #hits['n_in_r_kd'] = hits.apply(lambda r: tree.query_radius([[r['x2'], r['y2'], r['z2']]], r=0.3, count_only=True)[0], axis=1)
    #tree = neighbors.BallTree(hits[['x2', 'y2', 'z2']].values)
    #hits['dist_ball'] = hits.apply(lambda r: tree.query([[r['x2'], r['y2'], r['z2']]], k=1)[0][0][0], axis=1)
    #hits['n_in_r_ball'] = hits.apply(lambda r: tree.query_radius([[r['x2'], r['y2'], r['z2']]], r=0.3, count_only=True)[0], axis=1)
    preds = model.predict(xgb.DMatrix(hits[cols]), ntree_limit=model.best_ntree_limit)
    hits['particle_id'] = [1 if p>0.2 else 0 for p in preds]

    if len(hits[hits['particle_id']==0])>0:
        hits0 = hits[hits['particle_id']==0].reset_index(drop=True)
        hits1 = hits[hits['particle_id']==1].reset_index(drop=True)
        # Track pattern recognition
        try:
            model_cluster = Clusterer()
            labels = model_cluster.predict(hits1, RZ_SCALE)
            hits1['particle_id'] = labels + 1
        except:
            pass
#         hits1['particle_id'] = dbscan.fit_predict(scl.fit_transform(hits1[['x2', 'y2', 'z2']].values)) + 1
        hits = pd.concat([hits0, hits1], ignore_index=True).fillna(0).reset_index(drop=True)
    else:
        try:
            model_cluster = Clusterer()
            labels = model_cluster.predict(hits, RZ_SCALE)
            hits['particle_id'] = labels + 1
        except:
            pass
#         hits1['particle_id'] = labels + 1  #seems fishy !!!
          
#         hits['particle_id'] = dbscan.fit_predict(scl.fit_transform(hits[['x2', 'y2', 'z2']].values)) + 1
    df_test.append(hits[['event_id','hit_id','particle_id']].copy())
    print(e, len(hits['particle_id'].unique()))

df_test = pd.concat(df_test, ignore_index=True)
df_test.head()

sub = pd.merge(sub, df_test, how='left', on=['event_id','hit_id'])
sub['track_id'] = sub['particle_id'].astype(int)
sub[['event_id','hit_id','track_id']].to_csv('submission-001.csv', index=False)
