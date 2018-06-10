import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

RZ_SCALES = [0.4, 1.6, 0.5]
DBSCAN_GLOBAL_EPS = 0.0075
SCALED_DISTANCE = [1, 1, 0.4, 0.4]

class DBScanClusterer(object):
    
    def __init__(self,rz_scales=RZ_SCALES,eps=DBSCAN_GLOBAL_EPS):                        
        self.rz_scales=rz_scales
        self.eps=eps

    def _preprocess(self, hits):
        ss = StandardScaler()
        #tmp_scaled = [0.15, 0.15, 0.1, 0.1, 0.1]
        #X = ss.fit_transform(hits[['sina1','cosa1','z1','x1','x2']].values)
        tmp_scaled = [0.13, 0.13, 0.05, 0.05]
        X = ss.fit_transform(hits[['sina1','cosa1','z1', 'z2']].values)
        
        #X = np.multiply(X, SCALED_DISTANCE)
        X = np.multiply(X, tmp_scaled)

        return X
    
    def predict(self, hits):
        X = self._preprocess(hits)
        labels = DBSCAN(eps=0.0075,min_samples=1,algorithm='kd_tree', n_jobs=-1).fit(X).labels_
        
        return labels

def do_one_slice(df, labels, min_track_length, max_track_length, angle, delta_angle):
    # Perform slice on input hits
    df1 = df.loc[(df.arctan2>(angle - delta_angle)/180*np.pi) & (df.arctan2<(angle + delta_angle)/180*np.pi)]

    min_num_neighbours = len(df1)
    if min_num_neighbours<4:
        return labels

    df1_indices = df1.index.values

    model = DBScanClusterer()
    tracks = model.predict(df1.copy(deep=True))

    x,y,z = df1.as_matrix(columns=['x', 'y', 'z']).T
    r  = (x**2 + y**2)**0.5
    r  = r/1000
    a  = np.arctan2(y,x)
    tree = KDTree(np.column_stack([a,r]), metric='euclidean')

    max_track = np.amax(labels)
    tracks[tracks != 0] = tracks[tracks != 0] + max_track
    track_ids = list(np.unique(tracks))
    num_track_ids = len(track_ids)
    min_length_after_extension=2
    max_length_after_extension=30

    for i in range(num_track_ids):
        p = track_ids[i]
        if p==0: continue

        idx = np.where(tracks==p)[0]
        if len(idx)<min_track_length:
            tracks[tracks == p] = 0
            continue
        if len(idx)>max_track_length:
            tracks[tracks == p] = 0
            continue

        if angle>0:
            idx = idx[np.argsort( z[idx])]
        else:
            idx = idx[np.argsort(-z[idx])]

        ## start and end points  ##
        idx0,idx1 = idx[0],idx[-1]
        a0 = a[idx0]
        a1 = a[idx1]
        r0 = r[idx0]
        r1 = r[idx1]

        da0 = a[idx[1]] - a[idx[0]]  #direction
        dr0 = r[idx[1]] - r[idx[0]]
        divisor0 = (da0**2+dr0**2)**0.5
        if divisor0 == 0 : divisor0 = 1
        direction0 = np.array([da0/divisor0,dr0/divisor0])  

        da1 = a[idx[-1]] - a[idx[-2]]
        dr1 = r[idx[-1]] - r[idx[-2]]
        divisor1 = (da1**2+dr1**2)**0.5
        if divisor1 == 0 : divisor1 = 1
        direction1 = np.array([da1/divisor1,dr1/divisor1]) 

        ## extend start point
        ns = tree.query([[a0,r0]], k=min(20,min_num_neighbours), return_distance=False)
        ns = np.concatenate(ns)
        
        da0ns = a0-a[ns]
        dr0ns = r0-r[ns]
        divisor0ns = (da0ns**2+dr0ns**2)**0.5
        divisor0ns[divisor0ns == 0] = 1
        direction = np.array([da0ns/divisor0ns,dr0ns/divisor0ns]) 

        ns = ns[(r0-r[ns]>0.01) & (np.matmul(direction.T,direction0)>0.9991)]

        for n in ns:
            tracks[n] = p

        ## extend end point
        ns = tree.query([[a1,r1]], k=min(20,min_num_neighbours), return_distance=False)
        ns = np.concatenate(ns)
        da1ns = a[ns]-a1
        dr1ns = r[ns]-r1

        divisor1ns = (da1ns**2+dr1ns**2)**0.5
        divisor1ns[divisor1ns == 0] = 1
        direction = np.array([da1ns/divisor1ns,dr1ns/divisor1ns]) 

        ns = ns[(r[ns]-r1>0.01) & (np.matmul(direction.T,direction1)>0.9991)] 

        for n in ns:
            tracks[n] = p

        if len(idx)<min_length_after_extension:
            tracks[tracks == p] = 0
        if len(idx)>max_length_after_extension:
            tracks[tracks == p] = 0

    # Now we have tracks[], merge into global labels
    # Simple merging - the winner is the longer track
    trk_ix = 0
    for ix in df1_indices:
        if labels[ix] == 0:
            labels[ix] = tracks[trk_ix]
        else:
            w1_track = labels[ix]
            w2_track = tracks[trk_ix]
            w1 = np.where(labels == w1_track)[0]
            w2 = np.where(tracks == w2_track)[0]
            if len(w2) > len(w1):
                labels[ix] = tracks[trk_ix]
            #print('OLD len: ' + str(len(w1)) + ', NEW len: ' + str(len(w2)))
        trk_ix = trk_ix + 1

    return labels

def slice_cones(hits, min_track_length, max_track_length, do_swap=False, delta_angle=1.0):
    labels = np.zeros((len(hits)))
    df = hits.copy(deep=True)
    if do_swap:
        df = df.assign(xtmp = df.y)
        df = df.assign(y = -df.x)
        df = df.assign(x = df.xtmp)
    df = df.assign(d = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(r = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(arctan2 = np.arctan2(df.z, df.r))
    df = df.assign(a1 = np.arctan2(df.y, df.x))
    df = df.assign(z1 = df.z / df.r)
    df = df.assign(z2 = df.z / df.d)
    df = df.assign(x1 = df.a1 / df.z1)
    df = df.assign(x2 = 1 / df.z1)
    df = df.assign(sina1 = np.sin(df.a1))
    df = df.assign(cosa1 = np.cos(df.a1))

    uniq_list = np.unique(df.arctan2.values)
    #print('uniq arctan head: ' + str(uniq_list[0:5]) + ', tail: ' + str(uniq_list[-5:]))
    for angle in range(-180,180,1):
        df1 = df.loc[(df.arctan2>(angle - delta_angle)/180*np.pi) & (df.arctan2<(angle + delta_angle)/180*np.pi)]
        num_hits = len(df1)
        # Dynamically adjust the delta based on how many hits are found
        if num_hits > 1000:
            labels = do_one_slice(df, labels, min_track_length, max_track_length, angle-0.6, 0.4)
            labels = do_one_slice(df, labels, min_track_length, max_track_length, angle-0.2, 0.4)
            labels = do_one_slice(df, labels, min_track_length, max_track_length, angle+0.2, 0.4)
            labels = do_one_slice(df, labels, min_track_length, max_track_length, angle+0.6, 0.4)
        else:
            labels = do_one_slice(df, labels, min_track_length, max_track_length, angle, 1.0)
        #labels = do_one_slice(df, labels, min_track_length, max_track_length, angle, 1.0)

    labels = np.asarray([int(i) for i in labels])
    return labels