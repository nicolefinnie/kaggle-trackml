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
        hits['r'] = np.sqrt(hits.x**2+hits.y**2+hits.z**2)
        hits['rt'] = np.sqrt(hits.x**2+hits.y**2)
        hits['a1'] = np.arctan2(hits.y,hits.x)
        hits['z1'] = hits['z']/hits['rt'] 
        hits['z2'] = hits['z']/hits['r']    

        hits['x1'] = hits['a1']/hits['z1']
        hits['x2'] = 1/hits['z1']
        hits['sina1'] = np.sin(hits['a1'])
        hits['cosa1'] = np.cos(hits['a1'])
        
        ss = StandardScaler()
        tmp_scaled = [0.15, 0.15, 0.1, 0.1, 0.1]
        #tmp_scaled = [0.2, 0.2, 0.1, 0.1]
        X = ss.fit_transform(hits[['sina1','cosa1','z1','x1','x2']].values)
        #X = ss.fit_transform(hits[['sina1','cosa1','z1', 'z2']].values)
        
        #X = np.multiply(X, SCALED_DISTANCE)
        X = np.multiply(X, tmp_scaled)

        return X
    
    def predict(self, hits):
        X = self._preprocess(hits)
        labels = DBSCAN(eps=0.0075,min_samples=1,algorithm='kd_tree', n_jobs=-1).fit(X).labels_
        
        return labels

def slice_cones(hits, delta_angle=1.0):
    labels = np.zeros((len(hits)))
    df = hits.copy(deep=True)
    df = df.assign(d = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(r = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(arctan2 = np.arctan2(df.z, df.r))

    for angle in range(-180,180,1):

        #print ('\n %f'%angle, end='',flush=True)
        df1 = df.loc[(df.arctan2>(angle - delta_angle)/180*np.pi) & (df.arctan2<(angle + delta_angle)/180*np.pi)]

        min_num_neighbours = len(df1)
        if min_num_neighbours<4: continue

        df1_indices = df1.index.values
        #print('df1_indices: ' + str(df1_indices))
        labels[df1_indices] = -2

        model = DBScanClusterer()
        tracks = model.predict(df1.copy(deep=True))

        hit_ids = df1.hit_id.values
        x,y,z = df1.as_matrix(columns=['x', 'y', 'z']).T
        r  = (x**2 + y**2)**0.5
        r  = r/1000
        a  = np.arctan2(y,x)
        tree = KDTree(np.column_stack([a,r]), metric='euclidean')

        max_track = np.amax(labels)
        tracks[tracks != 0] = tracks[tracks != 0] + max_track
        track_ids = list(np.unique(tracks))
        num_track_ids = len(track_ids)
        min_length=3
        #print('angle: ' + str(angle) + ', num_hits: ' + str(min_num_neighbours) + ', track_ids: ' + str(track_ids))

        for i in range(num_track_ids):
            p = track_ids[i]
            if p==0: continue

            idx = np.where(tracks==p)[0]
            if len(idx)<min_length:
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
            direction0 = np.arctan2(dr0,da0) 

            da1 = a[idx[-1]] - a[idx[-2]]
            dr1 = r[idx[-1]] - r[idx[-2]]
            direction1 = np.arctan2(dr1,da1) 

        
            ## extend start point
            ns = tree.query([[a0,r0]], k=min(20,min_num_neighbours), return_distance=False)
            ns = np.concatenate(ns)
            direction = np.arctan2(r0-r[ns],a0-a[ns])
            ns = ns[(r0-r[ns]>0.01) &(np.fabs(direction-direction0)<0.04)]

            for n in ns:
                tracks[n] = p
                #df.loc[ df.hit_id==hit_ids[n],'track_id' ] = p 

            ## extend end point
            ns = tree.query([[a1,r1]], k=min(20,min_num_neighbours), return_distance=False)
            ns = np.concatenate(ns)

            direction = np.arctan2(r[ns]-r1,a[ns]-a1)
            ns = ns[(r[ns]-r1>0.01) &(np.fabs(direction-direction1)<0.04)] 
                
            for n in ns:
                tracks[n] = p
                #df.loc[ df.hit_id==hit_ids[n],'track_id' ] = p
        # Now we have tracks[], merge into global labels
        labels[labels == -2] = tracks
    #print ('\n')
    return labels