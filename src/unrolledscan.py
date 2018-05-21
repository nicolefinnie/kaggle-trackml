import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import hdbscan

class Clusterer(object):
    
    def __init__(self, eps):
        self.eps = eps
        
    
    def _preprocess(self, hits):
        
        x = hits.hx.values
        y = hits.hy.values
        z = hits.hz.values

        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r

        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        
        return X
    
    
    def predict(self, hits):
        
        X = self._preprocess(hits)
        
        cl = DBSCAN(eps=self.eps, min_samples=3, algorithm='kd_tree')
        labels = cl.fit_predict(X)

        #cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,cluster_selection_method='leaf',metric='braycurtis',leaf_size=50,approx_min_span_tree=False)
        #labels = cl.fit_predict(X) + 1
        
        return labels

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

# Heng's code to unroll helixes
input_path = '../input'
train_path = os.path.join(input_path, 'train_100_events')

event = '000001029'
event_prefix = 'event' + event

hits, cells, particles, truth = load_event(os.path.join(train_path, event_prefix))
hits['hx'] = hits['x'].map(lambda x: x)
hits['hy'] = hits['y'].map(lambda y: y)
hits['hz'] = hits['z'].map(lambda z: z)

truth = truth.merge(hits, on=['hit_id'], how='left')
truth = truth.merge(particles, on=['particle_id'], how='left')

df = truth
df['momentum'] = np.sqrt(df.px**2 + df.py**2 + df.pz**2)
df.loc[df.particle_id==0,'momentum']=0

def unroll_helixes(df, append_df, x_factor, y_factor, plot_it=False):
    particle_ids = list(df.particle_id.unique())
    num_particle_ids = len(particle_ids)

    if plot_it:
        fig1 = plt.figure(figsize=(30, 10))
        ax1 = fig1.add_subplot(122, projection='3d')
        ax2 = fig1.add_subplot(121, projection='3d')

    for j in range(num_particle_ids):
        #print(particle_ids[j])
        dfslice = df.loc[df.particle_id==particle_ids[j]].copy(deep=True)
        xyz_t = dfslice.as_matrix(columns=['x', 'y', 'z']).T
        x,y,z = xyz_t[0], xyz_t[1], xyz_t[2]

        # re-project
        d = np.sqrt(x**2 + y**2 + z**2)
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)/r # -1/180 * np.pi * r
        hx = r*np.cos(-phi) - r*np.sin(-phi)
        hx = hx * x_factor
        hy = r*np.sin(-phi) + r*np.cos(-phi)
        hy = hy * y_factor
        hz = z
        hd = np.sqrt(hx**2 + hy**2 + hz**2)
        hxse = pd.Series(hx.tolist())
        hyse = pd.Series(hy.tolist())
        hzse = pd.Series(hz.tolist())
        dfslice['hx'] = hxse.values
        dfslice['hy'] = hyse.values
        dfslice['hz'] = hzse.values
        append_df = append_df.append(dfslice, ignore_index = True)

        if plot_it:
            ax1.plot(x,y,z,'-.', markersize=8)
            ax2.plot(hx,hy,hz,'-.', markersize=8)

    if plot_it:
        plt.show()

    return append_df

append_df = df[(df.momentum<=1) | (df.momentum>=1.2)].copy(deep=True)
append_df['hx'] = append_df['x'].map(lambda x: x)
append_df['hy'] = append_df['y'].map(lambda y: y)
append_df['hz'] = append_df['z'].map(lambda z: z)

dfpxpy = df[(df.x>0) & (df.y>0) & (df.momentum>1) & (df.momentum<1.2)]
dfpxny = df[(df.x>0) & (df.y<0) & (df.momentum>1) & (df.momentum<1.2)]
dfnxpy = df[(df.x<0) & (df.y>0) & (df.momentum>1) & (df.momentum<1.2)]
dfnxny = df[(df.x<0) & (df.y<0) & (df.momentum>1) & (df.momentum<1.2)]

append_df = unroll_helixes(dfpxpy, append_df, 1, 1, plot_it=False)
append_df = unroll_helixes(dfpxny, append_df, 1, -1, plot_it=False)
append_df = unroll_helixes(dfnxpy, append_df, -1, 1, plot_it=False)
append_df = unroll_helixes(dfnxny, append_df, -1, -1, plot_it=False)

# Now, find way to cluster on new values
global_eps = 0.007

testit = []
testit.append(append_df)
testit.append(hits)
for t in testit:
    model = Clusterer(eps=global_eps)
    labels = model.predict(t)
    submission = create_one_event_submission(0, t, labels)
    score = score_event(truth, submission)
    print("Your score: ", score)
