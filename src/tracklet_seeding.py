#https://storage.googleapis.com/kaggle-forum-message-attachments/346809/9660/tracklet_seeding.py
import os
from trackml.score  import score_event
from trackml.dataset import load_event, load_dataset

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import merge as merge


#https://www.kaggle.com/cpmpml/a-faster-python-scoring-function
def score_event_fast_3(truth, submission):
    truth = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    truth['count_both'] = truth.groupby(['track_id', 'particle_id']).hit_id.transform('count')    
    truth['count_particle'] = truth.groupby(['particle_id']).hit_id.transform('count')
    truth['count_track'] = truth.groupby(['track_id']).hit_id.transform('count')

    score = truth[(truth.count_both > 0.5*truth.count_particle) & (truth.count_both > 0.5*truth.count_track)].weight.sum()
    results = truth
    return score, results 

def create_df_submission(event_id, df, labels):
    submission = pd.DataFrame(columns=['event_id', 'hit_id', 'track_id'],
        data=np.column_stack(([int(event_id),]*len(df), df.hit_id.values, labels))
    ).astype(int)
        
    return submission   


def dbscan_tracklet_seeding(event_id, hits, truth):
    truth = truth.merge(hits, on=['hit_id'], how='left')
    #--------------------------------------------------------
    df = truth.copy()
    df = df.assign(r   = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(d   = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(a   = np.arctan2(df.y, df.x))
    df = df.assign(cosa= np.cos(df.a))
    df = df.assign(sina= np.sin(df.a))
    df = df.assign(phi = np.arctan2(df.z, df.r))

    # df = df.loc[np.absolute(df.z) < 500] # consider dataset subset
    # df = df.loc[(df.r<200)  ]

    layer_id = df['layer_id'].values.astype(np.float32)
    x,y,z,r,a,cosa,sina,phi = df[['x', 'y', 'z', 'r', 'a', 'cosa', 'sina', 'phi']].values.astype(np.float32).T

    N = len(df)
    # do dbscan here =======================================
    dj=0
    di=0
    EPS=1e-12
    STEPEPS = 0.0015
    #if 1:

    candidates = []
    for dj in tqdm(np.arange(-40, 40+EPS, 5)):
        for di in np.arange(-0.003, 0.003+EPS, 0.00025):
            ar = a +  di*r
            zr = (z+dj)/r  *0.1
            data2 = np.column_stack([ar,zr])

            l = DBSCAN(eps=0.0033+di*STEPEPS, min_samples=1).fit(data2).labels_
            track_ids = np.unique(l)
            track_ids = track_ids[track_ids!=0]
            #neighbour = [ np.where(l==t)[0] for t in track_ids]

            unique,inverse,c = np.unique(l,return_counts=True,return_inverse=True)
            unique = unique[unique!=0]
            c = c[inverse]
            c[l==0] = 0

            for u in unique:
                candidate = np.where(l==u)[0]
                candidates.append(candidate)

                
    #---
    #<todo>
    #fix angle discontinunity problem here ...

    count = np.array([len(candidate) for candidate in candidates])
    sort = np.argsort(-count)
    candidates = [candidates[s] for s in sort]


    max_label=1
    labels = np.zeros(N,np.int32)
    count = np.zeros(N,np.int32)

    for candidate in candidates:
        n = candidate
        L = len(n)
        #print(L)


        #---- filtering (secret sauce) ----------
        #if L<3: continue
        n = n[np.argsort(np.fabs(z[n]))]

        layer_id0 = layer_id[n[:-1]]
        layer_id1 = layer_id[n[1: ]]
        ld = layer_id1-layer_id0
        if np.any(ld>2): continue

        m = count[n].max()
        if L<m: continue

        #---- filtering ----------------------

        count[n]=L
        labels[n]=max_label
        max_label += 1

    #labels, _ = merge.remove_small_tracks(labels, smallest_track_size=3)

    
    submission = pd.DataFrame(columns=['event_id', 'hit_id', 'track_id'],
        data=np.column_stack(([int(event_id),]*len(df), df.hit_id.values, labels))
    ).astype(int)
    score1 = score_event(df, submission)
    score2, results = score_event_fast_3(df, submission)
 

    #print results
    max_score = df.weight.sum()
    print('max_score = df.weight.sum() = %0.5f'%max_score)
    print('score1= %0.5f  (%0.5f)'%(score1*max_score,score1))
    print('score2= %0.5f  (%0.5f)'%(score2,score2/max_score))

    label_file='all_volume_'+ str(event_id) + '.csv'
    df = pd.DataFrame(labels)
    df.to_csv(label_file, index=False, header=['label'])

    print('end')
    exit(0)

if __name__ == '__main__':
    path_to_train = "../input/train_1"
    #event 1003
    for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=3, nevents=1):
        dbscan_tracklet_seeding(event_id, hits, truth)


    
    