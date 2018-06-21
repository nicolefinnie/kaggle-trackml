import numpy as np
from sklearn.neighbors import KDTree

def extend_labels(iter, labels, hits, do_swap=False, limit=0.04):
    df = hits.copy(deep=True)
    df['track_id'] = labels.tolist()
    return extend(iter, df, do_swap, limit).track_id.values

def extend_submission(iter, submissions, hits, do_swap=False, limit=0.04):
    df = submissions.merge(hits,  on=['hit_id'], how='left')
    df = extend(iter, df, do_swap, limit)
    return df[['event_id', 'hit_id', 'track_id']]       
    
def _one_cone_slice(df, angle, delta_angle, limit=0.04, num_neighbours=18):

    df1 = df.loc[(df.arctan2>(angle - delta_angle)/180*np.pi) & (df.arctan2<(angle + delta_angle)/180*np.pi)]

    min_num_neighbours = len(df1)
    if min_num_neighbours < 3: 
        return df

    hit_ids = df1.hit_id.values
    x,y,z = df1.as_matrix(columns=['x', 'y', 'z']).T
    r  = (x**2 + y**2)**0.5
    r  = r/1000
    a  = np.arctan2(y,x)
    c = np.cos(a)
    s = np.sin(a)
    tree = KDTree(np.column_stack([c,s,r]), metric='euclidean')

    track_ids = list(df1.track_id.unique())
    num_track_ids = len(track_ids)
    min_length=3

    for i in range(num_track_ids):
        p = track_ids[i]
        if p==0: continue

        idx = np.where(df1.track_id==p)[0]
        cur_track_len = len(idx)
        if cur_track_len<min_length: continue

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
        c0 = c[idx0]
        c1 = c[idx1]
        s0 = s[idx0]
        s1 = s[idx1]

        da0 = a[idx[1]] - a[idx[0]]  #direction
        dr0 = r[idx[1]] - r[idx[0]]
        direction0 = np.arctan2(dr0,da0)

        da1 = a[idx[-1]] - a[idx[-2]]
        dr1 = r[idx[-1]] - r[idx[-2]]
        direction1 = np.arctan2(dr1,da1)

        ## extend start point
        ns = tree.query([[c0, s0, r0]], k=min(num_neighbours, min_num_neighbours), return_distance=False)
        ns = np.concatenate(ns)

        direction = np.arctan2(r0 - r[ns], a0 - a[ns])
        diff = 1 - np.cos(direction - direction0)
        ns = ns[(r0 - r[ns] > 0.01) & (diff < (1 - np.cos(limit)))]
        for n in ns:
            df_ix = hit_ids[n] - 1
            old_track = df.loc[df_ix, 'track_id']
            if old_track == 0:
                df.loc[df_ix, 'track_id'] = p
            elif old_track != 0:
                # If the hit is already occupied by another track, only take ownership
                # of the hit if our track is longer than the current-occupying track.
                existing_track_len = len(np.where(df.track_id==old_track)[0])
                if cur_track_len > existing_track_len:
                    df.loc[df_ix, 'track_id'] = p
    

        ## extend end point
        ns = tree.query([[c1, s1, r1]], k=min(num_neighbours, min_num_neighbours), return_distance=False)
        ns = np.concatenate(ns)

        direction = np.arctan2(r[ns] - r1, a[ns] - a1)
        diff = 1 - np.cos(direction - direction1)
  
        ns = ns[(r[ns] - r1 > 0.01) & (diff < (1 - np.cos(limit)))]
        for n in ns:  
            df_ix = hit_ids[n] - 1
            old_track = df.loc[df_ix, 'track_id']
            if old_track == 0:
                df.loc[df_ix, 'track_id'] = p
            elif old_track != 0:
                # If the hit is already occupied by another track, only take ownership
                # of the hit if our track is longer than the current-occupying track.
                existing_track_len = len(np.where(df.track_id==old_track)[0])
                if cur_track_len > existing_track_len:
                    df.loc[df_ix, 'track_id'] = p
      
    return df

def extend(iter, df, do_swap=False, limit=0.04):
    if do_swap:
        df = df.assign(x = -df.x)
        df = df.assign(y = -df.y)

    df = df.assign(d = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(r = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(arctan2 = np.arctan2(df.z, df.r))


    for angle in range(-90,90,1):

        print ('\r%d %f '%(iter,angle), end='',flush=True)
        df1 = df.loc[(df.arctan2>(angle-1.0)/180*np.pi) & (df.arctan2<(angle+1.0)/180*np.pi)]

        num_hits = len(df1)
        # Dynamically adjust the delta based on how many hits are found
        if num_hits > 2000:
            df = _one_cone_slice(df, angle-0.6, 0.4, limit)
            df = _one_cone_slice(df, angle-0.2, 0.4, limit)
            df = _one_cone_slice(df, angle+0.2, 0.4, limit)
            df = _one_cone_slice(df, angle+0.6, 0.4, limit)
        else:
            df = _one_cone_slice(df, angle, 1, limit)
           
    return df

       