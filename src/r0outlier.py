import numpy as np
import pandas as pd
import math
import collections as coll
from scipy.optimize import least_squares


def estimate_helix_r0(track_ix, hits):
    def residuals_xy(param, x, y):
        x0, y0 = param
        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        d = r - r.mean()
        return d

    df = hits.iloc[track_ix]
    t = df[['x', 'y', 'z']].values
    t = t[np.argsort(np.fabs(t[:,2]))]

    x = t[:,0]
    y = t[:,1]
    #z = xyz[:,2]
    
    param0 = (x.mean(), y.mean())
    res_lsq = least_squares(residuals_xy, param0, loss='soft_l1', f_scale=1.0, args=(x,y))
    x0,y0 = res_lsq.x
    r0 = np.sqrt((x-x0)**2 + (y-y0)**2).mean()

    return r0

def is_horrible_track(track_ix, labels, hits):
    df = hits.iloc[track_ix]
    df = df.sort_values('z')
    vols = df.volume_id.values
    lays = df.layer_id.values
    zs = df.z.values
    dupz_count = 0
    seen_vols = [0]
    seen_lays = [0]
    horrible = False
    last_lay_count = 0
    for ix, vol in enumerate(vols):
        if vol != seen_vols[-1]:
            seen_lays = [lays[ix]]
            # Check if vol in seen_vols (i.e. go back and forth between volumes)
            # 8/14/0 (so-so, not too great though, too many false positives!)
            seen_vols.append(vol)
            last_lay_count = 1
        elif lays[ix] != seen_lays[-1]:
            seen_lays.append(lays[ix])
            last_lay_count = 1
        else:
            last_lay_count = last_lay_count + 1
            # count==7: 3/0/0, count==6: 15/2/0, count==5: 28/8/0, count==4: 83/87/2
            if last_lay_count == 5:
                horrible = True
                break
        if ix > 0 and zs[ix] == zs[ix-1] and vol == vols[ix-1] and lays[ix] == lays[ix-1]:
            dupz_count = dupz_count + 1
            # count==1: 30/17/0, count==2: 21/4/0, count==3: 12/2/0, count==4: 11/0/0
            if dupz_count == 2:
                horrible = True
                break

    return horrible
    
# def find_horrible_tracks(labels, hits):
#     tracks = np.unique(helix6)
#     horrible_tracks = []
#     for track in tracks:
#         if track == 0: continue
#         if is_horrible_track(track, labels, hits):
#             horrible_tracks.append(track)
#     return horrible_tracks

# def find_badr0_tracks(labels, hits):
#     bad_r0s = []
#     tracks = np.unique(labels)
#     for track in tracks:
#         if track == 0: continue
#         tix = np.where(labels==track)[0]
#         if len(tix) < 4:
#             continue
#         df = hits.iloc[tix]
#         t = hits.iloc[tix].as_matrix(columns=['x','y','z'])
#         t = t[np.argsort(np.fabs(t[:,2]))]
#         x0, y0, r0  = helix_estimate_param_from_track(t)
#         #print('ii: ' + str(ii) + ', r0: '+ str(r0))
#         if int(r0) >= 325:
#             bad_r0s.append(track)
#     return bad_r0s


def remove_badr0_tracks(labels, hits):
    tracks = np.unique(labels)
    for track in tracks:
        if track == 0: continue
        tix = np.where(labels==track)[0]
        if len(tix) < 4:
            labels[tix] = 0
            continue
        r0 = estimate_helix_r0(tix, hits)
        #print('ii: ' + str(ii) + ', r0: '+ str(r0))
        # >325 seems to find ratio of about 2/3 horrible tracks to 1/3 imperfect tracks
        if int(r0) >= 350:
            labels[tix] = 0
        elif is_horrible_track(tix, labels, hits):
            labels[tix] = 0

    return labels
