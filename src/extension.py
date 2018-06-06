import numpy as np
from sklearn.neighbors import KDTree
def extend(submission,hits):

		df = submission.merge(hits,  on=['hit_id'], how='left')
		df = df.assign(d = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
		df = df.assign(r = np.sqrt( df.x**2 + df.y**2))
		df = df.assign(arctan2 = np.arctan2(df.z, df.r))

		for angle in range(-180,180,1):

		    print ('\r %f '%angle, end='',flush=True)
		    #df1 = df.loc[(df.arctan2>(angle-0.5)/180*np.pi) & (df.arctan2<(angle+0.5)/180*np.pi)]
		    df1 = df.loc[(df.arctan2>(angle-1.0)/180*np.pi) & (df.arctan2<(angle+1.0)/180*np.pi)]

		    min_num_neighbours = len(df1)
		    if min_num_neighbours<4: continue

		    hit_ids = df1.hit_id.values
		    x,y,z = df1.as_matrix(columns=['x', 'y', 'z']).T
		    r  = (x**2 + y**2)**0.5
		    r  = r/1000
		    a  = np.arctan2(y,x)
		    tree = KDTree(np.column_stack([a,r]), metric='euclidean')

		    track_ids = list(df1.track_id.unique())
		    num_track_ids = len(track_ids)
		    min_length=3

		    for i in range(num_track_ids):
		        p = track_ids[i]
		        if p==0: continue

		        idx = np.where(df1.track_id==p)[0]
		        if len(idx)<min_length: continue

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
		            df.loc[ df.hit_id==hit_ids[n],'track_id' ] = p 

		        ## extend end point
		        ns = tree.query([[a1,r1]], k=min(20,min_num_neighbours), return_distance=False)
		        ns = np.concatenate(ns)

		        direction = np.arctan2(r[ns]-r1,a[ns]-a1)
		        ns = ns[(r[ns]-r1>0.01) &(np.fabs(direction-direction1)<0.04)] 
		        
		        for n in ns:
		            df.loc[ df.hit_id==hit_ids[n],'track_id' ] = p
		#print ('\r')
		df = df[['event_id', 'hit_id', 'track_id']]
		return df