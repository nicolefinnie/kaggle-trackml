import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.neighbors import NearestNeighbors

RZ_SCALES = [0.65, 0.965, 1.418] #1.41
LEAF_SIZE = 50

class Clusterer(object):
    
    def __init__(self, rz_scales):
        self.rz_scales = rz_scales
        
    
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
    
    
    def predict(self, hits):
        
        X = self._preprocess(hits)
        
        #cl = hdbscan.HDBSCAN(min_samples=3,min_cluster_size=7,cluster_selection_method='leaf',algorithm='boruvka_balltree')
        #cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,cluster_selection_method='leaf',metric='braycurtis',approx_min_span_tree=False,algorithm='boruvka_balltree')
        cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,cluster_selection_method='leaf',metric='braycurtis',leaf_size=LEAF_SIZE,approx_min_span_tree=False)
        labels = cl.fit_predict(X) + 1
        
        return labels


def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission



# Change this according to your directory preferred setting
path_to_train = "../input/train_100_events"

# This event is in Train_1
event_prefix = "event000001000"

hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))

hits.head()

model = Clusterer(rz_scales=RZ_SCALES)
labels = model.predict(hits)

print(labels)


submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)

print("Your score: ", score)

dataset_submissions = []
dataset_scores = []


for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=10, nevents=10):
        
    # Track pattern recognition
    model = Clusterer(rz_scales=RZ_SCALES)
    labels = model.predict(hits)
        
    # Prepare submission for an event
    one_submission = create_one_event_submission(event_id, hits, labels)
    dataset_submissions.append(one_submission)
    
    # Score for the event
    score = score_event(truth, one_submission)
    dataset_scores.append(score)
    
    print("Score for event %d: %.3f" % (event_id, score))
    
print('Mean score: %.3f' % (np.mean(dataset_scores)))


path_to_test = "../input/test"
test_dataset_submissions = []

create_submission = False # True for submission 

if create_submission:
    for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

        # Track pattern recognition
        model = Clusterer(rz_scales=RZ_SCALES)
        labels = model.predict(hits)

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels)
        test_dataset_submissions.append(one_submission)
        
        print('Event ID: ', event_id)

    # Create submission file
    submussion = pd.concat(test_dataset_submissions, axis=0)
    submussion.to_csv('submission.csv.gz', index=False, compression='gzip')

