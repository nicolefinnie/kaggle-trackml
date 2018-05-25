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

print(len(labels))
print(labels.shape)
print(labels)
count0 = np.count_nonzero(labels == 0)
count1 = np.count_nonzero(labels == 1)
count2 = np.count_nonzero(labels == 2)
#print(count0)
#print(count1)
#print(count2)
max_track = np.amax(labels)
for i in range(0,max_track+1):
    hit_count = np.count_nonzero(labels == i)
    if hit_count < 5:
        labels[labels == i] = 0
        #print('track i: ' + str(i) + ' has a hit count of: ' + str(hit_count))

def renumber_labels(labels):
    new_label = 0
    for old_label in np.unique(labels):
        if not old_label == new_label:
            labels[labels == old_label] = new_label
        new_label += 1

    return labels

labels = renumber_labels(labels)

print('truth')
print(truth.shape)
#print(truth.head)
truth_tracks = truth['particle_id'].values
total_tracks = len(np.unique(truth_tracks))
print('total truth tracks: ' + str(total_tracks))
print('predicted tracks: ' + str(max_track))

def matches_first_hits(labels, track, truth):
    match_found = False
    all_indices = np.where(labels == track)[0]
    # FIXME: Need to sort on z-value, and then see if first 'n' (3) particle_ids match
    if truth.iloc[all_indices[0]]['particle_id'] == truth.iloc[all_indices[1]]['particle_id']:
        if truth.iloc[all_indices[1]]['particle_id'] == truth.iloc[all_indices[2]]['particle_id']:
            match_found = True
    return match_found

max_track = np.amax(labels)
count = 0
for i in range(1,max_track+1):
    found_it = matches_first_hits(labels, i, truth)
    if found_it and count < 5:
        print('matched track i: ' + str(i))
        count = count + 1

submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)

print("Your score: ", score)

dataset_submissions = []
dataset_scores = []


nevents = 0 # 10
for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=10, nevents=nevents):
        
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
    
if nevents > 0:
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

