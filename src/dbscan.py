import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import seeds as sd


class Clusterer(object):
    
    def __init__(self, eps):
        self.eps = eps
        
    
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
        
        return X
    
    
    def predict(self, hits):
        
        X = self._preprocess(hits)
        
        cl = DBSCAN(eps=self.eps, min_samples=3, algorithm='kd_tree')
        labels = cl.fit_predict(X)
        
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

global_eps = 0.007

model = Clusterer(eps=global_eps)
labels = model.predict(hits)

print(labels)


submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)

print("Your score: ", score)

dataset_submissions = []
dataset_scores = []


for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=0, nevents=1):
        
    # Track pattern recognition
    model = Clusterer(eps=global_eps)
    labels = model.predict(hits)+1
        
    # Prepare submission for an event
    one_submission = create_one_event_submission(event_id, hits, labels)
    dataset_submissions.append(one_submission)
    
    # Score for the event
    score = score_event(truth, one_submission)
    dataset_scores.append(score)
    
    print("Score for event %d: %.3f" % (event_id, score))

    sd.count_truth_track_seed_hits(labels, truth, print_results=True)
    # volume_id -> 7,8,9

    truth = truth.merge(hits, on=['hit_id'], how='left')
    #truth = truth.merge(particles, on=['particle_id'], how='left')
    #print(truth.head)
    hits2 = hits.copy(deep=True)
    hits2['pred_track'] = 0
    my_volumes = [7, 8, 9]
    hits3 = hits2.loc[hits2['volume_id'].isin(my_volumes)]
    #indexes = hits2.index[hits2['volume_id'].isin(my_volumes)].tolist()
    model2 = Clusterer(eps=global_eps)
    hits4 = hits3.copy(deep=True)
    labels2 = model.predict(hits4)+1

    unique2 = np.unique(labels2)
    print('unique2: ' + str(unique2))
    truth2 = truth.loc[truth['volume_id'].isin(my_volumes)]
    truth2 = truth2.copy(deep=True)
    sd.count_truth_track_seed_hits(labels2, truth2, print_results=True)
    
print('Mean score: %.3f' % (np.mean(dataset_scores)))


path_to_test = "../input/test"
test_dataset_submissions = []

create_submission = False # True for submission 

if create_submission:
    for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

        # Track pattern recognition
        model = Clusterer(eps=global_eps)
        labels = model.predict(hits)

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels)
        test_dataset_submissions.append(one_submission)
        
        print('Event ID: ', event_id)

    # Create submission file
    submussion = pd.concat(test_dataset_submissions, axis=0)
    submussion.to_csv('submission.csv.gz', index=False, compression='gzip')

