import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
import multiprocessing as mp
import threading as thr

from sklearn.preprocessing import StandardScaler
#import hdbscan
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import argparse
import seeds as sd
import collections as coll
import math
from extension import extend_submission, extend_labels
import cone_slicing as cone
import merge as merge


SCALED_DISTANCE = [1,       1,       0.50, 0.125, 0.008, 0.008, 0.00175, 0.00175]
FEATURE_MATRIX = ['sina1', 'cosa1', 'z1', 'z2',  'xd', 'yd', 'px', 'py']

SCALED_DISTANCE_2 = [1,       1,       0.5, 0.008, 0.008, 0.00185, 0.00185]
FEATURE_MATRIX_2 = ['sina1', 'cosa1', 'z3', 'xd', 'yd', 'px', 'py']



STEPRR = 0.03

STEPEPS = 0.0000015
STEPS = 120
EXTENSION_STANDARD_LIMITS = [0.02, 0.04, 0.06, 0.08, 0.10]
EXTENSION_LIGHT_LIMITS = [0.03, 0.07]

DBSCAN_EPS = 0.0033

print ('########################################################################')

print('steps: ' + str(STEPS))
print('steprr: ' + str(STEPRR))

print('stepeps: ' + str(STEPEPS))
print('extension standard limits: ' + str(EXTENSION_STANDARD_LIMITS))
print('extension light limits: ' + str(EXTENSION_LIGHT_LIMITS))

print ('#######################################################################')

class Clusterer(object):
    def __init__(self, model_parameters):                        
        self.model_parameters = model_parameters

    def _dbscan(self, dfh, label_file_root):
        labels = []

        dfh['d'] = np.sqrt(dfh.x**2+dfh.y**2+dfh.z**2)
        dfh['r'] = np.sqrt(dfh.x**2+dfh.y**2)
        dfh['z5'] = (dfh.r - np.absolute(dfh.z))/dfh.r
        #theta
        dfh['rz'] = np.arctan2(dfh.r, dfh.z)
        rr = dfh['r']/1000      

        for loop in range(len(self.model_parameters[2])):
            label_file = label_file_root + '_dbscan' + str(loop+1) + '.csv'
            if os.path.exists(label_file):
                print('Loading dbscan loop ' + str(loop+1) + ' file: ' + label_file)
                labels.append(pd.read_csv(label_file).label.values)
            else:
                if loop%4 == 0:
                    dfh['a0'] = np.arctan2(dfh.y,dfh.x)
                    dfh['xd'] = dfh.x/dfh['d']
                    dfh['yd'] = dfh.y/dfh['d']
                elif loop%4 == 1:
                    dfh['a0'] = np.arctan2(dfh.x,-dfh.y)
                    dfh['xd'] = -dfh.y/dfh['d']
                    dfh['yd'] = dfh.x/dfh['d']
                elif loop%4 == 2:
                    dfh['a0'] = np.arctan2(-dfh.y,-dfh.x)
                    dfh['xd'] = -dfh.x/dfh['d']
                    dfh['yd'] = -dfh.y/dfh['d']
                else:
                    dfh['a0'] = np.arctan2(-dfh.x,dfh.y)
                    dfh['xd'] = dfh.y/dfh['d']
                    dfh['yd'] = -dfh.x/dfh['d']

                # Main DBSCAN loop. Longest-track-wins merging at each step.
                for ii in tqdm(np.arange(-STEPS, STEPS, 1)):
                    print ('\r steps: %d '%ii, end='',flush=True)
                    dfh['zshift'] = dfh.z + self.model_parameters[2][loop]
                    dfh['z1'] = dfh.zshift/dfh['r'] 
                    dfh['z2'] = dfh.zshift/dfh['d']
                    dfh['z3'] = np.log1p(np.absolute(dfh.zshift/dfh.r))*np.sign(dfh.zshift)

                    dfh['a1'] = dfh['a0'] + (rr + STEPRR*rr**2)*ii/180*np.pi + (0.00001*ii)*dfh.z*np.sign(dfh.z)/180*np.pi

                    # parameter space
                    dfh['px'] = -dfh.r*np.cos(dfh.a1)*np.cos(dfh.a0) - dfh.r*np.sin(dfh.a1)*np.sin(dfh.a0)
                    dfh['py'] = -dfh.r*np.cos(dfh.a1)*np.sin(dfh.a0) + dfh.r*np.sin(dfh.a1)*np.cos(dfh.a0)
                    
                    dfh['sina1'] = np.sin(dfh['a1'])
                    dfh['cosa1'] = np.cos(dfh['a1'])
                    
                    ss = StandardScaler()
                
                    dfs = ss.fit_transform(dfh[self.model_parameters[0]].values)
                    dfs = np.multiply(dfs, self.model_parameters[1])
            
                    self.clusters = DBSCAN(eps=DBSCAN_EPS  + ii*STEPEPS,min_samples=1, n_jobs=-1).fit(dfs).labels_

                    if ii == -STEPS:
                        dfh['s1'] = self.clusters
                        dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
                    else:
                        dfh['s2'] = self.clusters
                        dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
                        maxs1 = dfh['s1'].max()
                        cond = np.where( (dfh['N2'].values>dfh['N1'].values) & (dfh['N2'].values < 20) )
                        s1 = dfh['s1'].values
                        s1[cond] = dfh['s2'].values[cond]+maxs1
                        dfh['s1'] = s1
                        dfh['s1'] = dfh['s1'].astype('int64')
                        dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')

                # Save the predicted tracks to a file so we don't need to re-calculate next time.
                labels.append(np.copy(dfh['s1'].values))
                df = pd.DataFrame(labels[loop])
                df.to_csv(label_file, index=False, header=['label'])
 
        return (labels)

    def predict(self, hits, label_file_root): 
        (labels)  = self._dbscan(hits, label_file_root)

        return (labels)

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

def run_predictions(event_id, all_labels, all_hits, truth, model, label_file_root, unmatched_only=True, filter_hits=True, track_extension_limits=None, merge_overwrite_limit=4):
    """ Run a round of predictions on all or a subset of remaining hits.
    Parameters:
      all_labels: Input np array of labeled tracks, where the index in all_labels matches
        the index of the corresponding hit in the all_hits dataframe. Each value contains
        either 0 (if the corresponding hit is not associated with any track), or a unique
        track ID (all hits that form the same track will have the same track ID).
      all_hits: Dataframe containing all the hits to be predicted for an event.
      model: The model that predictions will be run on. This model must expose a
        'predict()' method that accepts the input hits to predict as the first parameter,
        and the input 'model_parameters' as the second input parameter.
      unmatched_only: True iff only unmatched hits should be predicted. Unmatched hits are
        determined from the all_labels input array, where an unmatched hit contains a
        track ID of 0. False for this parameter means that all hits in the all_hits
        dataframe will be used to make predictions.
      filter_hits: True iff the predicted hits should be filter to those known to be
        high quality, i.e. that have a specific minimum track length, and that
        contain hits in volumes 7, 8, or 9. False for this parameter means that no
        filtering will be performed.
      track_extension_limits: The angular limits to use when applying track extensions.
        Track extension is performed from the full list of tracks.

    Returns: The new np array of predicted labels/tracks, as well as the unfiltered version.
    """
    hits_to_predict = all_hits

    if unmatched_only:
        # Make a copy of the hits, removing all hits from valid_labels
        hits_to_predict = all_hits.copy(deep=True)
        drop_indices = np.where(all_labels != 0)[0]
        hits_to_predict = hits_to_predict.drop(hits_to_predict.index[drop_indices])

    # Run predictions on the input model
    (labels_subset) = model.predict(hits_to_predict, label_file_root)
    labels_full = []

    # Make sure max track ID is not larger than length of labels list.
    for i in range(len(labels_subset)):
        label_file = label_file_root + '_dbscan' + str(i+1) + '_processed.csv'
        if os.path.exists(label_file):
            print('Loading dbscan loop ' + str(i+1) + ' processed file: ' + label_file)
            labels_full.append(pd.read_csv(label_file).label.values)
            if truth is not None:
                one_submission = create_one_event_submission(event_id, all_hits, labels_full[i])
                score = score_event(truth, one_submission)
                print("Processed dbscan loop %d score for event %d: %.8f" % (i+1,event_id, score))
        else:
            labels_subset[i] = merge.renumber_labels(labels_subset[i])

            # If only predicting on unmatched hits, add any new predicted tracks directly
            # into the output labels. Otherwise, just return the newly predicted output labels.
            if unmatched_only:
                labels_full.append(np.copy(all_labels))
                labels_subset[i][labels_subset[i] == 0] = 0 - len(all_labels) - 1
                labels_subset[i] = labels_subset[i] + len(all_labels) + 1
                labels_full[i][labels_full[i] == 0] = labels_subset[i]
            else:
                labels_full.append(np.copy(labels_subset[i]))

            if truth is not None:
                one_submission = create_one_event_submission(event_id, all_hits, labels_full[i])
                score = score_event(truth, one_submission)
                print("Unfiltered dbscan loop %d score for event %d: %.8f" % (i+1,event_id, score))

            # If desired, extend tracks
            if track_extension_limits is not None:
                for ix, limit in enumerate(track_extension_limits):
                    labels_full[i] = extend_labels(ix, labels_full[i], all_hits, do_swap=(ix%2==1), limit=(limit))
                    if False and truth is not None:
                        one_submission = create_one_event_submission(event_id, all_hits, labels_full[i])
                        score = score_event(truth, one_submission)
                        print("Extension %d loop %d score for event %d: %.8f" % (ix,i+1,event_id, score))
                    
                labels_full[i] = merge.renumber_labels(labels_full[i])
                
                if truth is not None:
                    one_submission = create_one_event_submission(event_id, all_hits, labels_full[i])
                    score = score_event(truth, one_submission)
                    print("Unfiltered extended dbscan loop %d score for event %d: %.8f" % (i+1,event_id, score))
            else:
                labels_full[i] = merge.renumber_labels(labels_full[i])

            if filter_hits:
                # Filter out tracks that are too small, as well as hits that look
                # like outliers, i.e. duplicate-z values, slopes that do not match
                # other hits in the track, etc.
                labels_full[i] = merge.remove_outliers(labels_full[i], all_hits, smallest_track_size=5, print_counts=False)
                labels_full[i] = merge.renumber_labels(labels_full[i])

                if truth is not None:
                    one_submission = create_one_event_submission(event_id, all_hits, labels_full[i])
                    score = score_event(truth, one_submission)
                    print("Filtered non-outlier dbscan loop %d score for event %d: %.8f" % (i+1,event_id, score))

            # Save the predicted tracks to a file so we don't need to re-calculate next time.
            df = pd.DataFrame(labels_full[i])
            df.to_csv(label_file, index=False, header=['label'])

    # Merge all dbscan loop labels together
    for i in range(len(labels_full)):
        if i == 0:
            labels_merged = labels_full[0]
        else:
            labels_merged = merge.heuristic_merge_tracks(labels_merged, labels_full[i], overwrite_limit=merge_overwrite_limit, print_summary=False)
            if truth is not None:
                one_submission = create_one_event_submission(event_id, all_hits, labels_merged)
                score = score_event(truth, one_submission)
                print("Merged loop 1-%d score for event %d: %.8f" % (i+1,event_id, score))

    return (labels_merged)

def run_helix_unrolling_predictions(event_id, hits, truth, label_identifier, model_parameters):
    # Shortcut - if we've previously generated and saved labels, just use them
    # rather than re-generating.
    label_file_root = 'event_' + str(event_id)+'_labels_' + label_identifier
    label_file = label_file_root + '.csv'
    if os.path.exists(label_file):
        print(str(event_id) + ': load ' + label_file)
        labels = pd.read_csv(label_file).label.values
        if truth is not None:
            one_submission = create_one_event_submission(event_id, hits, labels)
            score = score_event(truth, one_submission)
            print("Loaded score for event %d: %.8f" % (event_id, score))
        return labels

    print(str(event_id) + ': clustering on ' + label_identifier)

    model = Clusterer(model_parameters)
    
    # For the first run, we do not have an input array of labels/tracks.
    label_file_root1 = label_file_root + '_phase1'
    (labels) = run_predictions(event_id, None, hits, truth, model, label_file_root1, unmatched_only=False, filter_hits=True, track_extension_limits=EXTENSION_STANDARD_LIMITS)

    if truth is not None:
        # Score for the event
        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)
        print("Filtered 1st pass score for event %d: %.8f" % (event_id, score))

    label_file_root2 = label_file_root + '_phase2'
    model = Clusterer(model_parameters)
    (labels) = run_predictions(event_id, labels, hits, truth, model, label_file_root2, unmatched_only=True, filter_hits=False, track_extension_limits=EXTENSION_LIGHT_LIMITS, merge_overwrite_limit=2)

    if truth is not None:
        # Score for the event
        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)
        print("2nd pass score for event %d: %.8f" % (event_id, score))

        # Un-comment this if you want to see the quality of the seeds generated.
        #seed_length = 5
        #my_volumes = [7, 8, 9]
        #labels = sd.filter_invalid_tracks(labels, hits, my_volumes, seed_length)
        #sd.count_truth_track_seed_hits(labels, truth, seed_length, print_results=True)

    df = pd.DataFrame(labels)
    df.to_csv(label_file, index=False, header=['label'])

    # label_file_root3 = label_file_root + '_phase3'
    # model = Clusterer(model_parameters)
    # (labels) = run_predictions(event_id, labels, hits, truth, model, label_file_root3, unmatched_only=True, merge_labels=True, filter_hits=False, track_extension=True)

    # # Save the generated labels, can avoid re-generation next run.


    # # Score for the event
    # if truth is not None:
    #     one_submission = create_one_event_submission(event_id, hits, labels)
    #     score = score_event(truth, one_submission)
    #     print("3rd pass score for event %d: %.8f" % (event_id, score))

    return labels

def print_info(helix_id, model_parameters):
    feature_matrix = model_parameters[0]
    scaled_distance = model_parameters[1]
    z_shift_matrix = model_parameters[2]

    print('==========================================================================')
    print('Helix model: ' + str(helix_id))
    print('Feature matrix: ' + str(feature_matrix))
    print('Scaled distance: ' + str(scaled_distance))
    print('z shift matrix: ' + str(z_shift_matrix))
    print('==========================================================================')
   

def predict_event(event_id, hits, train_or_test, truth):
    
    #DBSCAN_EPS_MATRIX = [0.0033, 0.0041, 0.0037, 0.0045]

    #The strongest model is  [3, 4, 10, -3]
    model_parameters = []
    model_parameters.append(FEATURE_MATRIX)
    model_parameters.append(SCALED_DISTANCE)
    model_parameters.append([3, -6, 4, 12, -9, 10, -3, 6, -10, 2, 8, -2])
    print_info(1, model_parameters)      
    labels_helix1 = run_helix_unrolling_predictions(event_id, hits, truth, train_or_test + '_helix1', model_parameters)
    

    model_parameters.clear()
    model_parameters.append(FEATURE_MATRIX_2)
    model_parameters.append(SCALED_DISTANCE_2)
    model_parameters.append([3, -6, 4, 12, -9, 10, -3, 6, -10, 2, 8, -2])
    print_info(2, model_parameters)      
    labels_helix2 = run_helix_unrolling_predictions(event_id, hits, truth, train_or_test + '_helix2', model_parameters)

    

    # Merge results from two sets of predictions, removing outliers first
    labels_helix1 = merge.remove_outliers(labels_helix1, hits, print_counts=False)
    labels_helix2 = merge.remove_outliers(labels_helix2, hits, print_counts=False)

    if truth is not None:
        one_submission = create_one_event_submission(event_id, hits, labels_helix1)
        score = score_event(truth, one_submission)
        print("After outlier removal helix1 %d: %.8f" % (event_id, score))
        
        one_submission = create_one_event_submission(event_id, hits, labels_helix2)
        score = score_event(truth, one_submission)
        print("After outlier removal helix2 %d: %.8f" % (event_id, score))

        
    labels = merge.heuristic_merge_tracks(labels_helix1, labels_helix2, overwrite_limit=3, print_summary=False)
    if truth is not None:
        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)
        print("Merged helix1&2 unrolling for event %d: %.8f" % (event_id, score))


    return labels


def run_single_threaded_training(skip, nevents):
    path_to_train = "../input/train_1"
    dataset_submissions = []
    dataset_scores = []

    for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=skip, nevents=nevents):

        labels = predict_event(event_id, hits, 'train', truth)

        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)

        # labels = merge.heuristic_merge_tracks(labels, labels_cone, print_summary=False)
        # one_submission = create_one_event_submission(event_id, hits, labels)
        # score = score_event(truth, one_submission)
        # print("Merged All unrolling and cone slicing for event %d: %.8f" % (event_id, score))

        # Un-comment this if you want to see the quality of the seeds generated.
        #seed_length = 5
        #my_volumes = [7, 8, 9]
        #labels = sd.filter_invalid_tracks(labels, hits, my_volumes, seed_length)
        #sd.count_truth_track_seed_hits(labels, truth, seed_length, print_results=True)

        # Append the final submission for this event, as well as the score.
        dataset_submissions.append(one_submission)
        dataset_scores.append(score)


    print('Mean score: %.8f' % (np.mean(dataset_scores)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', nargs=2, type=int)
    parser.add_argument('--training', nargs=2, type=int)
    args = parser.parse_args()
    test_skip = 0
    test_events = 0
    training_skip = 0
    training_events = 0

    if args.test is not None:
        test_skip = args.test[0]
        test_events = args.test[1]

    if args.training is not None:
        training_skip = args.training[0]
        training_events = args.training[1]

    if not training_events == 0:
        run_single_threaded_training(training_skip, training_events)

    path_to_test = "../input/test"
    test_dataset_submissions = []

    #create_submission = True # True for submission 
    if test_events > 0:
        print('Predicting test results, skip: %d, events: %d' % (test_skip, test_events))
        use_header = (test_skip == 0)
        for event_id, hits, cells in load_dataset(path_to_test, skip=test_skip, nevents=test_events, parts=['hits', 'cells']):

            print('Event ID: ', event_id)

            # Helix unrolling predictions
            labels = predict_event(event_id, hits, 'test', None)

            # Create our submission for this test event.
            one_submission = create_one_event_submission(event_id, hits, labels)
            test_dataset_submissions.append(one_submission)
            

        # Create submission file
        submission = pd.concat(test_dataset_submissions, axis=0)
        submission_file = 'submission_' + "{:03}".format(test_skip) + '_' + str(test_events) + '.csv'
        submission.to_csv(submission_file, index=False, header=use_header)

