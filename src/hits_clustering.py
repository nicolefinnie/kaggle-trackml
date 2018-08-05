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
import extension as xtnd
import cone_slicing as cone
import merge as merge
import free_hits as free
import straight_tracks as strt

INPUT_PATH = '../../input'

SCALED_DISTANCE = [1,       1,       0.50, 0.125, 0.008, 0.008, 0.00175, 0.00175]
FEATURE_MATRIX = ['sina1', 'cosa1', 'z1', 'z2',  'xd', 'yd', 'px', 'py']

SCALED_DISTANCE_2 = [1,       1,       0.5, 0.008, 0.008, 0.00185, 0.00185]
FEATURE_MATRIX_2 = ['sina1', 'cosa1', 'z3', 'xd', 'yd', 'px', 'py']

SCALED_DISTANCE_3 = [1,       1, 0.5, 0.25]
FEATURE_MATRIX_3 = ['sina1','cosa1', 'zr', 'z1']

SCALED_DISTANCE_4 = [1,       1, 0.5, 0.25]
FEATURE_MATRIX_4 = ['sina1','cosa1', 'zr', 'z3']

SCALED_DISTANCE_5 = [1, 1, 0.5, 0.25, 0.008, 0.008, 0.00175, 0.00175]
FEATURE_MATRIX_5 = ['sina1', 'cosa1', 'r0', 'z3', 'xd', 'yd', 'px', 'py']

SCALED_DISTANCE_6 = [1,       1,       0.5, 0.00185, 0.00185] 
FEATURE_MATRIX_6 = ['sina1', 'cosa1', 'zarc', 'px', 'py']


STEPRR = 0.03

STEPEPS = 0.0000015
STEPS = 100
EXTENSION_STANDARD_LIMITS = [0.02, 0.04, 0.06, 0.08, 0.10]
EXTENSION_LIGHT_LIMITS = [0.03, 0.07]


DBSCAN_EPS = 0.0033

SAMPLE = 50
R0_STEP_EPS = 0.00055
MIN_R0 = 270
MAX_R0 = 37500
R0_SAMPLE_EVENT = [1010, 1015, 1020, 1025, 1030, 1040, 1045, 1050, 1060, 1070, 1080, 1090]

# using radius of every hit to approximate the angle between y-axis and 
# the closest approach to the centre of helix
HELIX_UNROLL_R_MODE = 'Radius mode'
# using different r0 of every helix to guess the helix angle between y-axis and
# the closest approach to the centre of helix 
HELIX_UNROLL_R0_MODE = 'R0 mode'


print ('########################################################################')

print('steps: ' + str(STEPS))
print('steprr: ' + str(STEPRR))

print('stepeps: ' + str(STEPEPS))
print('r0 samples: ' + str(SAMPLE))
print('r0 step eps: ' + str(R0_STEP_EPS))
print('min r0: ' + str(MIN_R0))
print('r0 sample events: ' + str(R0_SAMPLE_EVENT))
print('extension standard limits: ' + str(EXTENSION_STANDARD_LIMITS))
print('extension light limits: ' + str(EXTENSION_LIGHT_LIMITS))


print ('#######################################################################')

class Clusterer(object):
    def __init__(self, model_parameters):                        
        self.model_parameters = model_parameters

    # To generate r0_list.csv, see src/notebooks/EDA/helix_parameter_space.ipynb
    # We use event_1030 as our sample data since it's more representative
    def _read_samples(self, event_id):
        path_to_r0 = os.path.join(INPUT_PATH, 'r0_list')
        return pd.read_csv(os.path.join(path_to_r0, 'r0_list_%s.csv'%event_id)).r0.values

    def _get_samples(self, num_samples, invert=True, duplicate=True, add_0=0, region1_percent=95, region1_type=3, region1_min=MIN_R0, region1_max=4500, region2_type=3, region2_min=4501, region2_max=MAX_R0):
        """
        num_samples - the number of sample points to generate
        invert - whether the values should be inverted before being returned
        duplicate - whether the result should be tiled (duplicated)
        add_0 - 0 does not add any 0 value, 1 adds the zero at the beginning of the sample, 2 adds the zero at the end
        region1_percent - amount of samples distributed to the first region
        region1_type - 1 for linear, 2 for narrow half-Gaussian, 3 for wide half-Gaussian
        region1_min - minimum value for samples of first region
        region1_max - approximate maximum value for samples of first region
        region2_type, region2_min, region2_max - see region1_type, region1_min, region1_max
        """
        def get_samples_single_region(num_samples, region_type, region_min, region_max):
            if region_type == 1:
                results = np.linspace(region_min, region_max, num=num_samples, endpoint=True, dtype=np.float32)
            else:
                mu = 0 # centre at 0 for now, adjust later
                if region_type == 2:
                    sigma = 0.25
                else:
                    sigma = 0.37
                results = np.abs(np.random.normal(mu, sigma, num_samples))
                results = (results * (region_max - region_min)) + region_min
                results[results > region_max] = results[results > region_max] - (region_max - region_min)
            return results

        np.random.seed(42)
        results = np.zeros(num_samples)
        num_region1 = math.floor(num_samples*(region1_percent/100))
        results[0:num_region1] = get_samples_single_region(num_region1, region1_type, region1_min, region1_max)
        results[num_region1:] = get_samples_single_region(num_samples-num_region1, region2_type, region2_min, region2_max)
        results.sort()
        if invert:
            results = 1 / results
        if add_0 == 1:
            results[0] = 0
        elif add_0 == 2:
            results[-1] = 0
        if duplicate:
            results = np.tile(results, 2)
        return results

    def _dbscan(self, dfh, label_file_root):
        labels = []

        dfh['d'] = np.sqrt(dfh.x**2+dfh.y**2+dfh.z**2)
        dfh['r'] = np.sqrt(dfh.x**2+dfh.y**2)
        dfh['zr'] = np.arctan2(dfh.z, dfh.r)
        
        rr = dfh['r']/1000      

        for loop in range(len(self.model_parameters[3])):

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
                dfh['zshift'] = dfh.z + self.model_parameters[3][loop]
                dfh['z1'] = dfh.zshift/dfh['r'] 
                dfh['z2'] = dfh.zshift/dfh['d']
                dfh['z3'] = np.log1p(np.absolute(dfh.zshift/ dfh.r))*np.sign(dfh.zshift)
                          
                # r0 loop 
                if self.model_parameters[0] is HELIX_UNROLL_R0_MODE:
                    #r0_list = self._get_samples(SAMPLE, invert=False)  
                    r0_list = self._read_samples(self.model_parameters[4][loop])
                    r0_list = r0_list[ r0_list > MIN_R0 ]
                    skip = int(r0_list.shape[0]/SAMPLE)
                    if skip > 1:
                        r0_list = r0_list[::skip]    
                    r0_list = np.tile(r0_list, 2)

                    for ii, r0 in enumerate(tqdm(r0_list)):
                        print ('\r steps: %d '%ii, end='',flush=True)
                    
                        dfh['cos_theta'] = dfh.r/2/r0
                        
                        r_inv = np.asarray(dfh['cos_theta'].values.astype(float))
                        r_inv_upd_ix = np.where(np.abs(r_inv) > 1 )[0]
                        r_inv[r_inv_upd_ix] = 1
                        dfh.cos_theta = r_inv
                        
                        #zero_ix = np.where(dfh['cos_theta'] == 0)
                        #non_zero_ix = np.where(dfh['cos_theta'] != 0)

                        if ii < r0_list.shape[0]/2:
                            dfh['a1'] = dfh['a0'] - np.arccos(dfh['cos_theta'])
                        else:
                            dfh['a1'] = dfh['a0'] + np.arccos(dfh['cos_theta'])
                        
                        # parameter space
                        #dfh['px'] = -dfh.r*np.cos(dfh.a1)*np.cos(dfh.a0) - dfh.r*np.sin(dfh.a1)*np.sin(dfh.a0)
                        #dfh['py'] = -dfh.r*np.cos(dfh.a1)*np.sin(dfh.a0) + dfh.r*np.sin(dfh.a1)*np.cos(dfh.a0)

                        #x0, y0
                        dfh['px'] = dfh.x + r0*np.cos(dfh.a0)*np.cos(dfh.a1) + r0*np.sin(dfh.a0)*np.sin(dfh.a1)
                        dfh['py'] = dfh.y + r0*np.cos(dfh.a0)*np.sin(dfh.a1) - r0*np.sin(dfh.a0)*np.cos(dfh.a1)
                        

                        dfh['sina1'] = np.sin(dfh['a1'])
                        dfh['cosa1'] = np.cos(dfh['a1'])

                        # z / angle                  
                        #dfh['zarc'] = 0
                        #zarc = dfh.zarc.values
                        #zarc[zero_ix] = dfh.zshift.iloc[zero_ix]/dfh.r.iloc[zero_ix]
                        #TODO if the arc angle is greater than pi
                        #zarc[non_zero_ix] = dfh.zshift.iloc[non_zero_ix]/ (np.arcsin(dfh.cos_theta.iloc[non_zero_ix]) * 2 )
                        #dfh['zarc'] = dfh.zshift/ np.arcsin (dfh.cos_theta*2)
                        #dfh['zarc'] = np.log1p(np.absolute(dfh.zshift/ (np.arcsin(dfh.cos_theta.iloc[non_zero_ix]) * 2 * r0 )))*np.sign(dfh.zshift)
                        #dfh['zarc'] = zarc
                        dfh['zarc'] = np.log1p(np.absolute(dfh.zshift/ (np.arcsin(dfh.cos_theta) * 2 * r0 )))*np.sign(dfh.zshift)
                        
                        ss = StandardScaler()
                        dfs = ss.fit_transform(dfh[self.model_parameters[1]].values)
                        dfs = np.multiply(dfs, self.model_parameters[2])
                
                        if ii < r0_list.shape[0]/2:
                            self.clusters = DBSCAN(eps=DBSCAN_EPS + ii/SAMPLE*R0_STEP_EPS,min_samples=1, n_jobs=-1).fit(dfs).labels_
                        else:
                            self.clusters = DBSCAN(eps=DBSCAN_EPS + (ii-SAMPLE)/SAMPLE*R0_STEP_EPS,min_samples=1, n_jobs=-1).fit(dfs).labels_
                            
                        if 's1' not in dfh:
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

                #r mode
                else:
                    for ii in tqdm(np.arange(-STEPS, STEPS, 1)):
                        print ('\r steps: %d '%ii, end='',flush=True)
                       
                        dfh['a1'] = dfh['a0'] + (rr + STEPRR*rr**2)*ii/180*np.pi + (0.00001*ii)*dfh.z*np.sign(dfh.z)/180*np.pi
                        dfh['t1'] = dfh['a0'] - np.pi*(ii/STEPS) 
                        dfh['r0'] = dfh.r/np.cos(dfh.t1) 
                        # parameter space
                        dfh['px'] = -dfh.r*np.cos(dfh.a1)*np.cos(dfh.a0) - dfh.r*np.sin(dfh.a1)*np.sin(dfh.a0)
                        dfh['py'] = -dfh.r*np.cos(dfh.a1)*np.sin(dfh.a0) + dfh.r*np.sin(dfh.a1)*np.cos(dfh.a0)
                        
                        dfh['sina1'] = np.sin(dfh['a1'])
                        dfh['cosa1'] = np.cos(dfh['a1'])
                        
                        ss = StandardScaler()
                    
                        dfs = ss.fit_transform(dfh[self.model_parameters[1]].values)
                        dfs = np.multiply(dfs, self.model_parameters[2])
                
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

def display_score(event_id, hits, labels, truth, message):
    if truth is not None:
        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)
        print(message + "%d: %.8f" % (event_id, score))

def run_predictions(event_id, all_labels, all_hits, truth, model, label_file_root, unmatched_only=True, filter_hits=True, track_extension_limits=None, merge_overwrite_limit=4, one_phase_only=False):
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
            message = 'Processed dbscan loop ' + str(i+1) + ' score for event '
            display_score(event_id, all_hits, labels_full[i], truth, message)
        else:
            label_file_extend = label_file_root + '_dbscan' + str(i+1) + '_extend.csv'
            if os.path.exists(label_file_extend):
                print('Loading dbscan loop ' + str(i+1) + ' processed file: ' + label_file_extend)
                labels_full.append(pd.read_csv(label_file_extend).label.values)
                message = 'Processed extended dbscan loop ' + str(i+1) + ' score for event '
                display_score(event_id, all_hits, labels_full[i], truth, message)
            else:
                labels_subset[i] = merge.renumber_labels(labels_subset[i])

                # If only predicting on unmatched hits, we still need to generate a full set of
                # labeled tracks for track extension, merging, etc. Set any hits assigned from
                # the previous phase to 0. We will merge with the tracks from the previous phase
                # at the end, after track extension and merging of the unmatched hits.
                if unmatched_only:
                    labels_full.append(np.copy(all_labels))
                    labels_subset[i][labels_subset[i] == 0] = 0 - len(all_labels) - 1
                    labels_subset[i] = labels_subset[i] + len(all_labels) + 1
                    labels_full[i][labels_full[i] != 0] = -1
                    labels_full[i][labels_full[i] == 0] = labels_subset[i]
                    labels_full[i][labels_full[i] == -1] = 0
                else:
                    labels_full.append(np.copy(labels_subset[i]))

                message = 'Unfiltered dbscan loop ' + str(i+1) + ' score for event '
                display_score(event_id, all_hits, labels_full[i], truth, message)

                # If desired, extend tracks
                if track_extension_limits is not None:
                    num_neighbours = 18
                    if one_phase_only:
                        num_neighbours = 25
                    labels_full[i] = xtnd.do_all_track_extensions(labels_full[i], all_hits, track_extension_limits, num_neighbours)
                    labels_full[i] = merge.renumber_labels(labels_full[i])
                
                    message = 'Unfiltered extended dbscan loop ' + str(i+1) + ' score for event '
                    display_score(event_id, all_hits, labels_full[i], truth, message)
                else:
                    labels_full[i] = merge.renumber_labels(labels_full[i])

                # Save the predicted tracks to a file so we don't need to re-calculate next time.
                df = pd.DataFrame(labels_full[i])
                df.to_csv(label_file_extend, index=False, header=['label'])

            if filter_hits:
                # Filter out tracks that are too small, as well as hits that look
                # like outliers, i.e. duplicate-z values, slopes that do not match
                # other hits in the track, etc.
                labels_full[i] = merge.remove_outliers(labels_full[i], all_hits, smallest_track_size=6, print_counts=False)
                labels_full[i] = merge.renumber_labels(labels_full[i])

                message = 'Filtered non-outlier dbscan loop ' + str(i+1) + ' score for event '
                display_score(event_id, all_hits, labels_full[i], truth, message)
            # Save the predicted tracks to a file so we don't need to re-calculate next time.
            df = pd.DataFrame(labels_full[i])
            df.to_csv(label_file, index=False, header=['label'])

    # Merge all dbscan loop labels together
    merge_count = 0
    for i in range(len(labels_full)):
        if i == 0:
            labels_merged = labels_full[0]
        else:
            labels_merged = merge.heuristic_merge_tracks(labels_merged, labels_full[i], all_hits, overwrite_limit=merge_overwrite_limit, print_summary=False)
            merge_count = merge_count + 1
            # Periodically remove small tracks/noise to help merge performance.
            # If we're only dealing with unmatched hits from a previous round, don't filter though,
            # since we have relatively few tracks already, and removing even small tracks hurts.
            if merge_count % 3 == 0 and not unmatched_only:
                (labels_merged, _) = merge.remove_small_tracks(labels_merged, smallest_track_size=3)
            message = 'Merged loop 1-' + str(i+1) + ' score for event '
            display_score(event_id, all_hits, labels_merged, truth, message)

    # If we merged the unmatched hits only, we now need to merge those with the full
    # set of tracks from a previous phase. We assume the previous phase (all_labels)
    # has a much stronger set of tracks, so those should be listed first on the merge.
    if unmatched_only:
        labels_merged = merge.heuristic_merge_tracks(all_labels, labels_merged, all_hits, overwrite_limit=merge_overwrite_limit, print_summary=False)
        display_score(event_id, all_hits, labels_merged, truth, 'Final merged loop 1 score for event ')

    return (labels_merged)

def run_helix_unrolling_predictions(event_id, hits, truth, label_identifier, model_parameters, one_phase_only=False):
    # Shortcut - if we've previously generated and saved labels, just use them
    # rather than re-generating.
    label_file_root = 'event_' + str(event_id)+'_labels_' + label_identifier
    label_file = label_file_root + '.csv'
    if os.path.exists(label_file):
        print(str(event_id) + ': load ' + label_file)
        labels = pd.read_csv(label_file).label.values
        display_score(event_id, hits, labels, truth, 'Loaded score for event ')
        return labels

    print(str(event_id) + ': clustering on ' + label_identifier)

    model = Clusterer(model_parameters)
    
    # For the first run, we do not have an input array of labels/tracks.
    label_file_root1 = label_file_root + '_phase1'
    (labels) = run_predictions(event_id, None, hits, truth, model, label_file_root1, unmatched_only=False, filter_hits=True, track_extension_limits=EXTENSION_STANDARD_LIMITS, one_phase_only=one_phase_only)
    display_score(event_id, hits, labels, truth, 'Filtered 1st pass score for event ')

    if not one_phase_only:
        label_file_root2 = label_file_root + '_phase2'
        model = Clusterer(model_parameters)
        (labels) = run_predictions(event_id, labels, hits, truth, model, label_file_root2, unmatched_only=True, filter_hits=False, track_extension_limits=EXTENSION_STANDARD_LIMITS)
        display_score(event_id, hits, labels, truth, '2nd pass score for event ')

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
    # display_score(event_id, hits, labels, truth, '3rd pass score for event ')

    # # Save the generated labels, can avoid re-generation next run.

    return labels

def print_info(helix_id, model_parameters):
    unroll_mode = model_parameters[0]
    feature_matrix = model_parameters[1]
    scaled_distance = model_parameters[2]
    z_shift_matrix = model_parameters[3]

    print('==========================================================================')
    print('Helix model: ' + str(helix_id))
    print('Unroll mode: ' + str(unroll_mode))
    print('Feature matrix: ' + str(feature_matrix))
    print('Scaled distance: ' + str(scaled_distance))
    print('z shift matrix: ' + str(z_shift_matrix))
    print('==========================================================================')
   

def predict_event(event_id, hits, train_or_test, truth):
    
    #DBSCAN_EPS_MATRIX = [0.0033, 0.0041, 0.0037, 0.0045]

    # Set up z/r, needed for outlier detection code
    hits['r'] = np.sqrt(hits.x**2+hits.y**2)
    hits['zr'] = hits['z'] / hits['r']

    #The strongest model is  [3, 4, 10, -3]
    model_parameters = []
    model_parameters.append(HELIX_UNROLL_R_MODE)
    model_parameters.append(FEATURE_MATRIX)
    model_parameters.append(SCALED_DISTANCE)
    model_parameters.append([3, -6, 4, 12, -9, 10, -3, 6, -10, 2, 8, -2])
    
    print_info(1, model_parameters)      
    labels_helix1 = run_helix_unrolling_predictions(event_id, hits, truth, train_or_test + '_helix1', model_parameters)
    

    model_parameters.clear()
    model_parameters.append(HELIX_UNROLL_R_MODE)
    model_parameters.append(FEATURE_MATRIX_2)
    model_parameters.append(SCALED_DISTANCE_2)
    model_parameters.append([3, -6, 4, 12, -9, 10, -3, 6, -10, 2, 8, -2])
    
    print_info(2, model_parameters)
    # Running helix2 in 2 phases hurts our score, so do a single-phase only
    labels_helix2 = run_helix_unrolling_predictions(event_id, hits, truth, train_or_test + '_helix2', model_parameters, one_phase_only=True)

    model_parameters.clear()
    model_parameters.append(HELIX_UNROLL_R_MODE)
    model_parameters.append(FEATURE_MATRIX_3)
    model_parameters.append(SCALED_DISTANCE_3)
    model_parameters.append([3])
    print_info(3, model_parameters)
    labels_helix3 = run_helix_unrolling_predictions(event_id, hits, truth, train_or_test + '_helix3', model_parameters, one_phase_only=True)

    model_parameters.clear()
    model_parameters.append(HELIX_UNROLL_R_MODE)
    model_parameters.append(FEATURE_MATRIX_4)
    model_parameters.append(SCALED_DISTANCE_4)
    model_parameters.append([-1])
    print_info(4, model_parameters)
    labels_helix4 = run_helix_unrolling_predictions(event_id, hits, truth, train_or_test + '_helix4', model_parameters, one_phase_only=True)

    model_parameters.clear()
    model_parameters.append(HELIX_UNROLL_R_MODE)
    model_parameters.append(FEATURE_MATRIX_5)
    model_parameters.append(SCALED_DISTANCE_5)
    model_parameters.append([2])
    print_info(5, model_parameters)
    labels_helix5 = run_helix_unrolling_predictions(event_id, hits, truth, train_or_test + '_helix5', model_parameters, one_phase_only=True)

    
    model_parameters.clear()
    model_parameters.append(HELIX_UNROLL_R0_MODE)
    model_parameters.append(FEATURE_MATRIX_6)
    model_parameters.append(SCALED_DISTANCE_6)
    #model_parameters.append([3, -6, 4, 12, -9, 10, -3, 6, -10, 2, 8, -2])  
    model_parameters.append([0])
    model_parameters.append(R0_SAMPLE_EVENT)
    print_info(6, model_parameters)
    labels_helix6 = run_helix_unrolling_predictions(event_id, hits, truth, train_or_test + '_helix6', model_parameters, one_phase_only=False)

    

    # Merge results from two sets of predictions, removing outliers first
    labels_helix1 = merge.remove_outliers(labels_helix1, hits, print_counts=False)
    labels_helix2 = merge.remove_outliers(labels_helix2, hits, print_counts=False)
    labels_helix3 = merge.remove_outliers(labels_helix3, hits, print_counts=False)
    labels_helix4 = merge.remove_outliers(labels_helix4, hits, print_counts=False)
    labels_helix5 = merge.remove_outliers(labels_helix5, hits, print_counts=False)
    labels_helix6 = merge.remove_outliers(labels_helix6, hits, print_counts=False)
    
    
    display_score(event_id, hits, labels_helix1, truth, 'After outlier removal helix1 ')
    display_score(event_id, hits, labels_helix2, truth, 'After outlier removal helix2 ')
    display_score(event_id, hits, labels_helix3, truth, 'After outlier removal helix3 ')
    display_score(event_id, hits, labels_helix4, truth, 'After outlier removal helix4 ')
    display_score(event_id, hits, labels_helix5, truth, 'After outlier removal helix5 ')
    display_score(event_id, hits, labels_helix6, truth, 'After outlier removal helix6 ')


    labels = merge.heuristic_merge_tracks(labels_helix1, labels_helix2, hits, overwrite_limit=6, print_summary=False)
    display_score(event_id, hits, labels, truth, 'Merged helix1&2 unrolling for event ')

    labels = merge.heuristic_merge_tracks(labels, labels_helix3, hits, overwrite_limit=6, print_summary=False)
    display_score(event_id, hits, labels, truth, 'Merged helix1&2&3 unrolling for event ')
    
    labels = merge.heuristic_merge_tracks(labels, labels_helix4, hits, overwrite_limit=6, print_summary=False)
    display_score(event_id, hits, labels, truth, 'Merged helix1&2&3&4 unrolling for event ')

    labels = merge.heuristic_merge_tracks(labels, labels_helix5, hits, overwrite_limit=6, print_summary=False)
    display_score(event_id, hits, labels, truth, 'Merged helix1&2&3&4&5 unrolling for event ')

    labels = merge.heuristic_merge_tracks(labels, labels_helix6, hits, overwrite_limit=6, print_summary=False)
    display_score(event_id, hits, labels, truth, 'Merged helix1&2&3&4&5&6 unrolling for event ')

    labels = strt.extend_straight_tracks(labels, hits)
    display_score(event_id, hits, labels, truth, 'Merged helix1&2&3&4&5&6 unrolling straight-extended for event ')


    labels = free.assign_free_hits(labels, hits)
    display_score(event_id, hits, labels, truth, 'Merged free-hit-reassigned score for event ')


    return labels


def run_single_threaded_training(skip, nevents):
    path_to_train = os.path.join(INPUT_PATH, 'train_1')
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

    path_to_test = os.path.join(INPUT_PATH, 'test')
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

