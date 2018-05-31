import numpy as np
import pandas as pd
import math
import collections as coll

def do_seed_match_internal(pred_truth_for_track, num_hits, match_head):
    seed = 0
    perfect_match = 0
    if match_head:
        particles = pred_truth_for_track['particle_id'].values[0:num_hits]
    else:
        particles = pred_truth_for_track['particle_id'].values[-num_hits:]
    particles = particles[particles != 0]
    if (len(particles) > 0):
        unique, counts = np.unique(particles, return_counts=True)
        max_count = np.amax(counts)
        if max_count >= math.ceil(num_hits/2):
            seed = coll.Counter(particles).most_common(1)[0][0]
            if max_count == num_hits:
                perfect_match = 1
    return seed, perfect_match

def matches_truth_first_hits(pred_truth_for_track, num_hits):
    return do_seed_match_internal(pred_truth_for_track, num_hits, True)

def matches_truth_last_hits(pred_truth_for_track, num_hits):
    return do_seed_match_internal(pred_truth_for_track, num_hits, False)

# head majority matches, head perfect matches, tail majority matches, tail perfect matches, head/tail same
def internal_count_truth_seed_pred(labels, track, truth, num_hits):
    all_indices = np.where(labels == track)[0]
    pred_truth_for_track = truth.iloc[all_indices,:]
    sorted_truth = pred_truth_for_track.sort_values('tz_abs')
    (head_seed, head_perfect) = matches_truth_first_hits(sorted_truth, num_hits)
    (tail_seed, tail_perfect) = matches_truth_last_hits(sorted_truth, num_hits)
    head_match = int(head_seed != 0)
    tail_match = int(tail_seed != 0)
    head_tail_match = int((head_seed != 0) and (head_seed == tail_seed))
    return (head_match, head_perfect, tail_match, tail_perfect, head_tail_match)


def renumber_labels(labels):
    new_label = 0
    for old_label in np.unique(labels):
        if not old_label == new_label:
            labels[labels == old_label] = new_label
        new_label += 1

    return labels

# To return:
# head majority matches, head perfect matches, tail majority matches, tail perfect matches, head/tail same
def count_truth_track_seed_hits(labels, truth, print_results=False):
    labels = renumber_labels(labels)
    max_track = np.amax(labels)
    seeds = 0
    truth['tz_abs'] = truth['tz'].abs()
    head_matches = 0
    head_matches_perfect = 0
    tail_matches = 0
    tail_matches_perfect = 0
    head_tail_matches = 0
    for i in range(1,max_track+1):
        (head_match, head_perfect, tail_match, tail_perfect, head_tail_match) = internal_count_truth_seed_pred(labels, i, truth, 3)
        head_matches = head_matches + head_match
        head_matches_perfect = head_matches_perfect + head_perfect
        tail_matches = tail_matches + tail_match
        tail_matches_perfect = tail_matches_perfect + tail_perfect
        head_tail_matches = head_tail_matches + head_tail_match

    if print_results:
        truth_tracks = truth['particle_id'].values
        total_tracks = len(np.unique(truth_tracks))
        print('total truth tracks: ' + str(total_tracks))
        print('predicted tracks: ' + str(max_track))
        print('Found head seeds: ' + str(head_matches))
        print('Found head perfect seeds: ' + str(head_matches_perfect))
        print('Found tail seeds: ' + str(tail_matches))
        print('Found tail perfect seeds: ' + str(tail_matches_perfect))
        print('Found head+tail matches: ' + str(head_tail_matches))
    return head_matches

def find_first_seeds(labels, num_seeds, hits):
    seeds = np.copy(labels)
    seeds.fill(0)
    hits['z_abs'] = hits['z'].abs()
    tracks = np.unique(labels)
    for track in tracks:
        all_indices = np.where(labels == track)[0]
        hits_for_track = hits.iloc[all_indices,:]
        first_hits = hits_for_track.sort_values('z_abs')[:num_seeds].index
        for i in range(0,num_seeds):
            seeds[first_hits[i]] = track

    return seeds