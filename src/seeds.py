import numpy as np
import pandas as pd

def matches_truth_first_hits(labels, track, truth):
    match_found = False
    all_indices = np.where(labels == track)[0]
    truth_for_track = truth.iloc[all_indices,:]
    sorted_truth = truth_for_track.sort_values('tz_abs')
    particles = sorted_truth['particle_id'].values[0:5]
    particles = particles[particles != 0]
    if (len(particles) > 0):
        unique, counts = np.unique(particles, return_counts=True)
        max_count = np.amax(counts)
        if max_count >= 3:
            match_found = True
    return match_found

def matches_truth_last_hits(labels, track, truth):
    match_found = False
    all_indices = np.where(labels == track)[0]
    truth_for_track = truth.iloc[all_indices,:]
    sorted_truth = truth_for_track.sort_values('tz_abs')
    particles = sorted_truth['particle_id'].values[-5:]
    particles = particles[particles != 0]
    if (len(particles) > 0):
        unique, counts = np.unique(particles, return_counts=True)
        max_count = np.amax(counts)
        if max_count >= 3:
            match_found = True
    return match_found

def renumber_labels(labels):
    new_label = 0
    for old_label in np.unique(labels):
        if not old_label == new_label:
            labels[labels == old_label] = new_label
        new_label += 1

    return labels

def count_truth_track_seed_hits(labels, truth, print_results=False):
    labels = renumber_labels(labels)
    max_track = np.amax(labels)
    seeds = 0
    truth['tz_abs'] = truth['tz'].abs()
    for i in range(1,max_track+1):
        found_it = matches_truth_first_hits(labels, i, truth) or matches_truth_last_hits(labels, i, truth)
        if found_it:
            seeds = seeds + 1

    if print_results:
        truth_tracks = truth['particle_id'].values
        total_tracks = len(np.unique(truth_tracks))
        print('total truth tracks: ' + str(total_tracks))
        print('predicted tracks: ' + str(max_track))
        print('Found seeds: ' + str(seeds))
    return seeds

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