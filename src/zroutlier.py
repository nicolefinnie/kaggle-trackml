import numpy as np
import pandas as pd
import math
import collections as coll

def classify_zr_shape(abs_zrs, diff_zrs):
    """
    5 shapes:
     0: unknown (no trend identified)
     1: increasing values
     2: decreasing values
     3: hill (increase then decrease)
     4: valley (decrease then increase)
    """
    def slice_diffs(diff_zrs):
        slices = []
        len_diffs = len(diff_zrs)
        if len_diffs > 12:
            len_part_diffs = int(len_diffs/3)
            slices.append(diff_zrs[0:len_part_diffs])
            slices.append(diff_zrs[len_part_diffs:2*len_part_diffs])
            slices.append(diff_zrs[2*len_part_diffs:])
        else:
            len_part_diffs = int(len_diffs/2)
            slices.append(diff_zrs[0:len_part_diffs])
            slices.append(diff_zrs[len_part_diffs:])
        return slices

    def trend_increase(diff_zrs):
        ret = False
        len_diffs = len(diff_zrs)
        if np.all(diff_zrs >= 0):
            ret = True
        elif len(np.where(diff_zrs >= 0)[0]) >= math.ceil(0.6*len_diffs):
            slices = slice_diffs(diff_zrs)
            ret = True
            for sample in slices:
                if len(np.where(sample >= 0)[0]) < math.ceil(0.5*len(sample)):
                    ret = False
                    break
        return ret

    def trend_decrease(diff_zrs):
        ret = False
        len_diffs = len(diff_zrs)
        if np.all(diff_zrs <= 0):
            ret = True
        elif len(np.where(diff_zrs <= 0)[0]) >= math.ceil(0.6*len_diffs):
            slices = slice_diffs(diff_zrs)
            ret = True
            for sample in slices:
                if len(np.where(sample <= 0)[0]) < math.ceil(0.5*len(sample)):
                    ret = False
                    break
        return ret

    def trend_hill(diff_zrs):
        ret = False
        slices = slice_diffs(diff_zrs)
        if trend_increase(slices[0]):
            for ix, sample in enumerate(slices):
                if ix == 0: continue
                if trend_decrease(sample):
                    ret = True
                    break
        return ret

    def trend_valley(diff_zrs):
        ret = False
        slices = slice_diffs(diff_zrs)
        if trend_decrease(slices[0]):
            for ix, sample in enumerate(slices):
                if ix == 0: continue
                if trend_increase(sample):
                    ret = True
                    break
        return ret
    
    shape = 0
    if trend_increase(diff_zrs):
        shape = 1
    elif trend_decrease(diff_zrs):
        shape = 2
    elif trend_hill(diff_zrs):
        shape = 3
    elif trend_valley(diff_zrs):
        shape = 4
        
    return shape

# TODO pi, -pi discontinuity 
def find_track_outliers_zr(track, labels, hits, truth=None, debug=False):
    def find_extreme_jump(diff_zrs, head_threshold=10, tail_threshold=20):
        """The idea is to find jumps in the z/r value that are much more
        extreme than usual, by default 10-20x larger than the mean jump.
        Those extremes are classified as potential outliers."""
        rem_ix = -1
        mean_diff_zr = diff_zrs.mean()
        filtered_diffs = diff_zrs[np.where(diff_zrs < 4*mean_diff_zr)[0]]
        new_mean = filtered_diffs.mean()
        min_removal_factor = head_threshold
        for ix, diff in enumerate(diff_zrs):
            # for curved tracks, the end of the track can have larger
            # values, so increase the removal threshold in the last 3rd
            if ix > int(len(diff_zrs)*0.65):
                min_removal_factor = tail_threshold
            if diff > min_removal_factor*new_mean:
                if ix == 0:
                    # May need to pick better candidate here. Makes sense to remove
                    # the first element (0) though - if the second element (1) was an
                    # outlier, that should cause the next diff magnitude to be wrong too.
                    rem_ix = 0
                else:
                    rem_ix = ix + 1
                break
        return rem_ix
    
    def find_opposing_extreme_jump(diff_zrs, head_threshold=10, tail_threshold=20, reverse_opt=False):
        """The idea is to look for two large jumps in opposing directions, with up
        to two points in between. The first large jump is likely the incorrect one."""
        rem_ix = []
        abs_diff_zrs = np.absolute(diff_zrs)

        # First try to filter out any extreme positive or negative values to get
        # a more accurate mean. In some cases, this can filter out all values,
        # so retry using the mean of the absolute values.
        mean_diff_zr = diff_zrs.mean()
        filtered_diffs = diff_zrs[np.where(abs_diff_zrs < 4*mean_diff_zr)[0]]
        if (filtered_diffs.shape[0] == 0):
            mean_diff_zr = abs_diff_zrs.mean()
            filtered_diffs = abs_diff_zrs[np.where(abs_diff_zrs < 4*mean_diff_zr)[0]]
            new_mean = filtered_diffs.mean()
        else:
            new_mean = filtered_diffs.mean()

        min_removal_factor = head_threshold
        first_jump_ix = -1
        first_jump_val = 0
        second_jump_ix = -1
        second_jump_val = 0
        second_jump_factor = 1
        for ix, diff in enumerate(diff_zrs):
            # for curved tracks, the end of the track can have larger
            # values, so increase the removal threshold in the last 3rd
            if ix > int(len(diff_zrs)*0.65):
                min_removal_factor = tail_threshold
            if abs(diff) > min_removal_factor*new_mean*second_jump_factor:
                if first_jump_ix == -1:
                    first_jump_ix = ix
                    first_jump_val = diff
                    second_jump_factor = 0.8
                else:
                    second_jump_ix = ix
                    second_jump_val = diff
                    break
        if (first_jump_ix != -1) and (second_jump_ix != -1) and ((second_jump_ix - first_jump_ix) <= 3):
            # Compare the jumps and distance between jumps to determine
            # potential outliers
            if ((first_jump_val >= 0) and (second_jump_val <= 0)) or ((first_jump_val <= 0) and (second_jump_val >= 0)):
                abs_diff = abs(first_jump_val + second_jump_val)
                if first_jump_ix == 0 and first_jump_val < 0 and (abs(first_jump_val) > 10*new_mean) and (abs(first_jump_val) > abs(second_jump_val)*1.2):
                    rem_ix.append(0)
                elif reverse_opt and first_jump_val < 0 and first_jump_ix == 1 and diff_zrs[0] < 0 and (abs(first_jump_val) > 10*new_mean) and diff_zrs[first_jump_ix+1] > 0:
                    rem_ix.append(0)
                    rem_ix.append(1)
                elif reverse_opt and first_jump_val < 0 and first_jump_ix == 1 and diff_zrs[0] > 0 and (abs(first_jump_val) > 10*new_mean) and diff_zrs[first_jump_ix+1] > 0:
                    rem_ix.append(1)
                elif (abs(second_jump_val) < abs(first_jump_val)) or (abs_diff < 0.02*abs(first_jump_val)):
                    for i in range(first_jump_ix, second_jump_ix):
                        rem_ix.append(i+1)
        return rem_ix

    def find_negative_extreme_jump(diff_zrs, zrs, mean_diffs):
        """The idea is to look for the biggest negative jump in a trending positive slope."""
        most_negative = 0
        most_negative_ix = -1
        rem_ix = -1
        for ix, diff in enumerate(diff_zrs):
            if diff < most_negative:
                most_negative = diff
                if ix == 0:
                    # Should be a better way to choose between 0 or 1 as outlier.
                    # Using 0 yields a better score, and is safer - track extension
                    # can recover lost hits from the end
                    most_negative_ix = 0
                else:
                    most_negative_ix = ix + 1
        if (most_negative < 0) and (abs(most_negative) > abs(mean_diffs)*2):
            # can have cases like 0.01, 0.01, 0.5, -0.4, 0.1. -0.4 looks like
            # the outlier, except it's likely that 0.5 was too big a jump.
            # look at neighbours to see which one is likely the outlier.
            if most_negative_ix > 1 and (diff_zrs[most_negative_ix-2] > 0) and (abs(diff_zrs[most_negative_ix-2]) > abs(diff_zrs[most_negative_ix-1])):
                most_negative_ix = most_negative_ix - 1

            # below should be better than just picking 0, but gives worse score,
            # maybe better to just always remove end, since it can be extended anyways...
            #elif most_negative_ix == 0:
            #    if (zrs[2] > zrs[0]) and (zrs[2] < zrs[1]):
            #        most_negative_ix = 1

            rem_ix = most_negative_ix

        return rem_ix

    def find_positive_extreme_jump(diff_zrs, zrs, mean_diffs):
        """The idea is to look for the biggest positive jump in a trending negative slope."""
        most_positive = 0
        most_positive_ix = -1
        rem_ix = -1
        for ix, diff in enumerate(diff_zrs):
            if diff > most_positive:
                most_positive = diff
                if ix == 0:
                    # Default to 0, code below will switch to 1 if that looks like
                    # the real outlier.
                    most_positive_ix = 0
                else:
                    most_positive_ix = ix + 1
        if (most_positive > 0) and (most_positive > abs(mean_diffs)*2):
            # can have cases like -0.01, -0.01, -0.5, 0.4, -0.1. 0.4 looks like
            # the outlier, except it's likely that -0.5 was too big a jump.
            # look at neighbours to see which one is likely the outlier.
            if most_positive_ix > 1 and (diff_zrs[most_positive_ix-2] < 0) and (abs(diff_zrs[most_positive_ix-2]) > abs(diff_zrs[most_positive_ix-1])):
                most_positive_ix = most_positive_ix - 1
            elif most_positive_ix == 0 and (zrs[2] < zrs[0]) and (zrs[2] > zrs[1]):
                most_negative_ix = 1

            rem_ix = most_positive_ix

        return rem_ix

    outlier_ix = []
    hit_ix = np.where(labels == track)[0]

    # Need at least 4 values to determine if any look like outliers
    if len(hit_ix) < 4:
        return outlier_ix

    df = hits.loc[hit_ix]        
    df = df.sort_values('z')
    hit_ix2 = df.index.values
    
    zr_values = df['zr'].values
    abs_zrs = np.absolute(zr_values)
    diff_zrs = np.diff(abs_zrs)
    abs_diff_zrs = np.absolute(diff_zrs)
    min_zr = zr_values.min()
    max_zr = zr_values.max()
    mean_diff_zr = diff_zrs.mean()
    median_zr = abs(np.median(zr_values))
    allowed_min = median_zr * 0.95
    allowed_max = median_zr * 1.05
    outlier_min = median_zr * 0.1
    outlier_max = median_zr * 3.0
    count_outliers = 0

    # If all diffs < 5% of the median value, track seems good
    if np.all(abs_diff_zrs < (median_zr * 0.05)):
        return outlier_ix

    shape = classify_zr_shape(abs_zrs, diff_zrs)

    if shape != 1 and shape != 2:
        return outlier_ix

    rem_stage = 0
    rem_ix = -1
    new_mean = mean_diff_zr # Should re-calculate this after removing the outlier.
    if (np.all(diff_zrs >= 0) or np.all(diff_zrs <= 0)):
        # Check for scale - any excessive changes indicate a potential outlier
        rem_ix = find_extreme_jump(abs_diff_zrs)
        if rem_ix != -1:
            outlier_ix.append(hit_ix2[rem_ix])
    elif shape == 1 or shape == 2:
        # Trending positive (1) or negative (2) slope
        if shape == 1:
            rem_ixes = find_opposing_extreme_jump(diff_zrs, head_threshold=10, tail_threshold=20, reverse_opt=False)
        else:
            # Convert our negative slope to a positive slope
            ndiff_zrs = np.copy(diff_zrs) * -1
            rem_ixes = find_opposing_extreme_jump(ndiff_zrs, head_threshold=10, tail_threshold=20, reverse_opt=True)

        if len(rem_ixes) > 0:
            rem_stage = 1
            rem_ix = rem_ixes[0]
            for ix in rem_ixes:
                outlier_ix.append(hit_ix2[ix])
        else:
            rem_ix = find_extreme_jump(abs_diff_zrs, head_threshold=20, tail_threshold=30)
            if rem_ix != -1:
                rem_stage = 2
                outlier_ix.append(hit_ix2[rem_ix])
            else:
                if shape == 1:
                    rem_ix = find_negative_extreme_jump(diff_zrs, zr_values, new_mean)
                else:
                    rem_ix = find_positive_extreme_jump(diff_zrs, zr_values, new_mean)
                if rem_ix != -1:
                    rem_stage = 3
                    outlier_ix.append(hit_ix2[rem_ix])

    elif False and shape == 1:
        rem_ixes = find_opposing_extreme_jump(diff_zrs, head_threshold=10, tail_threshold=20)
        if len(rem_ixes) > 0:
            rem_stage = 1
            rem_ix = rem_ixes[0]
            for ix in rem_ixes:
                outlier_ix.append(hit_ix2[ix])
        else:
            rem_ix = find_extreme_jump(abs_diff_zrs, head_threshold=20, tail_threshold=30)
            if rem_ix != -1:
                rem_stage = 2
                outlier_ix.append(hit_ix2[rem_ix])
            else:
                rem_ix = find_negative_extreme_jump(diff_zrs, zr_values, new_mean)
                if rem_ix != -1:
                    rem_stage = 3
                    outlier_ix.append(hit_ix2[rem_ix])
    elif False and shape == 2:
        ndiff_zrs = diff_zrs * -1
        rem_ixes = find_opposing_extreme_jump(ndiff_zrs, head_threshold=10, tail_threshold=20, reverse_opt=True)
        if len(rem_ixes) > 0:
            rem_stage = 1
            rem_ix = rem_ixes[0]
            for ix in rem_ixes:
                outlier_ix.append(hit_ix2[ix])
        else:
            rem_ix = find_extreme_jump(abs_diff_zrs, head_threshold=20, tail_threshold=30)
            if rem_ix != -1:
                rem_stage = 2
                outlier_ix.append(hit_ix2[rem_ix])
            else:
                rem_ix = find_positive_extreme_jump(diff_zrs, zr_values, new_mean)
                if rem_ix != -1:
                    rem_stage = 3
                    outlier_ix.append(hit_ix2[rem_ix])

    if rem_ix == -1:
        return outlier_ix

    #print(str(shape) + ', ' + str(rem_ix) + ', ' + str(new_mean) + ', ' + str(mean_diff_zr) + ', ' + str(diff_zrs))# + ', ' + str(abs_zrs))
    #print('ami: ' + str(allowed_min) + ', amx: ' + str(allowed_max) + ', all: ' + str(abs_zrs))
    #print(diff_zrs)
    #print(hit_ix2)
        
    if truth is not None:
        tdf = truth.loc[hit_ix]
        truth_count = coll.Counter(tdf.particle_id.values).most_common(2)
        truth_particle_id = truth_count[0][0]
        if truth_particle_id == 0 and len(truth_count) > 1:
            truth_particle_id = truth_count[1][0]
        truth_ix = []
        count_true = 0
        count_false = 0
        for ix in hit_ix2:
            truth_ix.append(truth.loc[ix].particle_id == truth_particle_id)
            if truth.loc[ix].particle_id == truth_particle_id:
                count_true = count_true + 1
            else:
                count_false = count_false + 1
        tt_ix = np.where(truth.particle_id.values == truth_particle_id)[0]
        majority1 = (count_true >= count_false)
        majority2 = (count_true >= int(len(tt_ix)/2))
        #print(str(len(hit_ix2)) + ' ' + str(majority1) + ' ' + str(majority2) + ', Truth length: ' + str(len(tt_ix)) + ', True: ' + str(count_true) + ', False: ' + str(count_false))
        #print(truth_ix)
        #if truth_ix[rem_ix] == False:
        #    print('AWESOME: ' + str(truth_ix))
        #else:
        if debug and count_true > 5 and truth_ix[rem_ix] == True:
            print(str(shape) + ', ' + str(rem_ix) + ', ' + str(new_mean) + ', ' + str(mean_diff_zr) + ', ' + str(diff_zrs))# + ', ' + str(abs_zrs))
            print('CRAPPY:  ' + str(rem_stage) + ', ' + str(truth_ix))

    return outlier_ix


def remove_outliers_zr(labels, hits):
    labels = np.copy(labels)
    tracks = np.unique(labels)
    hits['z_abs'] = hits.z.abs()
    hits['r'] = np.sqrt(hits.x**2+hits.y**2)
    hits['zr'] = hits['z'] / hits['r']
    count_rem_zr_slope = 0
    for track in tracks:
        if track == 0:
            continue
        track_hits = np.where(labels == track)[0]
        if len(track_hits) > 4:
            outliers = find_track_outliers_zr(track, labels, hits)
            if len(outliers) > 0:
                count_rem_zr_slope = count_rem_zr_slope + len(outliers)
                for oix in outliers:
                    labels[oix] = 0
            
    print('zr outliers removed: ' + str(count_rem_zr_slope))

    return labels

def safe_outlier_removal(labels, hits, truth, debug=False):
    labels = np.copy(labels)
    tracks = np.unique(labels)
    hits['z_abs'] = hits.z.abs()
    hits['r'] = np.sqrt(hits.x**2+hits.y**2)
    hits['zr'] = hits['z'] / hits['r']
    count_removed = 0
    count_not_removed = 0
    for track in tracks:
        if track == 0:
            continue
        track_hits = np.where(labels == track)[0]
        if len(track_hits) > 3:
            outlier_ix = find_track_outliers_zr(track, labels, hits, truth=truth, debug=debug)
            if len(outlier_ix) > 0:
                tdf = truth.loc[track_hits]
                truth_count = coll.Counter(tdf.particle_id.values).most_common(1)
                truth_particle_id = truth_count[0][0]
                for out_ix in outlier_ix:
                    if tdf.loc[out_ix].particle_id != truth_particle_id:
                        labels[out_ix] = 0
                        count_removed = count_removed + 1
                    else:
                        count_not_removed = count_not_removed + 1

    print('safe count_removed: ' + str(count_removed))
    print('safe count_not_removed: ' + str(count_not_removed))
    return labels


