"""
Eli Bulger

January 2020

LiMN Lab - Carnegie Mellon University

This contains functions to help show a sequence of visual stimuli with changing characteristics

"""

from helpers import *
from numpy import random as rd
import warnings
import numpy as np
import math


# ----------------------------------------------------------------------------------
#                          Generate a Random Trajectory based on Frequencies
# __________________________________________________________________________________

def generate_signal(T, dt, minmaxRange, freq_lims, seed=None):
    """

    :param T:
    :param dt:
    :param minmaxRange:
    :param freq_lims:
    :param seed:
    :return:
    """

    import numpy as np
    import matplotlib.pyplot as plt

    mean = 0
    std = 1

    limit = freq_lims

    steps = int(T / dt)
    time = np.linspace(0, T, steps)
    freq = np.arange(steps) / T - (steps) / (2 * T)

    if seed is None:
        # set a seed randomly if one is not specified
        seed = rd.randint(100000000)

    # set the seed
    np.random.seed(seed)

    # generating the X(w) values
    w_real = np.random.normal(mean, std, size=int((steps) / 2 + 1))
    w_complex = np.random.normal(mean, std, size=int((steps) / 2 + 1)) * 1j

    # creating X(w) array
    w_right = w_real + w_complex
    w_conj = np.flipud(np.conjugate(w_right[1:]))  # 500 values, 2nd fft value starts at 0
    w_right = w_right[:-1]  # 500 values, 1st fft value starts at 0

    # generating in form where 0 frequency is the first element of the array
    x_S = np.fft.fftshift((np.concatenate([w_right, w_conj])))

    # impose limit on frequency
    x_S[np.logical_or(np.abs(freq) >= limit[1], np.abs(freq) <= limit[0])] = 0

    # transform to real signal
    x_t = np.real(np.fft.ifft(np.fft.ifftshift(x_S)))

    # rescale x matrix
    current_max = np.max(x_t)
    current_min = np.min(x_t)
    new_min = minmaxRange[0]
    new_max = minmaxRange[1]

    # normalize to pixel limits specified
    x_t = (new_max - new_min) / (current_max - current_min) * (x_t - current_max) + new_max

    """
    # rescale both
    if current_max > pixel_max and current_min < pixel_min:
        x_t = (pixel_max - pixel_min) / (current_max - current_min) * (x_t - current_max) + pixel_min

    # rescale upper
    elif current_max > pixel_max:
        x_t = (pixel_max - current_min) / (current_max - current_min) * (x_t - current_max) + pixel_min

    # rescale lower
    elif current_min < pixel_min:
        x_t = (current_max - pixel_min) / (current_max - current_min) * (x_t - current_max) + pixel_min

    # don't rescale otherwise
    """

    return x_t, x_S, time, freq, seed


def generate_degrees(signal, cycles, parts=False, stimsPerSequence=5, stimNum=0):
    """

    :param signal:
    :param cycles:
    :return:
    """

    max_deg = cycles * 360

    # create an array for going around the circle

    if parts is True:
        deg_chunk = max_deg / stimsPerSequence
        radian_array = np.linspace(deg_chunk*stimNum,
                                   deg_chunk*(stimNum+1),
                                   len(signal))

    else:
        radian_array = np.linspace(0,
                                   max_deg,
                                   len(signal))


    return radian_array


def generate_absent_trajectory(split_input_signal, param_dict=None, type='invert'):
    """

    :param split_input_signal:
    :param param_dict:
    :param type:
    :return:
    """

    # choose a stimulus to base off of
    stimulus_choice = rd.choice(split_input_signal)

    if type == 'invert':
        # invert
        signal_mean = np.mean(stimulus_choice)
        absent_probe = -stimulus_choice + 2 * signal_mean

    elif type == 'random':
        new_signal, comp, time_arr, freq, seed = generate_signal(T=param_dict.T,
                                                                 dt=param_dict.dt,
                                                                 minmaxRange=param_dict.F0_limit,
                                                                 freq_lims=param_dict.req_lims)
        # split and choose a random stimulus
        split_new_signal = split_array(new_signal, param_dict.stimsPerSequence)
        absent_probe = rd.choice(split_new_signal)

    # get metrics for mean difference, mean derivative difference

    difference = np.zeros((len(split_input_signal),))
    der_difference = np.zeros((len(split_input_signal),))

    # go through all possible elements in the original sequence, return the minimum differences
    for i, stim in enumerate(split_input_signal):

        # calc derivative arrays
        der_orig_array = np.gradient(stim)
        der_absent_array = np.gradient(absent_probe)

        # difference arrays
        diff_array = np.subtract(absent_probe, stim)
        diff_der_array = np.subtract(der_absent_array, der_orig_array)

        # mean differences - NOT NECESSARILY FROM THE SAME SIGNAL
        difference[i] = np.mean(np.abs(diff_array))
        der_difference[i] = np.mean(np.abs(diff_der_array))

    # return the set or two sets of stimuli that are closest to the absent stimulus
    norm_diffs = [np.min(difference), der_difference[np.where(difference == np.min(difference))]]
    der_diffs = [difference[np.where(der_difference == np.min(der_difference))], np.min(der_difference)]

    return absent_probe, norm_diffs, der_diffs


# ----------------------------------------------------------------------------------
#                         Discrete Elements
# __________________________________________________________________________________
# generates a set of sequences by calling rand_seq
def generate_stims_SD(nSequences, stimsPerSequence, colour_space_size=6, rand_distance=500, angle=30):
    """

    Generates stimuli and probe for same/different task

    :param nSequences:
    :param stimsPerSequence:
    :param colour_space_size:
    :param rand_distance:
    :return:
    """

    if not (nSequences % 2 == 0):
        raise Exception("Please ensure that nSequences is a multiple of 2")

    # get the same / different information set
    sd = np.array(['same', 'different'])
    infoSD = np.repeat(sd, nSequences / 2)
    rd.shuffle(infoSD)

    # store the information holding the stimuli details
    stimuli_tags = np.zeros((nSequences,), dtype=object)
    probe_tags = np.zeros((nSequences,), dtype=object)

    # store the sequence data and probe data
    for i in range(nSequences):

        # depending on the method we use, we will chenge this func
        stimuli_tags[i] = rand_seq_angle(stimsPerSequence,
                                         distance_in_pixels=rand_distance,
                                         angle=angle,
                                         colour_space_size=colour_space_size)
        if infoSD[i] == 'same':
            probe_tags[i] = stimuli_tags[i]


        if infoSD[i] == 'different':
            probe_tags[i] = switch_2cols(stimuli_tags[i])

    return stimuli_tags, probe_tags, infoSD


# generates a set of sequences by calling rand_seq
def generate_stims_AP(nSequences, stimsPerSequence, absentDiff, colour_space_size=6, rand_distance=500, angle=None):
    """

    Generates stimuli and probe for absent/present task

    :param nSequences:
    :param stimsPerSequence:
    :param absentDiff:
    :param colour_space_size:
    :param rand_distance:
    :return:
    """

    if not (nSequences % 2 == 0):
        raise Exception("Please ensure that nSequences is a multiple of 2")

    # get the absent / present information set
    sd = np.array(['absent', 'present'])
    infoAP = np.repeat(sd, nSequences / 2)
    rd.shuffle(infoAP)

    # store the information holding the stimuli details
    stimuli_tags = np.zeros((nSequences,), dtype=object)
    probe_tags = np.zeros((nSequences,), dtype=object)
    dist_from_avg = np.zeros((nSequences,), dtype=object)

    # store the sequence data and probe data
    for i in range(nSequences):
        # depending on the method we use, we will
        stimuli_tags[i] = rand_seq_angle(stimsPerSequence,
                                         distance_in_pixels=rand_distance,
                                         angle=angle,
                                         colour_space_size=colour_space_size)

    # determine what the probe stimulus should be
    for k in range(nSequences):
        if infoAP[k] == 'absent':
            # probe_tags[k] = generate_absent(stimuli_tags[k][:, rStim], difference=absentDiff)
            probe_tags[k], dist_from_avg[k] = generate_absent(stimuli_tags[k], difference=absentDiff)

        if infoAP[k] == 'present':
            # pick a stimulus to base probe off of
            rStim = np.random.choice(np.arange(stimsPerSequence))
            probe_tags[k] = stimuli_tags[k][:, rStim]

    return stimuli_tags, probe_tags, infoAP, dist_from_avg


# generates one random sequence of position, shape and colour (of pre-defined length)
def rand_seq(l_seq, distance_in_pixels, colour_space_size=6):
    """

    :param l_seq:
    :param distance_in_pixels:
    :return:
    """

    seq_tags = np.zeros((3, l_seq), dtype=object)
    shape = 'circle'

    r1_old, r2_old = None, None
    choice = None

    # generate sequence
    for k in range(l_seq):
        # function to generate points
        r1_old, r2_old = generate_spaced_points(distance_in_pixels, r1_old, r2_old)

        # store location data here
        seq_tags[0, k] = [r1_old, r2_old]

        # randomize colour
        # colour, choice = randomize_colour(colour_space_size, choice, chosen_colour='blue')
        colour, choice = [-1, -1, 1], -1

        seq_tags[1, k] = colour

        # set shape
        seq_tags[2, k] = shape

    return seq_tags


# generates one random sequence of position, shape and colour (of pre-defined length)
def rand_seq_angle(l_seq, distance_in_pixels, angle=None, colour_space_size=6):
    """

    generates random sequence, makes sure that next stimulus is at a certain angle from previous

    :param l_seq:
    :param distance_in_pixels:
    :param angle:
    :param colour_space_size:
    :return:
    """

    seq_tags = np.zeros((3, l_seq), dtype=object)
    shape = 'circle'

    r1p, r2p = None, None
    r1pp, r2pp = None, None

    choice = None
    stoplim = 0

    # generate sequence
    for k in range(l_seq):

        if k >= 1:
            # store 2nd back points
            r1_temp, r2_temp = r1p, r2p

        # function to generate points
        r1p, r2p = generate_spaced_points_angle(distance_in_pixels, angle, r1p, r2p, r1pp, r2pp)

        if r1p is False:
            # repeat the function then break so it doesn't repeat again
            stoplim += 1
            if stoplim > 10000:
                raise Exception('Your distance/angle specifications are too strict - try again with better values!')
            seq_tags = rand_seq_angle(l_seq, distance_in_pixels, angle=angle)
            break

        # store location data here
        seq_tags[0, k] = [r1p, r2p]

        # randomize colour
        # colour, choice = randomize_colour(colour_space_size, choice, chosen_colour='blue')
        colour, choice = [-1, -1, 1], -1

        seq_tags[1, k] = colour

        # set shape
        seq_tags[2, k] = shape

        if k >= 1:
            # set 2nd back points
            r1pp, r2pp = r1_temp, r2_temp

    return seq_tags


def generate_absent(stimulus, difference):
    """
    Generates stimulus parameters with specified differences from the input stimulus

    :param stimulus: all pieces of sequence
    :param difference:
    :param colour_space_size:
    :return:
    """

    # stimulus
    absent_stim = np.zeros((3,), dtype=object)

    # locations
    stimulus_set = stimulus[0, :]

    # get average of x points
    # get average of y points
    x = 0
    y = 0

    for location in stimulus_set:
        x += location[0]
        y += location[1]

    x = x / len(stimulus_set)
    y = y / len(stimulus_set)

    # generate random cartesian coordinates - ensure its a specific distance away from all other points
    # r1, r2 = generate_random_points(min_dist=difference[0], r1_old=stimulus[0][0], r2_old=stimulus[0][1])
    r1, r2 = select_random_points(locations=stimulus_set, min_dist=difference[0])

    absent_stim[0] = [r1, r2]

    # note the difference in location from the points
    dist_from_avg = location_dist_cart(x, y, r1, r2)

    # generate colour with specified difference --- hardcoded that its blue here
    # absent_stim[1] = randomize_colour_dist(old_choice=stimulus[1][0], colour_diff=difference[1])
    # changed to make it always blue
    absent_stim[1] = [-1, -1, 1]

    # make it a circle
    absent_stim[2] = 'circle'

    return absent_stim, dist_from_avg


# ----------------------------------------------------------------------------------
#                         Randomizing parameters - locations
# __________________________________________________________________________________
# choosing points a specified distance away - on the surface of a circlular boundary
def generate_spaced_points(distance_in_pixels, r1_old=None, r2_old=None):
    """

    choosing points a specified distance away - on the surface of a circlular boundary

    updated to only generate close to origin

    :param distance_in_pixels:
    :param r1_old:
    :param r2_old:
    :return:
    """

    # raise an error is there's a problem with values not existing
    if ((r1_old is None) and (r2_old is not None)) \
            or ((r1_old is not None) and (r2_old is None)):
        raise Exception("there is an issue with assigning location values")

    # generate starting location if old points aren't yet defined
    elif (r1_old is None) and (r2_old is None):

        # r1_new = -900 + rd.rand() * 1800
        r1_new = -500 + rd.rand() * 1000
        r2_new = -500 + rd.rand() * 1000

    # check distance metric - (r1_old is not None) and (r2_old is not None)
    else:
        satisfied = False

        while satisfied is False:

            # generate points a specified distance away from previous points
            r1, r2 = gen_cart_dist_circle(distance_in_pixels)

            # check if points are valid
            r1_new = (r1_old + r1)
            r2_new = (r2_old + r2)

            # ensure it's within fixation distance
            # if np.abs(r1_new) < 900 and np.abs(r2_new) < 500:
            if location_dist_cart(x1=r1_new, x2=r2_new, x1new=0, x2new=0) < 500:
                satisfied = True

    return r1_new, r2_new


# choosing points a specified distance away - on the surface of a circlular boundary - controlling angle
def generate_spaced_points_angle(distance_in_pixels, angle, r1p=None, r2p=None, r1pp=None, r2pp=None):
    """

    choosing points a specified distance away - on the surface of a circlular boundary

    updated to only generate close to origin

    :param distance_in_pixels:
    :param angle:
    :param r1p:
    :param r2p:
    :param r1pp:
    :param r2pp:
    :return:
    """

    # generate starting location if old points aren't yet defined
    if (r1p is None) and (r2p is None):

        # r1_new = -900 + rd.rand() * 1800
        r1_new = -500 + rd.rand() * 1000
        r2_new = -500 + rd.rand() * 1000

    # if double back points are not correct, do standard procedure to generate new point
    elif (r1pp is None) and (r2pp is None):

        satisfied = False

        while satisfied is False:

            # generate points a specified distance away from previous points
            r1, r2 = gen_cart_dist_circle(distance_in_pixels)

            # check if points are valid
            r1_new = (r1p + r1)
            r2_new = (r2p + r2)

            # ensure it's within fixation distance
            # if np.abs(r1_new) < 900 and np.abs(r2_new) < 500:
            if location_dist_cart(x1=r1_new, x2=r2_new, x1new=0, x2new=0) < 500:
                satisfied = True

    # use old points, ensure that new point is a specified amount of degrees away
    else:

        satisfied = False

        # reference points to old point - now they are a set distance from the origin
        ref_r1 = r1p - r1pp
        ref_r2 = r2p - r2pp

        # get the angle (degrees currently) from the x axis
        old_angle = math.degrees(math.atan(ref_r2 / ref_r1))

        # list angles to try, shuffle to make it random
        angles = [angle, -angle]
        np.random.shuffle(angles)

        for ang in angles:

            # here are the parameters we want for the new point, references from origin, convert to cartesian
            new_angle = old_angle + ang
            radius = distance_in_pixels

            x, y = ppct.pol2cart(new_angle, radius, units='deg')

            # add new measures to the previous point in cartesian coords
            r1_new = x + r1p
            r2_new = y + r2p

            # ensure its
            if location_dist_cart(x1=r1_new, x2=r2_new, x1new=0, x2new=0) < 500:
                satisfied = True
                break

        if satisfied is False:
            print("Neither options for generating a new point at a specified angle"
                  " satisfy the requirement for distance to fixation!")
            return False, False

    return r1_new, r2_new


# Generates locations in cartesian co-ordinates along a circle with specified distance from origin
def gen_cart_dist_circle(distance_in_pixels):
    """

    Generates locations in cartesian co-ordinates along a circle with specified distance from origin

    :param distance_in_pixels:
    :return:
    """

    # generate random ratios
    r1 = -1 + 2 * rd.rand()
    r2 = -1 + 2 * rd.rand()

    # ratio
    ratio = (r1 ** 2) / (r2 ** 2)

    # get values normalized to the distance we want
    r2_norm = np.sqrt(distance_in_pixels ** 2 / (1 + ratio))
    r1_norm = np.sqrt(ratio * (r2_norm ** 2))

    # account for negative values
    r2_final = r2_norm * (r2 / np.abs(r2))
    r1_final = r1_norm * (r1 / np.abs(r1))

    return r1_final, r2_final


# choosing new points between the radii of two circular boundaries
def generate_ranged_points(range_in_pixels, r1_old=None, r2_old=None):
    """

    :param distance_in_pixels:
    :param r1_old:
    :param r2_old:
    :return:
    """

    # raise an error is there's a problem with values not existing
    if ((r1_old is None) and (r2_old is not None)) \
            or ((r1_old is not None) and (r2_old is None)):
        raise Exception("there is an issue with assigning location values")

    # generate starting location if old points aren't yet defined
    elif (r1_old is None) and (r2_old is None):

        r1_new = -900 + rd.rand() * 1800
        r2_new = -500 + rd.rand() * 1000

    # check distance metric - (r1_old is not None) and (r2_old is not None)
    else:
        satisfied = False

        while satisfied is False:

            # generate points a specified distance away from previous points
            r1, r2 = gen_cart_dist_circle(range_in_pixels)

            # check if points are valid
            r1_new = (r1_old + r1)
            r2_new = (r2_old + r2)

            # ensure it's not out of bounds
            # if np.abs(r1_new) < 900 and np.abs(r2_new) < 500:
            if location_dist_cart(x1=r1_new, x2=r2_new, x1new=0, x2new=0) < 500:
                satisfied = True

    return r1_new, r2_new


# Generates locations in cartesian co-ordinates between the radii of two circular boundaries
def gen_cart_dist_range(range_in_pixels):
    """

    Generates locations in cartesian co-ordinates between the radii of two circular boundaries

    :param range_in_pixels:
    :return:
    """

    # chose distance from range
    distance_in_pixels = np.random.uniform(range_in_pixels[0], range_in_pixels[1])

    # generate random ratios
    r1 = -1 + 2 * rd.rand()
    r2 = -1 + 2 * rd.rand()

    # ratio
    ratio = (r1 ** 2) / (r2 ** 2)

    # get values normalized to the distance we want
    r2_norm = np.sqrt(distance_in_pixels ** 2 / (1 + ratio))
    r1_norm = np.sqrt(ratio * (r2_norm ** 2))

    # account for negative values
    r2_final = r2_norm * (r2 / np.abs(r2))
    r1_final = r1_norm * (r1 / np.abs(r1))

    return r1_final, r2_final


# generate points randomly with a minimum distance between them
def generate_random_points(min_dist, r1_old=None, r2_old=None):
    """
    generate points randomly with a minimum distance between them

    has NOT been changed to only be around fixation

    :param min_dist:
    :param r1_old:
    :param r2_old:
    :return:
    """

    # raise an error is there's a problem with values not existing
    if ((r1_old is None) and (r2_old is not None)) \
            or ((r1_old is not None) and (r2_old is None)):
        raise Exception("there is an issue with assigning location values")

    # generate starting location if old points aren't yet defined
    if (r1_old is None) and (r2_old is None):
        r1_new = -900 + rd.rand() * 1800
        r2_new = -500 + rd.rand() * 1000

        return r1_new, r2_new

    satisfied = False

    while satisfied is False:

        # generate random point on screen
        r1 = -900 + rd.rand() * 1800
        r2 = -500 + rd.rand() * 1000

        # check if points are valid
        r1_new = (r1_old + r1)
        r2_new = (r2_old + r2)

        # make sure it's not out of bounds / 500 pixels away from the specified
        # if np.abs(r1_new) < 900 and np.abs(r2_new) < 500:
        if location_dist_cart(x1=r1_new, x2=r2_new, x1new=0, x2new=0) < 500:

            # make sure it's far away enough from the old point
            if location_dist_cart(x1=r1_new, x2=r2_new, x1new=r1_old, x2new=r2_old) > min_dist:
                satisfied = True

    return r1_new, r2_new


# generate points with random location and a minimum distance between them
def select_random_points(locations, min_dist):
    """

    returns a new location with a specified minimum distance away from all points in locations

    minimum distance is the exact distance away from one of the points

    :param locations:
    :param min_dist:
    :return:
    """

    # loop though points, generate a new point the specified difference away,
    # ensure it's far away enough from the other points
    satisfied = False
    counter = 0

    while satisfied is False:

        counter += 1

        # randomly pick a location?
        rc_location = np.random.choice(locations)

        # generate point
        r1, r2 = gen_cart_dist_circle(min_dist)

        # set new points
        new_r1 = rc_location[0] + r1
        new_r2 = rc_location[1] + r2

        # create numpy array to hold distances
        dist_array = np.zeros((len(locations),))
        boundary_array = np.zeros((len(locations),))

        # check all distances
        for i, loc in enumerate(locations):
            # distance metric rounded in case of small changes
            dist_array[i] = np.round(np.sqrt((new_r1 - loc[0]) ** 2 + (new_r2 - loc[1]) ** 2))
            boundary_array[i] = location_dist_cart(new_r1, new_r2, 0, 0)

        check_dist = dist_array >= min_dist
        check_bound = boundary_array <= 500

        if np.all(check_dist) and np.all(check_bound):
            satisfied = True

        if counter > 10000:
            warnings.warn("There probably isn't a point that satifies the absent generation condition "
                          "- try reducing minimum distance")

    return new_r1, new_r2


# ----------------------------------------------------------------------------------
#                          Randomizing parameters - colour
# __________________________________________________________________________________
# randomly choosing colour
def randomize_colour(colour_space_size, old_choice=None, chosen_colour='blue'):
    """

    chooses a random colour from white to the specified color (red, green, blue)
    ensures that the choice is not the same as the previously chosen colour

    :param colour_space_size:
    :param old_choice:
    :param chosen_colour:
    :return:
    """
    # colour on scale ranges from -1 to 1
    colour_set = np.linspace(-1, 1, colour_space_size)

    # remove the previous colour
    colour_set = colour_set[~(colour_set == old_choice)]

    # randomize colour
    choice = np.random.choice(colour_set)

    # convert to -1 to 1 RGB scale
    if chosen_colour == "blue":
        colour = [choice, choice, 1]
    elif chosen_colour == "red":
        colour = [1, choice, choice]
    elif chosen_colour == "green":
        colour = [choice, 1, choice]

    return colour, choice


# randomly choosing colour a specified distance away on the colour scale
def randomize_colour_dist(old_choice=None, colour_diff=0, chosen_colour='blue'):
    """
    chooses a random colour from white to the specified color (red, green, blue)
    ensures that the choice is not the same as the previously chosen colour

    :param colour_space_size:
    :param old_choice:
    :param colour_diff:
    :param chosen_colour:
    :return:
    """

    """
    # colour on scale ranges from -1 to 1
    colour_set = np.linspace(-1, 1, colour_space_size)

    # remove the previous colour
    colour_set = colour_set[~(colour_set == old_choice)]

    # randomize colour
    choice = np.random.choice(colour_set)
    """

    choices = [old_choice - colour_diff,
               old_choice + colour_diff]

    for i in range(2):
        if np.abs(choices[i]) > 1:
            if i == 0:
                choices = choices[1]
                break
            else:
                choices = choices[0]
                break

    if np.array([choices]).size == 0:
        raise Exception("There are no possible colours that satisfy the absent difference colour requirement")

    # randomly choose a colour
    if np.array([choices]).size > 1:
        choice = np.random.choice(choices)
    else:
        choice = choices

    # convert to -1 to 1 RGB scale
    if chosen_colour == "blue":
        colour = [choice, choice, 1]
    elif chosen_colour == "red":
        colour = [1, choice, choice]
    elif chosen_colour == "green":
        colour = [choice, 1, choice]

    return colour
