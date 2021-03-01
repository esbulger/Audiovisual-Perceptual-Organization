import numpy as np
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from psychopy import visual, core, monitors  # import some libraries from PsychoPy
from psychopy.tools import coordinatetools as ppct
from generate_stims import *
import os

# ----------------------------------------------------------------------------------
#                           Helpers / Misc.
# __________________________________________________________________________________

def convert_polar2cart(theta, radius):
    """

    :param theta: the degrees array
    :param radius: the signal array
    :return:
    """

    x, y = ppct.pol2cart(theta, radius, units='deg')

    return x, y


def split_array(array, nStims):
    """

    Splits a 1-D array into multiple pieces

    :param array:
    :param nStims:
    :return:
    """

    import numpy as np

    # warn if mismatched
    if (len(array) % 2) is False:
        import warnings
        warnings.warn("The signal array is not evenly divisible by nStims")

    # now we have one array per stimulus
    split_array = np.array_split(array, nStims)

    return split_array


def process_audio_on_off(rand_signal, stimsPerSequence, on_off_ratio, split=True):
    """

    optionally splits the array into number of stimuli
    sets intensities to zero for off portions
    recombines signal to be 1 array

    :param rand_signal:
    :param stimsPerSequence:
    :param on_off_ratio:
    :param split: if True, the signal is split up into stimsPerSequence chunks
    :return:
    """

    if split is True:
        chopped_signal = split_array(rand_signal, stimsPerSequence)
    else:
        chopped_signal = rand_signal

    processed_signal = np.array([])

    for element in chopped_signal:

        # set intensities to zero for off portions
        element[int(len(element) * on_off_ratio):] = 0

        processed_signal = np.concatenate([processed_signal, element])

    return processed_signal


def process_visual_on_off(rand_signal, stimsPerSequence, on_off_ratio, cycles, split=True):
    """

    optionally splits the array into number of stimuli
    sets intensities to np.nan for off portions

    :param rand_signal:
    :param stimsPerSequence:
    :param on_off_ratio:
    :param cycles:
    :param split:  if True, the signal is split up into stimsPerSequence chunks
    :return:
    """

    from generate_stims import generate_degrees

    if split is True:

        theta = generate_degrees(rand_signal, cycles=cycles)

        # convert to x, y coordinates
        x, y = convert_polar2cart(theta, rand_signal)

        # split into sequence elements
        split_x = split_array(x, stimsPerSequence)
        split_y = split_array(y, stimsPerSequence)

        # convert the off components to be np.nan

        for k, (x, y) in enumerate(zip(split_x, split_y)):
            x[int(len(x) * on_off_ratio):] = np.nan
            y[int(len(y) * on_off_ratio):] = np.nan

    else:
        # already split, so just go through each element and set the split_x and split_y elements
        split_x = np.zeros((stimsPerSequence,), dtype=object)
        split_y = np.zeros((stimsPerSequence,), dtype=object)

        for k, element in enumerate(rand_signal):

            theta = generate_degrees(element,
                                     cycles=cycles,
                                     parts=True,
                                     stimsPerSequence=stimsPerSequence,
                                     stimNum=k)

            # convert to x, y coordinates
            split_x[k], split_y[k] = convert_polar2cart(theta, element)

            # for each element, set off ratios to np.nan
            split_x[k][int(len(split_x[k]) * on_off_ratio):] = np.nan
            split_y[k][int(len(split_y[k]) * on_off_ratio):] = np.nan

    return split_x, split_y


def switch_2cols(array, seed=666666):
    """

    :param array:
    :return:
    """

    new_matrix = array.copy()

    nCols = len(array)
    colNums = np.arange(nCols - 1)

    if seed is None:
        # set a seed randomly if one is not specified
        seed = rd.randint(100000000)

    # set the seed
    np.random.seed(seed)

    # choose column to switch
    col1 = np.random.choice(colNums)
    col2 = col1 + 1

    """
    # ex. options if 5 positions
    [0, 1] -> [1, 0]
    [1, 2] -> [2, 1]
    [2, 3] -> [3, 2]
    [3, 4] -> [4, 3]
    """

    # reassign matrix columns
    # new_matrix[[col1, col2]] = new_matrix[[col2, col1]]

    new_matrix[col1], new_matrix[col2] = new_matrix[col2], new_matrix[col1]

    return new_matrix, seed


def pause():
    input("Press the <ENTER> key to continue...")


def rec_frames(win, time, fps):
    """

    :param time:
    :param fps:
    :return:
    """

    for i in range(int(time * fps)):
        win.getMovieFrame(buffer='front')

    return win


def location_dist_cart(x1, x2, x1new, x2new):
    """

    returns distance between two points in 2D

    :param x1:
    :param x2:
    :param x1new:
    :param x2new:
    :return:
    """

    return np.sqrt((np.square(x1 - x1new) + np.square(x2 - x2new)))


def switch_2_list_cols(list1, list2):
    """

    new list 1

    :param list1:
    :param list2:
    :return:
    """
    new_list1 = list1.copy()
    new_list2 = list2.copy()


    nCols = len(list1)
    colNums = np.arange(nCols - 1)

    # choose column to switch
    col1 = np.random.choice(colNums)
    col2 = col1 + 1

    """
    # ex. options if 5 positions
    [0, 1] -> [1, 0]
    [1, 2] -> [2, 1]
    [2, 3] -> [3, 2]
    [3, 4] -> [4, 3]
    """

    # reanew_list1ssign matrix columns
    new_list1[col1], new_list1[col2] = new_list1[col2], new_list1[col1]
    new_list2[col1], new_list2[col2] = new_list2[col2], new_list2[col1]

    return new_list1, new_list2


def concat_clips(filepath, basename, prename, final, fps, delete=True):
    """

    :param filepath:
    :param basename:
    :param prename: the appended name to the basename
    :param final: if true, renames with final in front
    :param fps:
    :param delete:
    :return:
    """
    # from moviepy.editor import *
    import os
    from natsort import natsorted

    L = []

    whatchuWANT = len(prename + basename)

    for root, dirs, files in os.walk(filepath):

        # files.sort()
        files = natsorted(files)
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                if os.path.splitext(file)[0][0:whatchuWANT] == prename + basename:
                    filePath = os.path.join(root, file)
                    video = VideoFileClip(filePath)
                    L.append(video)

    final_clip = concatenate_videoclips(L)
    final_clip.to_videofile("all_" + prename + basename + ".mp4", fps=fps, remove_temp=False)

    if delete is True:
        for root, dirs, files in os.walk(filepath):

            # files.sort()
            files = natsorted(files)
            for file in files:
                if os.path.splitext(file)[1] == '.mp4':
                    if os.path.splitext(file)[0][0:whatchuWANT] == prename + basename:
                        filePath = os.path.join(root, file)
                        os.remove(filePath)

    if final is True:
        os.rename("all_" + prename + basename + ".mp4",
                  "final_" + basename + ".mp4")

    return

def save_audio_mp3(audio_basename, origsavepath, delaysavepath, probesavepath):
    """

    :param audio_basename:
    :param origsavepath:
    :param delaysavepath:
    :param probesavepath:
    :return:
    """

    cwd = os.getcwd()

    # combine wav files together
    from pydub import AudioSegment
    AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"

    sound_orig = AudioSegment.from_file(origsavepath)
    sound_delay = AudioSegment.from_file(delaysavepath)
    sound_probe = AudioSegment.from_file(probesavepath)

    orig_delay = sound_orig.append(sound_delay)
    orig_delay_probe = orig_delay.append(sound_probe)

    file_wav = os.path.dirname(__file__) + '/final_' + audio_basename + '.wav'
    file_mp3 = file_wav[:-3] + 'mp3'

    # orig_delay_probe.export(file_wav, format='wav')
    orig_delay_probe.export(file_mp3, format='mp3')

    delete = True

    if delete is True:
        from natsort import natsorted

        for root, dirs, files in os.walk(cwd):
            # files.sort()
            files = natsorted(files)
            for file in files:
                if os.path.splitext(file)[1] == '.wav':
                    if os.path.splitext(file)[0][0:len(audio_basename)] == audio_basename:
                        filePath = os.path.join(root, file)
                        os.remove(filePath)


def rescale_signal(signal, minmaxRange, split=False):
    """

    :param signal:
    :param minmaxRange:
    :param split:
    :return:
    """


    # rescale x matrix
    current_max = np.max(signal)
    current_min = np.min(signal)
    new_min = minmaxRange[0]
    new_max = minmaxRange[1]

    # normalize to pixel limits specified
    x_t = (new_max - new_min) / (current_max - current_min) * (signal - current_max) + new_max

    return x_t

def downsample(array, factor):
    """

    downsample an array by a factor specified

    :param array:
    :param factor:
    :return:
    """

    arr_len = len(array)

    arr_inds = np.arange(0, arr_len)

    # mark array - only integer quotients will be extracted
    arr_inds_adj = np.divide(arr_inds, factor)

    # downsample
    ds_arr = array[np.where(np.round(arr_inds_adj) == arr_inds_adj)]

    return ds_arr


def recombine_array(array):
    """
    recombines an array consisting of any number of sub-arrays
    useful for combining a numpy array with dtype=object, holding sub arrays

    :param array:
    :return:
    """

    # empty array
    recombined_array = np.array([])

    # recombine
    for sub_arr in array:

        recombined_array = np.concatenate([recombined_array, sub_arr])

    return recombined_array

def combine_audio(vidname, audname, outname, fps=60):
    """
    from:
    https://stackoverflow.com/questions/63881088/how-to-merge-mp3-
    and-mp4-in-python-mute-a-mp4-file-and-add-a-mp3-file-on-it

    :param vidname:
    :param audname:
    :param outname:
    :param fps:
    :return:
    """
    import moviepy.editor as mpe
    my_clip = mpe.VideoFileClip(vidname)
    audio_background = mpe.AudioFileClip(audname)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(outname, fps=fps)

    return





















