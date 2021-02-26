"""
Eli Bulger

January 2020

LiMN Lab - Carnegie Mellon University

This contains functions to help show a sequence of visual stimuli with changing characteristics

"""

import random
from numpy import random as rd
from psychopy import visual, core, monitors  # import some libraries from PsychoPy
from psychopy.tools import coordinatetools as ppct
import psychopy as pp
import numpy as np
import warnings
import math
from helpers import *


# ----------------------------------------------------------------------------------
#                         Presenting Trajectory Stimuli
# __________________________________________________________________________________

def present_signal(mywin, split_x, split_y, dt, on_off_ratio=0.7,
                   record=False, fps=2, basename='default',
                   type='orig', probeDelay=1):
    """

    timestep / refresh rate (dt)

    :param mywin:
    :param split_x:
    :param split_y:
    :param dt:
    :param on_off_ratio:  (default at .7)
    :param record:
    :param fps:
    :param basename:
    :return:
    """

    # # randomization options
    # switchRup = False
    # shuffle = False
    #
    # # option to make it totally random (not in a circle, but still trajectories)
    # if shuffle:
    #     np.random.shuffle(split_x)
    #     np.random.shuffle(split_y)
    #
    # if switchRup:
    #     split_x, split_y = switch_2_list_cols(split_x, split_y)

    # blank time
    wait_time = dt * (len(split_x[0])) * (1 - on_off_ratio)

    # loop through the stimulus
    for k, (x, y) in enumerate(zip(split_x, split_y)):

        for i in range(int(len(x) * on_off_ratio)):

            stim_pos = [x[i], y[i]]

            stim_colour = [-1, -1, 1]

            stim = pp.visual.Circle(win=mywin, size=65,
                                    pos=stim_pos,
                                    fillColor=stim_colour, lineColor=stim_colour)

            # draw the stimuli
            stim.draw()
            mywin.update()

            if record is True:
                mywin = rec_frames(mywin, dt, fps)

            """
            if i == int((len(x) * on_off_ratio)/4):
                if record is True:
                    # save movie frames before blank screen
                    mywin.saveMovieFrames(fileName=f'{basename}_Object{k}_quar.mp4', fps=fps, clearFrames=True)
            

            if i == int((len(x) * on_off_ratio)/2):
                if record is True:
                    # save movie frames before blank screen
                    mywin.saveMovieFrames(fileName=f'{basename}_Object{k}_half.mp4', fps=fps, clearFrames=True)

             
                if i == int((len(x) * on_off_ratio)/(3/4)):
                if record is True:
                    # save movie frames before blank screen
                    mywin.saveMovieFrames(fileName=f'{basename}_Object{k}_thre.mp4', fps=fps, clearFrames=True)
            """

            core.wait(dt)

        if record is True:
            # save movie frames before blank screen
            mywin.saveMovieFrames(fileName=f'{basename}_Object{k}_firs.mp4', fps=fps, clearFrames=True)

        # blank screen
        mywin.flip()
        core.wait(wait_time)

        if record is True:
            mywin = rec_frames(mywin, wait_time, fps)

        if record is True:
            # save movie frames of blank screen
            mywin.saveMovieFrames(fileName=f'{basename}_Object{k}_wait.mp4', fps=fps, clearFrames=True)

    if type == 'orig':
        fixation(time=probeDelay,
                 win=mywin,
                 record=record,
                 fps=fps)

        if record is True:
            # save movie frames of blank screen
            mywin.saveMovieFrames(fileName=f'{basename}_Object{k}_wait_fixation.mp4', fps=fps, clearFrames=True)

    return


def fixation(time, win, record=False, fps=2):
    """

    show fixation cross, optionally record frames of it

    :param time:
    :param win:
    :param record:
    :param fps:
    :return:
    """

    # show cross for 1 second
    message = visual.TextStim(win, text='+', height=85, font='Courier New',
                              bold=False, color=[-1, -1, -1])
    message.draw()
    win.flip()

    if record is True:
        rec_frames(win, time, fps)

    core.wait(time)


# ----------------------------------------------------------------------------------
#                                Presenting Stimuli
# __________________________________________________________________________________
def present_AP(seq_tags, mywin, tstim=1, tblank=0, record=False, fps=2):
    """

    Draws stimuli on screen according to characteristics defining them

    Saves a movie too

    :param seq_tags:
    :param tstim:
    :param tblank:
    :param mywin:
    :return:
    """

    # draw and update
    stim_pos = seq_tags[0]
    stim_colour = seq_tags[1]

    stim = pp.visual.Circle(win=mywin, size=75,
                            pos=stim_pos,
                            fillColor=stim_colour, lineColor=stim_colour)

    # draw the stimuli
    stim.draw()
    mywin.update()

    if record is True:
        rec_frames(mywin, tstim, fps)

    core.wait(tstim)

    # blank screen
    mywin.flip()
    core.wait(tblank)

    return


def present_sequence(seq_tags, mywin, tstim=1, tblank=0, record=False, fps=2):
    """
    Draws stimuli on screen according to characteristics defining them

    Optionally records the sequence

    :param seq_tags:
    :param mywin:
    :param tstim:
    :param tblank:
    :param record:
    :param fps:
    :return:
    """

    # draw and update
    for i in range(len(seq_tags[0, :])):

        stim_pos = seq_tags[0, i]
        stim_colour = seq_tags[1, i]

        stim = pp.visual.Circle(win=mywin, size=75,
                                pos=stim_pos,
                                fillColor=stim_colour, lineColor=stim_colour)

        # draw the stimuli
        stim.draw()
        mywin.update()

        if record is True:
            rec_frames(mywin, tstim, fps)

        core.wait(tstim)

        # blank screen
        mywin.flip()
        core.wait(tblank)

    return

