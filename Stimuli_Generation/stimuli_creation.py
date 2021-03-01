import numpy as np
from generate_stims import *
from present_stims import *
from plot_stims import *
from helpers import *
from tdklatt import *
import psychopy as pp
from psychopy import visual, core, event, monitors  # import some libraries from PsychoPy
from psychopy.tools import coordinatetools as ppct
from win32api import GetSystemMetrics
import time
import os
import pandas as pd

event.globalKeys.add(key='escape', func=core.quit)


def generate_stimuli(param_dict, trial_type, all_stim_data):
    """

    :param param_dict:
    :param trial_type:
    :param all_stim_data:
    :return:
    """

    freq_lims = [np.round(param_dict.center_f - param_dict.rad_f, 3),
                 np.round(param_dict.center_f + param_dict.rad_f, 3)]

    # initialise norm parameters to be None
    norm_diffs = None
    der_diffs = None
    norm_diffs_v = None
    norm_diffs_v = None

    # make original audio object
    kl_obj = KlattParam1980(FS=10000, N_FORM=5, DUR=5.0, F0=100,
                            FF=[400, 1500, 2500, 3250, 3700],
                            BW=[60, 90, 150, 200, 200],
                            AV=60, AVS=0, AH=0, AF=0,
                            SW=0, FGP=0, BGP=100, FGZ=1500, BGZ=6000,
                            FNP=250, BNP=100, FNZ=250, BNZ=100, BGS=200)

    s = klatt_make(kl_obj)

    # ----------------------------------------------------------------------------------
    #             Generating Random Signal for Both Audio and Visual Stimuli
    # __________________________________________________________________________________

    # generate random signal using audio F0 frequency
    rand_signal_audio, comp, time_arr, freq, seed = generate_signal(T=param_dict.T,
                                                                    dt=param_dict.dt,
                                                                    minmaxRange=param_dict.F0_limit,
                                                                    freq_lims=freq_lims)

    # using same seed, but
    rand_signal_visual, comp, time_arr, freq, seed = generate_signal(T=param_dict.T,
                                                                     dt=param_dict.dt,
                                                                     minmaxRange=param_dict.pixel_limit,
                                                                     freq_lims=freq_lims,
                                                                     seed=seed)

    # re-sample the visual signal so we can present it without problems
    ds_factor = 400
    rand_signal_visual = downsample(rand_signal_visual, ds_factor)

    plot_signal(time=time_arr,
                rSignal=rand_signal_audio,
                limit=freq_lims[0],
                limit_high=freq_lims[1],
                name=f'x(t) at {freq_lims[0]}-{freq_lims[1]} Hz - auditory')

    plot_signal(time=downsample(time_arr, ds_factor),
                rSignal=rand_signal_visual,
                limit=freq_lims[0],
                limit_high=freq_lims[1],
                name=f'x(t) at {freq_lims[0]}-{freq_lims[1]} Hz - visual')

    # plot the derivative of the signal
    # der_signal = np.gradient(rand_signal)
    # plot_signal(time=time_arr, rSignal=der_signal, limit=F0_limit[0], limit_high=F0_limit[1])

    # ----------------------------------------------------------------------------------
    #                           Processing signals + generating probe
    # __________________________________________________________________________________
    # split the signal into different pieces
    split_orig_signal_audio = split_array(array=rand_signal_audio,
                                          nStims=param_dict.stimsPerSequence)

    split_orig_signal_visual = split_array(array=rand_signal_visual,
                                           nStims=param_dict.stimsPerSequence)

    # generate what to play based on - stimulus type - this step is valid for both audio and visual
    if trial_type == 'same':

        # keep the signal the same
        split_probe_signal_audio = split_orig_signal_audio
        split_probe_signal_visual = split_orig_signal_visual

        probe_stims = param_dict.stimsPerSequence

        probe_obj = KlattParam1980(FS=10000, N_FORM=5, DUR=5.0, F0=100,
                                   FF=[400, 1500, 2500, 3250, 3700],
                                   BW=[60, 90, 150, 200, 200],
                                   AV=60, AVS=0, AH=0, AF=0,
                                   SW=0, FGP=0, BGP=100, FGZ=1500, BGZ=6000,
                                   FNP=250, BNP=100, FNZ=250, BNZ=100, BGS=200)

        s_probe = klatt_make(probe_obj)

    elif trial_type == 'diff':

        # switch 2 columns only
        split_probe_signal_audio, seed = switch_2cols(split_orig_signal_audio, seed=seed)
        split_probe_signal_visual, seed = switch_2cols(split_orig_signal_visual, seed=seed)

        probe_stims = param_dict.stimsPerSequence

        probe_obj = KlattParam1980(FS=10000, N_FORM=5, DUR=5.0, F0=100,
                                   FF=[400, 1500, 2500, 3250, 3700],
                                   BW=[60, 90, 150, 200, 200],
                                   AV=60, AVS=0, AH=0, AF=0,
                                   SW=0, FGP=0, BGP=100, FGZ=1500, BGZ=6000,
                                   FNP=250, BNP=100, FNZ=250, BNZ=100, BGS=200)

        s_probe = klatt_make(probe_obj)

    elif trial_type == 'pres':

        # choose one element randomly
        np.random.seed(seed)
        rand_int = np.random.randint(0, param_dict.stimsPerSequence - 1)

        split_probe_signal_audio = [split_orig_signal_audio[rand_int]]
        split_probe_signal_visual = [split_orig_signal_visual[rand_int]]

        probe_stims = 1

        probe_obj = KlattParam1980(FS=10000, N_FORM=5, DUR=param_dict.T / param_dict.stimsPerSequence, F0=100,
                                   FF=[400, 1500, 2500, 3250, 3700],
                                   BW=[60, 90, 150, 200, 200],
                                   AV=60, AVS=0, AH=0, AF=0,
                                   SW=0, FGP=0, BGP=100, FGZ=1500, BGZ=6000,
                                   FNP=250, BNP=100, FNZ=250, BNZ=100, BGS=200)

        s_probe = klatt_make(probe_obj)

    elif trial_type == 'abst':

        # sketchiest part of the whole process here - but also cool
        split_probe_signal_audio, norm_diffs, der_diffs = generate_absent_trajectory(split_orig_signal_audio,
                                                                                     seed)
        split_probe_signal_visual, norm_diffs_v, der_diffs_v = generate_absent_trajectory(split_orig_signal_visual,
                                                                                          seed)

        probe_stims = 1

        probe_obj = KlattParam1980(FS=10000, N_FORM=5, DUR=param_dict.T / param_dict.stimsPerSequence, F0=100,
                                   FF=[400, 1500, 2500, 3250, 3700],
                                   BW=[60, 90, 150, 200, 200],
                                   AV=60, AVS=0, AH=0, AF=0,
                                   SW=0, FGP=0, BGP=100, FGZ=1500, BGZ=6000,
                                   FNP=250, BNP=100, FNZ=250, BNZ=100, BGS=200)

        s_probe = klatt_make(probe_obj)

    else:
        raise Exception('Trial type inputted into function is not valid')

    # process the signal for audio playback - split up into different pieces
    processed_orig_signal_audio = process_audio_on_off(rand_signal=split_orig_signal_audio,
                                                       stimsPerSequence=param_dict.stimsPerSequence,
                                                       on_off_ratio=param_dict.on_off_ratio,
                                                       split=False)

    processed_probe_signal_audio = process_audio_on_off(rand_signal=split_probe_signal_audio,
                                                        stimsPerSequence=probe_stims,
                                                        on_off_ratio=param_dict.on_off_ratio,
                                                        split=False)

    # process the signal for visual playback - split up into different pieces
    processed_orig_signal_x, \
    processed_orig_signal_y = process_visual_on_off(rand_signal=split_orig_signal_visual,
                                                    stimsPerSequence=param_dict.stimsPerSequence,
                                                    on_off_ratio=param_dict.on_off_ratio,
                                                    cycles=param_dict.cycles,
                                                    split=False)

    processed_probe_signal_x, \
    processed_probe_signal_y = process_visual_on_off(rand_signal=split_probe_signal_visual,
                                                     stimsPerSequence=probe_stims,
                                                     on_off_ratio=param_dict.on_off_ratio,
                                                     cycles=param_dict.cycles,
                                                     split=False)

    # split before plot
    plot_signal(time=time_arr,
                rSignal=processed_orig_signal_audio,
                limit=freq_lims[0],
                limit_high=freq_lims[1],
                name=f'x(t) at {freq_lims[0]}-{freq_lims[1]} Hz - audio original')

    if param_dict.trial_type == 'abst' or param_dict.trial_type == 'pres':
        time_arr1 = time_arr[0:int(len(time_arr) / param_dict.stimsPerSequence)]
    else:
        time_arr1 = time_arr

    # plot what it looks like now
    plot_signal(time=time_arr1,
                rSignal=processed_probe_signal_audio,
                limit=freq_lims[0],
                limit_high=freq_lims[1],
                name=f'x(t) at {freq_lims[0]}-{freq_lims[1]} Hz - audio probe')

    if False is True:
        # plot the probe
        probe2plot = recombine_array(split_probe_signal_visual)
        time_arr2 = downsample(time_arr, ds_factor)[0:len(probe2plot)]

        plot_signal(time=time_arr2,
                    rSignal=probe2plot,
                    limit=freq_lims[0],
                    limit_high=freq_lims[1],
                    name=f'x(t) at {freq_lims[0]}-{freq_lims[1]} Hz - visual probe')

    # ----------------------------------------------------------------------------------
    #                               Present the visual signal
    # __________________________________________________________________________________
    fps = int(1 / (param_dict.dt * ds_factor))

    mon = monitors.Monitor('testMonitor')

    # create a window
    mywin = visual.Window([GetSystemMetrics(0), GetSystemMetrics(1)], monitor=mon, units="pix", fullscr=True,
                          allowGUI=True, screen=1, color=[1, 1, 1])

    visual_basename = f"Visual_" \
                      f"Range{param_dict.pixel_limit[0]}-{param_dict.pixel_limit[1]}_" \
                      f"Freq{freq_lims[0]}-{freq_lims[1]}_" \
                      f"Seed-{seed}_" \
                      f"Stims-{param_dict.stimsPerSequence}_" \
                      f"Type-{param_dict.trial_type}"

    visual_basename_orig = visual_basename + 'orig'
    visual_basename_prob = visual_basename + 'prob'

    # present both the original stimulus and the probe stimulus
    present_signal(mywin=mywin,
                   split_x=processed_orig_signal_x,
                   split_y=processed_orig_signal_y,
                   dt=param_dict.dt * ds_factor,
                   on_off_ratio=param_dict.on_off_ratio,
                   record=param_dict.record,
                   fps=fps,
                   basename=visual_basename_orig,
                   type='orig',
                   probeDelay=param_dict.probeDelay)

    present_signal(mywin=mywin,
                   split_x=processed_probe_signal_x,
                   split_y=processed_probe_signal_y,
                   dt=param_dict.dt * ds_factor,
                   on_off_ratio=param_dict.on_off_ratio,
                   record=param_dict.record,
                   fps=fps,
                   basename=visual_basename_prob,
                   type='prob')

    mywin.close()

    # ----------------------------------------------------------------------------------
    #                               Save the visual signal
    # __________________________________________________________________________________
    # try doing this separately!
    if param_dict.record is True:
        cwd = os.getcwd()

        concat_clips(filepath=cwd, basename=visual_basename_orig, prename='', final=False, fps=fps)
        concat_clips(filepath=cwd, basename=visual_basename_prob, prename='', final=False, fps=fps)

        concat_clips(filepath=cwd, basename=visual_basename, prename='all_', final=True, fps=fps)

    # ----------------------------------------------------------------------------------
    #                               Present the auditory signal
    # __________________________________________________________________________________
    audio_basename = f"Audio_" \
                     f"Range{param_dict.F0_limit[0]}-{param_dict.F0_limit[1]}_" \
                     f"Freq{freq_lims[0]}-{freq_lims[1]}_" \
                     f"Seed-{seed}_" \
                     f"Stims-{param_dict.stimsPerSequence}_" \
                     f"Type-{param_dict.trial_type}"

    # assign processed stimulus to the Klaat object
    s.params["F0"] = processed_orig_signal_audio
    s.run()  # Rerun synthesizer
    # s.play()  # Play the trajectory in auditory domain

    # time.sleep(param_dict.T)

    s_probe.params["F0"] = processed_probe_signal_audio
    s_probe.run()  # Rerun synthesizer
    # s_probe.play()  # Play the trajectory in auditory domain

    # ----------------------------------------------------------------------------------
    #                               Save the auditory signal
    # __________________________________________________________________________________

    if param_dict.record is True:
        origsavepath = os.path.dirname(__file__) + '/' + audio_basename + "_orig.wav"
        s.save(origsavepath)
        probesavepath = os.path.dirname(__file__) + '/' + audio_basename + "_prob.wav"
        s_probe.save(probesavepath)

        # add one second of quiet - kinda sketchy way to do it, oh well
        delay_obj = KlattParam1980(FS=10000, N_FORM=5, DUR=param_dict.probeDelay, F0=100,
                                   FF=[400, 1500, 2500, 3250, 3700],
                                   BW=[60, 90, 150, 200, 200],
                                   AV=60, AVS=0, AH=0, AF=0,
                                   SW=0, FGP=0, BGP=100, FGZ=1500, BGZ=6000,
                                   FNP=250, BNP=100, FNZ=250, BNZ=100, BGS=200)

        s_delay = klatt_make(delay_obj)
        s_delay.params["F0"] = np.zeros((s_delay.params["N_SAMP"]), )
        delaysavepath = os.path.dirname(__file__) + '/' + audio_basename + "_delay.wav"
        s_delay.save(delaysavepath)

        # save the audio by appending each to the other
        save_audio_mp3(audio_basename=audio_basename,
                       origsavepath=origsavepath,
                       delaysavepath=delaysavepath,
                       probesavepath=probesavepath)

        # save combined stimuli! super exciting
        combined_basename = f"/final_AV_" \
                            f"Range{param_dict.pixel_limit[0]}-{param_dict.pixel_limit[1]}_" \
                            f"Freq{freq_lims[0]}-{freq_lims[1]}_" \
                            f"Seed-{seed}_" \
                            f"Stims-{param_dict.stimsPerSequence}_" \
                            f"Type-{param_dict.trial_type}"

        combine_audio(vidname=os.path.dirname(__file__) + '/final_' + visual_basename + '.mp4',
                      audname=os.path.dirname(__file__) + '/final_' + audio_basename + '.mp3',
                      outname=os.path.dirname(__file__) + combined_basename + '.mp4',
                      fps=fps)

    # Store all important data in a dataframe

    stim_df = pd.DataFrame({'Type': param_dict.trial_type,
                            'Stims_Per_Sequence': param_dict.stimsPerSequence,
                            'Seed': seed,
                            'F0_low': param_dict.F0_limit[0],
                            'F0_high': param_dict.F0_limit[1],
                            'Pixel_low': param_dict.pixel_limit[0],
                            'Pixel_high': param_dict.pixel_limit[1],
                            'Center_F': param_dict.center_f,
                            'Rad_F': param_dict.rad_f,
                            'On_Off_Ratio': np.round(param_dict.on_off_ratio, 3),
                            'Visual_Cycles': param_dict.cycles,
                            'Probe_Delay': param_dict.probeDelay,
                            'Absent_Method': 'invert',
                            'Absent_Norm_Diff': norm_diffs,
                            'Absent_Der_Diff': der_diffs}, index=[0])

    all_stim_data = pd.concat([all_stim_data, stim_df], ignore_index=True)

    return all_stim_data


class ParameterClass:
    def __init__(self, stimsPerSequence, on_off_ratio, cycles, T, dt, probeDelay, record,
                 center_f, rad_f, F0_limit, pixel_limit, trial_type):
        self.trial_type = trial_type
        self.pixel_limit = pixel_limit
        self.F0_limit = F0_limit
        self.rad_f = rad_f
        self.center_f = center_f
        self.record = record
        self.probeDelay = probeDelay
        self.dt = dt
        self.T = T
        self.cycles = cycles
        self.on_off_ratio = on_off_ratio
        self.stimsPerSequence = stimsPerSequence


import os

param_dict = ParameterClass(stimsPerSequence=5,
                            on_off_ratio=2 / 3,
                            cycles=2,
                            T=5,
                            dt=0.0001,
                            probeDelay=1,
                            record=True,
                            center_f=0.3,
                            rad_f=0.1,
                            F0_limit=np.array([25, 250]),
                            pixel_limit=np.array([100, 400]),
                            trial_type='pres')

stimuli_data = pd.DataFrame({'Type': [None],
                             'Stims_Per_Sequence': [None],
                             'Seed': [None],
                             'F0_low': [None],
                             'F0_high': [None],
                             'Pixel_low': [None],
                             'Pixel_high': [None],
                             'Center_F': [None],
                             'Rad_F': [None],
                             'On_Off_Ratio': [None],
                             'Visual_Cycles': [None],
                             'Probe_Delay': [None],
                             'Absent_Method': [None],
                             'Absent_Norm_Diff': [None],
                             'Absent_Der_Diff': [None]})

# import the garbage collector
import gc


stim_type = ['same', 'diff']

for s_type in stim_type:
    param_dict.trial_type = s_type

    # for i in range(5):

    stimuli_data = generate_stimuli(param_dict=param_dict,
                                    trial_type=param_dict.trial_type,
                                    all_stim_data=stimuli_data)

data_save_path = os.path.dirname(os.path.dirname(__file__)) + '/stimuli_data.xlsx'

stimuli_data.to_excel(data_save_path, index=False)
