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

event.globalKeys.add(key='escape', func=core.quit)


def generate_stimuli(param_dict, trial_type):
    """

    :param param_dict:
    :param trial_type:
    :return:
    """

    freq_lims = [param_dict.center_f - param_dict.rad_f,
                 param_dict.center_f + param_dict.rad_f]

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

        probe_obj = KlattParam1980(FS=10000, N_FORM=5, DUR=5.0, F0=100,
                                   FF=[400, 1500, 2500, 3250, 3700],
                                   BW=[60, 90, 150, 200, 200],
                                   AV=60, AVS=0, AH=0, AF=0,
                                   SW=0, FGP=0, BGP=100, FGZ=1500, BGZ=6000,
                                   FNP=250, BNP=100, FNZ=250, BNZ=100, BGS=200)

        s_probe = klatt_make(probe_obj)

    elif trial_type == 'pres':

        # choose one element randomly
        split_probe_signal_audio = np.random.choice(split_orig_signal_audio)
        split_probe_signal_visual = np.random.choice(split_orig_signal_visual)

        probe_obj = KlattParam1980(FS=10000, N_FORM=5, DUR=param_dict.T / param_dict.stimsPerSequence, F0=100,
                                   FF=[400, 1500, 2500, 3250, 3700],
                                   BW=[60, 90, 150, 200, 200],
                                   AV=60, AVS=0, AH=0, AF=0,
                                   SW=0, FGP=0, BGP=100, FGZ=1500, BGZ=6000,
                                   FNP=250, BNP=100, FNZ=250, BNZ=100, BGS=200)

        s_probe = klatt_make(probe_obj)

    elif trial_type == 'abst':

        # sketchiest part of the whole process here - but also cool
        split_probe_signal_audio, norm_diffs, der_diffs = generate_absent_trajectory(split_orig_signal_audio)
        split_probe_signal_visual, norm_diffs_v, der_diffs_v = generate_absent_trajectory(split_orig_signal_visual)

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
                                                        stimsPerSequence=param_dict.stimsPerSequence,
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
                                                     stimsPerSequence=param_dict.stimsPerSequence,
                                                     on_off_ratio=param_dict.on_off_ratio,
                                                     cycles=param_dict.cycles,
                                                     split=False)

    # split before plot
    plot_signal(time=time_arr,
                rSignal=processed_orig_signal_audio,
                limit=freq_lims[0],
                limit_high=freq_lims[1],
                name=f'x(t) at {freq_lims[0]}-{freq_lims[1]} Hz - audio original')

    # plot what it looks like now
    plot_signal(time=time_arr,
                rSignal=processed_probe_signal_audio,
                limit=freq_lims[0],
                limit_high=freq_lims[1],
                name=f'x(t) at {freq_lims[0]}-{freq_lims[1]} Hz - audio probe')

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
    fps = int(1/(param_dict.dt * ds_factor))

    mon = monitors.Monitor('testMonitor')

    # create a window
    mywin = visual.Window([GetSystemMetrics(0), GetSystemMetrics(1)], monitor=mon, units="pix", fullscr=True,
                          allowGUI=True, screen=1, color=[1, 1, 1])

    visual_basename = f"Visual_" \
                      f"Range{param_dict.pixel_limit[0]}-{param_dict.pixel_limit[1]}_" \
                      f"Freq{freq_lims[0]}-{freq_lims[1]}_" \
                      f"Seed-{seed}_" \
                      f"Stims-{param_dict.stimsPerSequence}_" \
                      f"Type-{param_dict.trial_type}_"

    visual_basename_orig = visual_basename + 'orig'
    visual_basename_prob = visual_basename + 'prob'
    all_basename = "all_" + visual_basename

    # present both the original stimulus and the probe stimulus
    present_signal(mywin=mywin,
                   split_x=processed_orig_signal_x,
                   split_y=processed_orig_signal_y,
                   dt=param_dict.dt*ds_factor,
                   on_off_ratio=param_dict.on_off_ratio,
                   record=param_dict.record,
                   fps=fps,
                   basename=visual_basename_orig,
                   type='orig',
                   probeDelay=param_dict.probeDelay)

    present_signal(mywin=mywin,
                   split_x=processed_probe_signal_x,
                   split_y=processed_probe_signal_y,
                   dt=param_dict.dt*ds_factor,
                   on_off_ratio=param_dict.on_off_ratio,
                   record=param_dict.record,
                   fps=fps,
                   basename=visual_basename_prob,
                   type='prob')

    # try doing this separately!
    if param_dict.record is True:
        cwd = os.getcwd()

        concat_clips(filepath=cwd, basename=visual_basename_orig, fps=fps)
        concat_clips(filepath=cwd, basename=visual_basename_prob, fps=fps)

        # concat_clips(filepath=cwd, basename=all_basename, fps=fps)

    mywin.close()

    # ----------------------------------------------------------------------------------
    #                               Present the auditory signal
    # __________________________________________________________________________________
    audio_basename = f"Audio_" \
                     f"Range{param_dict.F0_limit[0]}-{param_dict.F0_limit[1]}_" \
                     f"Freq{freq_lims[0]}-{freq_lims[1]}_" \
                     f"Seed-{seed}_" \
                     f"Stims-{param_dict.stimsPerSequence}_" \
                     f"Type-{param_dict.trial_type}_"

    # assign processed stimulus to the Klaat object
    s.params["F0"] = processed_orig_signal_audio
    s.run()  # Rerun synthesizer
    s.play()  # Play the trajectory in auditory domain


    # time.sleep(param_dict.T)

    s_probe.params["F0"] = processed_probe_signal_audio
    s_probe.run()  # Rerun synthesizer
    # s.play()  # Play the trajectory in auditory domain

    # option to save
    if param_dict.record is True:
        savepath = os.path.dirname(__file__) + '/' + audio_basename + "_orig.wav"
        s.save(savepath)
        probesavepath = os.path.dirname(__file__) + '/' + audio_basename + "_prob.wav"
        s_probe.save(probesavepath)

    # time.sleep(param_dict.T)

    return


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


param_dict = ParameterClass(stimsPerSequence=5,
                            on_off_ratio=2/3,
                            cycles=2,
                            T=5,
                            dt=0.0001,
                            probeDelay=1,
                            record=True,
                            center_f=0.3,
                            rad_f=0.1,
                            F0_limit=np.array([25, 250]),
                            pixel_limit=np.array([100, 400]),
                            trial_type='diff')

generate_stimuli(param_dict=param_dict,
                 trial_type='diff')
