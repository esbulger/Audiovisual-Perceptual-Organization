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

# ----------------------------------------------------------------------------------
#                                       Parameters
# __________________________________________________________________________________

# visual
nBlocks = 1
nSequences = 2
stimsPerSequence = 5
tstim = 0.25

# visual, trial level
cycles = 2

# audiovisual
on_off_ratio = 2 / 3

# audio
T = 5.0

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

# center controls how fast the signal changes, and rad controls the scale of the variability
scale = 1  # necessary t control size of array
center_f = 1
rad_f = 0.01
dt = 0.0001  # must be 10,000 for the Klatt synthesizer to work (I don't know why >:( )
F0_limit = np.array([25, 250])

freq_lims = [center_f - rad_f,
              center_f + rad_f]

# generate random signal
rand_signal, comp, time_arr, freq, seed = generate_signal(T=T, dt=dt, minmaxRange=F0_limit, freq_lims=freq_lims)

plot_signal(time_arr, rSignal=rand_signal, limit=freq_lims[0], limit_high=freq_lims[1])

# plot the derivative of the signal
# der_signal = np.gradient(rand_signal)
# plot_signal(time_arr, rSignal=der_signal, limit=F0_limit[0], limit_high=F0_limit[1])


# ----------------------------------------------------------------------------------
#                           Processing signals
# __________________________________________________________________________________
# split the signal
split_rand_signal = split_array(rand_signal, stimsPerSequence)

# process the signal for audio playback - split up into different pieces
processed_signal = process_audio_on_off(rand_signal, stimsPerSequence, on_off_ratio)

# plot what it looks like now
plot_signal(time_arr, rSignal=processed_signal, limit=freq_lims[0], limit_high=freq_lims[1])

# visual stimuli
split_x, split_y = process_visual_on_off(rand_signal, stimsPerSequence, on_off_ratio, cycles)

# ----------------------------------------------------------------------------------
#                               Present the signals
# __________________________________________________________________________________
mon = monitors.Monitor('testMonitor')

# create a window
mywin = visual.Window([GetSystemMetrics(0), GetSystemMetrics(1)], monitor=mon, units="pix", fullscr=True,
                      allowGUI=True, screen=1, color=[0.5, 0.5, 0.5])

endname = f'freq_{t_div_freq}'
basename = f"Trajectory_Task-Pix{F0_limit[0]}-{F0_limit[1]}_" \
           f"Freq{np.round(T / t_div_freq[1], 2)}-{np.round(T / t_div_freq[0], 2)}"

f02use = rand_signal*scale

# assign processed stimulus to the Klaat object
s.params["F0"] = f02use
s.run()  # Rerun synthesizer
s.play()  # Play the trajectory in auditory domain

fps = 2

present_signal(mywin, split_x, split_y, dt=dt, on_off_ratio=0.7, record=False, fps=2, basename='default')

savepath = os.path.dirname(__file__) + "/example_simple.wav"
s.save(savepath)

time.sleep(T)
