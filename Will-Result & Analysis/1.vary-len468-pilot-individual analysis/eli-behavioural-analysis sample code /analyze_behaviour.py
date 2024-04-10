"""

Eli Bulger

April 2022

"""

import os
from analyze_behaviour_helper import *
from analyze_behaviour_calc import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


timing = False

# ID's of participants with data we don't want to include in analysis
bad_IDs = []
Parts = 6
nParts = Parts - len(bad_IDs)

base_Dir = os.path.dirname(__file__)

columns = ['subject', 'stimFilename', 'trialType', 'seqLength', 'interElementTime', 'interSeqTime', 'interTrialTime',
           'blockNum', 'trialNum', 'nTalkers', 'nSyllables', 'nPitches', 'respTime',
           'respCorrect', 'sequenceItems', 'switch_items', 'sensoryModality', 'Serial_Pos', 'Order']

# task data excel sheet relative path
data_talker, IDs_talker = GetData(base_Dir + '/in_lab_results/talkers study/',
                                  nParts,
                                  columns=columns,
                                  bad_ID=bad_IDs)

if timing:
    data_timing, IDs_timing = GetData(base_Dir + '/in_lab_results/timing study/',
                                      nParts,
                                      columns=columns,
                                      bad_ID=bad_IDs)

# get d-prime statistics - same/different
dp_talker_sd = calc_signal_detection(data_talker, cond_col='nTalkers', conditions=[1, 6],
                                  same_val='same', diff_val='different')
if timing:
    dp_timing_sd = calc_signal_detection(data_timing, cond_col='interElementTime', conditions=[0, 400],
                                  same_val='same', diff_val='different')

# get d-prime statistics - present/absent
dp_talker_ap = calc_signal_detection(data_talker, cond_col='nTalkers', conditions=[1, 6],
                                  same_val='present', diff_val='absent')

if timing:
    dp_timing_ap = calc_signal_detection(data_timing, cond_col='interElementTime', conditions=[0, 400],
                                  same_val='present', diff_val='absent')


df_talk_sd = pd.DataFrame(dp_talker_sd, columns=['1 talker', '6 talkers'])
df_talk_sd['trialType'] = 'sd'
df_talk_sd['subject'] = np.arange(Parts)

df_talk_ap = pd.DataFrame(dp_talker_ap, columns=['1 talker', '6 talkers'])
df_talk_ap['trialType'] = 'ap'
df_talk_ap['subject'] = np.arange(Parts)

if timing:
    df_time_sd = pd.DataFrame(dp_timing_sd, columns=['0 ms', '400 ms'])
    df_time_sd['trialType'] = 'sd'
    df_time_sd['subject'] = np.arange(Parts)

    df_time_ap = pd.DataFrame(dp_timing_ap, columns=['0 ms', '400 ms'])
    df_time_ap['trialType'] = 'ap'
    df_time_ap['subject'] = np.arange(Parts)

df_talk = pd.concat([df_talk_sd, df_talk_ap])
df_talk_long = pd.melt(df_talk, id_vars=['subject', 'trialType'], value_vars=['1 talker', '6 talkers'], value_name="d'")

if timing:
    df_time = pd.concat([df_time_sd, df_time_ap])
    df_time_long = pd.melt(df_time, id_vars=['subject', 'trialType'], value_vars=['0 ms', '400 ms'], value_name="d'")

# plotting
sns.catplot(x='variable', y="d'", hue='trialType', kind='bar', data=df_talk_long)
plt.title(f'Talker Study Detection Results (n={Parts})')
plt.ylabel("d'")
plt.xlabel("Condition")
plt.tight_layout(pad=5)

if timing:
    sns.catplot(x='variable', y="d'", hue='trialType', kind='bar', data=df_time_long)
    plt.title(f'Timing Study Detection Results (n={Parts})')
    plt.ylabel("d'")
    plt.xlabel("Condition")
    plt.tight_layout(pad=5)

plt.close()

sns.pointplot(data=df_talk_long, x="variable", y="d'", hue="trialType", col='subject', marker='o')

g = sns.FacetGrid(df_talk_long, col="subject", hue="trialType", col_wrap=2, height=2, ylim=(0, 5))
g.map(sns.pointplot, "variable", "d'", ci=None, legend=True)
g.add_legend()

if timing:
    sns.pointplot(data=df_time_long, x="variable", y="d'", hue="trialType", col='subject', marker='o')

    g = sns.FacetGrid(df_time_long, col="subject", hue="trialType", col_wrap=2, height=2, ylim=(0, 5))
    g.map(sns.pointplot, "variable", "d'", ci=None, legend=True)
    g.add_legend()

"""
avg_talker_sd = np.mean(dp_talker_sd, axis=0)
se_talker_sd = np.std(dp_talker_sd, axis=0)/np.sqrt(Parts)

avg_timing_sd = np.mean(dp_timing_sd, axis=0)
se_timing_sd = np.std(dp_timing_sd, axis=0)/np.sqrt(Parts)

avg_talker_ap = np.mean(dp_talker_ap, axis=0)
se_talker_ap = np.std(dp_talker_ap, axis=0)/np.sqrt(Parts)

avg_timing_ap = np.mean(dp_timing_ap, axis=0)
se_timing_ap = np.std(dp_timing_ap, axis=0)/np.sqrt(Parts)


plt.bar([1, 2], avg_talker)
plt.errorbar([1, 2], avg_talker, se_talker, fmt='none')
plt.title('Same/Different Detection - Talkers Study')
plt.ylabel(['d'])
plt.show()

plt.bar([1, 2], avg_timing)
plt.errorbar([1, 2], avg_timing, se_timing, fmt='none')
plt.title('Same/Different Detection - Timing Study')
plt.ylabel(['d'])
plt.xlabel(['1: 0 ms, 2: 400 ms'])
plt.show()
"""