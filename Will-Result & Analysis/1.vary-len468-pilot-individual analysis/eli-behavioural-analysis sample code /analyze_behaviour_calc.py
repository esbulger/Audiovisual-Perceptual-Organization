"""

Calculate d prime

"""



def calc_signal_detection(df, cond_col='nTalkers', conditions=[1, 6],
                          same_val='same', diff_val='different'):
    """

    :param df:
    :param cond_col:
    :param conditions:
    :return:
    """

    import numpy as np

    # set differentiators
    subjects = df['subject'].unique()

    dp_results = np.zeros((len(subjects), len(conditions)))

    # for each subject
    for i, subj in enumerate(subjects):

        df_subj = df.loc[df['subject'] == subj]

        # for each condition
        for j, cond in enumerate(conditions):

            df_temp = df_subj.loc[df_subj[cond_col] == cond]

            if df_temp.empty:
                dp_results[i, j] = np.nan
            else:
                dp_results[i, j] = calc_d_prime(df_temp, same_val, diff_val)

    return dp_results



def calc_d_prime(df, same_val='same', diff_val='different'):
    """
    input a dataframe with a column for 'correct' response and a column for 'same vs different' trialType

    :return:
    """

    from pychoacoustics import pysdt

    same_trials = df.loc[df['trialType'] == same_val]
    diff_trials = df.loc[df['trialType'] == diff_val]

    # get the proportions for the overall dataset
    nCA = same_trials['respCorrect'].sum()
    nTA = float(len(same_trials['respCorrect']))

    nIB = diff_trials['respCorrect'].sum()
    nTB = float(len(diff_trials['respCorrect']))

    dp = pysdt.dprime_SD_from_counts(nCA, nTA, nIB, nTB, meth='diff', corr=True)

    #HR, FA = pysdt.compute_proportions(nCA, nTA, nIB, nTB, corr='loglinear')
    #dp = pysdt.dprime_SD(HR, FA, 'diff')

    return dp
