
# --------------------------------++++++++++++++++++++++++++--------------------------------
# --------------------------------       Extract Data       --------------------------------
# --------------------------------++++++++++++++++++++++++++--------------------------------

# function to get data from all gorilla participants and organize based on column
def GetData(dirname, nParts, columns, bad_ID=None):
    """

    Extracts data from a short form sheet (csv, but can be changed to xlsx by editing ExtractDataCSV)
    Ignores rows from participants with poor data

    :param dirname: directory containing files
    :param nParts: number of participants to include in data array
    :param columns: The column titles in the excel spreadsheet
    :param bad_ID: array of participant public IDs with poor data / excluded for any reason

    :return: numpy object array storing dataframes of each participant
    """

    from os import listdir
    from os.path import isfile, join
    import numpy as np
    import pandas as pd

    all_IDs = np.array([])

    # get all file names in directory
    onlyfiles = [f for f in listdir(dirname) if isfile(join(dirname, f))]

    df = np.zeros((len(onlyfiles)), dtype=object)
    count = 0

    # for each file, read into pandas, then append
    for sheet in onlyfiles:

        # get dataframe with specified columns
        df_i = ExtractDataCSV(dirname+sheet, columns)

        # get list of IDs in the spreadsheet
        IDs = np.array(df_i['subject'].unique())
        IDs = IDs[~(IDs.astype(str) == 'nan')]

        # remove the bad participant's IDs from the IDs array
        good_IDs = IDs[~(np.isin(IDs, bad_ID))]

        # for each participant ID
        for i in good_IDs:

            # get dataframe for each participant
            df[count] = df_i.loc[df_i['subject'] == i]
            count += 1

        all_IDs = np.append(all_IDs, good_IDs)

    df_concat = pd.concat(df)

    # returns a numpy array with a dataframe for each participant
    return df_concat, all_IDs


# --------------------------------++++++++++++++++++++++++++--------------------------------
# --------------------------------     Other Helper Functions     --------------------------------
# --------------------------------++++++++++++++++++++++++++--------------------------------

# function to convert xlsx data into a dataframe
def ExtractDataXlsx(data_path, columns):
    """

    Extracts xlsx data and converts to a dataframe
    Only keeps columns of specified names

    :param data_name:
    :param columns: 1D array of column names to be kept
    :return:
    """

    import pandas as pd
    import os

    # script_dir = os.path.dirname(__file__)
    # abs_file_path = os.path.join(script_dir, data_path)

    df = pd.read_excel(data_path)

    # Select the ones you want
    df_clean = df[columns]

    return df_clean


def ExtractDataCSV(data_path, columns):
    """

    Extracts xlsx data and converts to a dataframe
    Only keeps columns of specified names

    :param data_name:
    :param columns: 1D array of column names to be kept
    :return:
    """

    import pandas as pd
    import os

    # get working directory and combine with file name (data_path)
    # script_dir = os.path.dirname(__file__)
    # abs_file_path = os.path.join(script_dir, data_path)

    # use pandas to read csv
    df = pd.read_csv(data_path)

    # Select the ones you want
    df_clean = df[columns]

    return df_clean


# function to keep the responses that we are looking for only
def ExtractResponses(df, resp_column, resp_type):
    """
    Takes a dataframe
    Returns a dataframe where specified column holds values specified

    ex. In Gorilla (online psychology software), there can be  multiple rows for one trial
    This function returns a dataframe holding only the rows corresponding to the response

    resp_column typically would be: Zone Type or Zone Name
    resp_type can be: advancement zone OR response_keyboard, continue_button, etc.

    :param df:
    :param resp_column:
    :param resp_type:
    :return:
    """

    df_resp = df.loc[df[resp_column] == resp_type]

    return df_resp


