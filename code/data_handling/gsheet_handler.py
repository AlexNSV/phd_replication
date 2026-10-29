import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from gspread.exceptions import APIError
from gspread.utils import rowcol_to_a1
from requests import ConnectionError

import pandas as pd
import random
import logging
import time
from functools import wraps

from data_handling.datavalue import DataValue, get_datavalue_dict

def setup_google_client():
    # client email: docanomi-gsheet@docanomi.iam.gserviceaccount.com
    gc = gspread.service_account(filename='../docanomi-eb8ee4c227f6.json')
    logging.debug("Google client set up")
    return gc


def retry(exceptions, total_tries=7, initial_wait=4, backoff_factor=2, logger=logging.getLogger(__name__)):
    """
    calling the decorated function applying an exponential backoff.
    Args:
        exceptions: Exception(s) that trigger a retry, can be a tuple
        total_tries: Total tries
        initial_wait: Time to first retry
        backoff_factor: Backoff multiplier (e.g. value of 2 will double the delay each retry).
    """
    def retry_decorator(f):
        @wraps(f)
        def func_with_retries(*args, **kwargs):
            _tries, _delay = total_tries + 1, initial_wait
            while _tries > 1:
                try:
                    log(f'{total_tries + 2 - _tries}. try:', logger)
                    return f(*args, **kwargs)
                except exceptions as e:
                    _tries -= 1
                    print_args = args if args else 'no args'
                    if _tries == 1:
                        msg = str(f'Function: {f.__name__}\n'
                                  f'Failed despite best efforts after {total_tries} tries.\n'
                                  f'args: {print_args}, kwargs: {kwargs}')
                        log(msg, logger)
                        raise
                    msg = str(f'Function: {f.__name__}\n'
                              f'Exception: {e}\n'
                              f'Retrying in {_delay} seconds!, args: {print_args}, kwargs: {kwargs}\n')
                    log(msg, logger)
                    print(msg)
                    time.sleep(_delay)
                    _delay *= backoff_factor

        return func_with_retries
    return retry_decorator

def log(msg, logger=None):
    if logger:
        logger.debug(msg)
    else:
        print(msg)

def remove_empty(df):
    """Removes empty indices and columns from google sheets import

    Arguments:
        df (Pandas.DataFrame): df, presumably from gspread_dataframe.get_as_dataframe
    """
    df = df.loc[pd.notna(df.index),:]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    df = df.dropna(how='all')
    return df

@retry((APIError, ConnectionError))
def read_gsheet(tablename: str, sheetname: str, evaluate_formulas=True, index_col=None, clear_empty=True, skiprows=1):
    """Returns a sheet from google table as a Pandas.DataFrame

    Arguments:
        tablename (str): name of the table in Google.Sheets
        sheetname (str): name of the sheet in the table
        evaluate_formulas (bool): whether to avaluate Google.Sheets formulas when reading gsheet
        index_col (str): name of the column to be treated as the index, default is None
        clear_empty (bool): whether to remove empty rows and Unnamd columns, default is True

    Returns:
        df (Pandas.DataFrame): data from Google.Sheets in a DataFrame
    """
    logging.debug(f"reading google table {tablename}, sheet {sheetname}")
    gtable = gc.open(tablename)  # Note to self: implement cashing some day to avoid additional API calls
    gsheet = gtable.worksheet(sheetname)
    df = get_as_dataframe(gsheet, evaluate_formulas=evaluate_formulas, index_col=index_col, skiprows=skiprows)
    if clear_empty: df = remove_empty(df)
    return df

@retry((APIError, ConnectionError))
def read_gsheets(tablename: str, sheets: list, evaluate_formulas=True, index_col=None, clear_empty=True, return_format='dict'):
    """Returns multiple sheets from a google table as a dict of DataFrames

    Arguments:
        tablename (str): name of the table in Google.Sheets
        sheets (list of str): list of names of the sheet in the table
        evaluate_formulas (bool): whether to avaluate Google.Sheets formulas when reading gsheet
        index_col (str): name of the column to be treated as the index, default is None
        clear_empty (bool): whether to remove empty rows and Unnamd columns, default is True
        return_format : str, format to return values 'dict' or 'list', default is 'dict'

    Returns:
        dfs (dict | list) : Dictionary {'sheetname1':df1, 'sheetname2':df2, ...} or list as [df1, df2, ...]
    """
    logging.debug(f"reading google table {tablename}, sheets {sheets}")
    gtable = gc.open(tablename)
    
    if return_format == 'dict':
        dfs = {}
    elif return_format == 'list':
        dfs = []
    else:
        raise ValueError(f"{return_format} is invalid for return_format, use 'list' or 'dict' instead")
    
    for sheetname in sheets:
        gsheet = gtable.worksheet(sheetname)
        df = get_as_dataframe(gsheet, evaluate_formulas=evaluate_formulas, index_col=index_col)
        if clear_empty: df = remove_empty(df)
        if return_format == 'dict':
            dfs[sheetname] = df    
        else: 
            dfs += df,
    return dfs

@retry((APIError, ConnectionError))
def replace_gsheet(tablename: str, sheetname: str, df: pd.DataFrame, include_index=True):
    """Replaces a google sheet with a Pandas.DataFrame.
    Efficient (one API call), but risky, as all data is rewritten.

    Arguments:
        tablename (str): name of the table in Google.Sheets
        sheetname (str): name of the sheet in the table
        df (Pandas.DataFrame) : DataFrame to replace google sheet
        include_index (bool): whether to include df index, default is True

    Returns:
        df (Pandas.DataFrame): data from Google.Sheets in a DataFrame
    """
    logging.info(f"replacing google table {tablename}, sheet {sheetname} with dataframe of size {df.shape}")
    gsheet = gc.open(tablename).worksheet(sheetname)
    set_with_dataframe(gsheet, df, include_index=include_index)


@retry((APIError, ConnectionError))
def update_gsheet(datavalues: list, include_index=True):
    """Updates a google sheet with a Pandas.DataFrame using multiple datavalues
    

    Arguments:
        datavalues (list): list of DataValue objects
        include_index (bool): whether to include df index when writing to gsheet, default is True
    """
    logging.info(f"Updating using batch method, total changes = {len(datavalues)}")
    batches_dict = get_datavalue_dict(datavalues)
    
    for datamart in batches_dict:
        gtable = gc.open(datamart)
        for sheet in batches_dict[datamart]:
            gsheet = gtable.worksheet(sheet)
            update_list = list()  # list of updates to database
            for datavalue in batches_dict[datamart][sheet]:
                update_list += {
                    'range' : datavalue.a1,
                    'values' : [[datavalue.value]]
                },
            logging.debug(f"Updating datamart {datamart}, sheet {sheet} with values \n{update_list}")
            # See batch_update docs - https://docs.gspread.org/en/latest/api/models/worksheet.html#gspread.worksheet.Worksheet.batch_update
            gsheet.batch_update(update_list)

def test():
    print(f"Testing {__file__}")
    logging.info(f"Testing {__file__}")
    print("READING")
    df = read_gsheet('int_main_copy', 'i_main')
    print("WRITING")
    dv1 = DataValue('int_main_copy', 'i_main', 'i00204m', 'c_name_en', 'WORKS!', df=df, index_col='i_triple_id')
    dv2 = DataValue('int_main_copy', 'i_main', 'i00205m', 'c_name_en', 'WORKS!11111', df, index_col='i_triple_id')
    update_gsheet([dv1, dv2])
    print("TEST DONE")


gc = setup_google_client()
if __name__ == '__main__':
    test()