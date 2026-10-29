import pandas as pd
import logging
from os import listdir

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


TRADE_PATH = "../data/raw/trade/"

def load_trade():
    trade_files = listdir(TRADE_PATH)#.remove('.ipynb_checkpoints')
    dfs=[]
    for file in trade_files:
        if 'syncthing' in file:
            continue
        if '~' in file:
            continue
        trade_data_piece = pd.read_csv(TRADE_PATH+file, encoding='latin-1', index_col=False)
        logging.info(f'reading {file}')
        if 'mirror' in file:  # For mirrored data replace ego an alter labels
            trade_data_piece.rename({'partnerISO':'reporterISO', 'reporterISO':'partnerISO'}, inplace=True, axis=1)
        if trade_data_piece.shape[0] >= 250000:
            logging.error(f'file to big, some data is most probavly cut ({file}:{trade_data_piece.shape[0]})')
        dfs+=trade_data_piece,
    trade_df = pd.concat(dfs)
    
    # removing possible duplicates due to mirror operations, keeping largest
    trade_df.sort_values(by='primaryValue', ascending=False, inplace=True)
    duplicates = trade_df[trade_df.duplicated(subset=['refYear', 'reporterISO', 'partnerISO', 'cmdCode'])]
    logging.info(f"Removing {duplicates.shape[1]} rows from {duplicates['reporterISO'].unique()}, years {duplicates['refYear'].unique()}")
    trade_df.drop_duplicates(subset=['refYear', 'reporterISO', 'partnerISO', 'cmdCode'], inplace=True, keep='first')
    
    return trade_df

def preprocess_trade(df, year_start=1985, year_end=2022, rolling_window=5, test_data=True):
    #def preprocessed_trade(df, year_start, rolling_window=5):
    logging.info("Preprocessing trade data")  # basic preprocessing already done
    source_name = preprocess_trade.__name__.split('_')[1]
    
    EGO_LABEL = 'reporterISO'
    ALTER_LABEL = 'partnerISO'
    YEAR_LABEL = 'refYear'
    VALUE_LABEL = 'primaryValue'
    #removing old data
    df = df[df[YEAR_LABEL].astype(int) >= year_start]
    df = df[df[YEAR_LABEL].astype(int) <= year_end]
    
    df = convert_country_df(df, EGO_LABEL, standard_to_convert='STATE_en_UN', purge=True)
    df = convert_country_df(df, ALTER_LABEL, standard_to_convert='STATE_en_UN', purge=True)
    
    countries_all = get_all_countries()
    df_triple = df.groupby([YEAR_LABEL, EGO_LABEL, ALTER_LABEL]).sum()
    df_triple = df_triple[df_triple[VALUE_LABEL]!=0]
    
    df_triple = pd.DataFrame(df_triple[VALUE_LABEL])
    
    if test_data:
        test_df(df_triple.reset_index(), source_name, year_start=year_start, year_end=year_end, alter_label=ALTER_LABEL, ego_label=EGO_LABEL, year_label=YEAR_LABEL, value_label=VALUE_LABEL) 
        
    # filling an empty year*country*country dataframe with zeroes
    
    empty_df=get_empty_country_df(years=df[YEAR_LABEL].unique(), countries_all=countries_all, names=[YEAR_LABEL, EGO_LABEL, ALTER_LABEL])

    # merging the df with zero df
    df_triple = empty_df.merge(df_triple,left_index=True, right_index=True, how='outer').fillna(0)
    
    # counting rolling average
    df_triple = df_triple.reset_index().set_index(YEAR_LABEL).groupby([ALTER_LABEL, EGO_LABEL]).rolling(rolling_window, min_periods=1).mean()
    
    df_triple = df_triple.reset_index().set_index([YEAR_LABEL, ALTER_LABEL, EGO_LABEL])
    
    df_triple.index.names = ['year', 'alter', 'ego']
    df_triple = df_triple.rename({VALUE_LABEL:'value'}, axis=1)
    
    # Normalization
    df_triple['value'] = df_triple['value'] / df_triple['value'].max()
    
    logging.info("Done preprocessing Trade Data (COMTRADE)")
    return df, df_triple