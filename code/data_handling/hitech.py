import pandas as pd
import numpy as np
import logging
from os import listdir
import csv

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


hitech_PATH = "../data/raw/hitech/yearly/"

def load_hitech():
    
    def get_removal_values(df_in, year, ego, alter, value):
        # removes sum of df_in from df_outs
        # df_in for exceptions and df_out for main values
        df_temp = df_in[(df_in['refYear']==year) & (df_in['reporterISO']==ego) & (df_in['partnerISO']==alter)]
        #print(df_temp)
        removal = df_temp['primaryValue'].sum()
        return value - removal
        
    hitech_files = listdir(hitech_PATH)
    try:
        hitech_files.remove('.ipynb_checkpoints')
    except Exception:
        pass
    print(hitech_files)
    dfs=[]
    for file in hitech_files:
        if 'syncthing' in file:
            continue
        if '~' in file:
            continue
        logging.info(f'reading {file}')
        hitech_data_piece = pd.read_csv(hitech_PATH+file, encoding='latin-1', index_col=False)#, index_col=None, header=None, quoting=csv.QUOTE_ALL)
        if 'mirror' in file:  # For mirrored data replace ego an alter labels
            hitech_data_piece.rename({'partnerISO':'reporterISO', 'reporterISO':'partnerISO'}, inplace=True, axis=1)
        dfs += hitech_data_piece,
    hitech_df = pd.concat(dfs)
    hitech_df['cmdCode']=hitech_df['cmdCode'].replace({'TOTAL':np.nan})
    hitech_df=hitech_df.dropna(subset='cmdCode')

    # removing possible duplicates due to mirror operations, keeping largest
    hitech_df.sort_values(by='primaryValue', ascending=False, inplace=True)
    duplicates = hitech_df[hitech_df.duplicated(subset=['refYear', 'reporterISO', 'partnerISO', 'cmdCode'])]
    logging.info(f"Removing {duplicates.shape[0]} rows from {duplicates['reporterISO'].unique()}, years {duplicates['refYear'].unique()}")
    hitech_df.drop_duplicates(subset=['refYear', 'reporterISO', 'partnerISO', 'cmdCode'], inplace=True, keep='first')
    
    df_notin = hitech_df[~hitech_df['cmdCode'].isin(['71489', '71499', '76439', '76499', '89965', '77861', '77866', '77869'])]
    #df_notin = df_notin.groupby(["refYear", "reporterISO", "partnerISO"]).sum()
    df_in = hitech_df[hitech_df['cmdCode'].isin(['71489', '71499', '76439', '76499', '89965', '77861', '77866', '77869'])]  # exceptions to be removed
    #df_in = df_in.groupby(["refYear", "reporterISO", "partnerISO"]).sum()
    hitech_df=df_notin.groupby(['refYear', 'reporterISO', 'partnerISO']).sum().reset_index()
    hitech_df['value']=hitech_df.apply(lambda x: get_removal_values(df_in, x['refYear'], x['reporterISO'], x['partnerISO'], x['primaryValue']), axis=1)
    return hitech_df
    

def preprocess_hitech(df, year_start=1985, year_end=2022, rolling_window=5, normalize=True, test_data=True):

    source_name = preprocess_hitech.__name__.split('_')[1]
    EGO_LABEL = 'reporterISO'
    ALTER_LABEL = 'partnerISO'
    YEAR_LABEL = 'refYear'
    VALUE_LABEL = 'primaryValue'
    
    #def preprocessed_hitech(df, year_start, rolling_window=5):
    logging.info("Preprocessing hitech data")  # basic preprocessing already done
    #removing old or too contemporary data
    df = df[df[YEAR_LABEL].astype(int) >= year_start]
    df = df[df[YEAR_LABEL].astype(int) <= year_end]
    
    df = convert_country_df(df, ALTER_LABEL, standard_to_convert='STATE_en_UN', purge=True)
    df = convert_country_df(df, EGO_LABEL, standard_to_convert='STATE_en_UN', purge=True)
    
    countries_all = get_all_countries()
    df_triple = df.groupby([YEAR_LABEL, ALTER_LABEL, EGO_LABEL]).sum()
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
    if normalize:
        df_triple['value'] = df_triple['value'] / df_triple['value'].max()
    
    logging.info("Done preprocessing hitech Data (COMhitech)")
    return df, df_triple