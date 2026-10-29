import pandas as pd
import numpy as np
import logging
from os import listdir

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


energy_PATH = "../data/raw/energy/new/"
oil_PATH  = "../data/raw/energy/petro_only/"

def load_energy():
    
    energy_files = listdir(energy_PATH)
    try:
        energy_files.remove('.ipynb_checkpoints')
    except Exception:
        pass
    print(energy_files)
    dfs=[]
    for file in energy_files:
        if 'syncthing' in file:
            continue
        if '~' in file:
            continue
        logging.info(f'reading {file}')
        energy_data_piece = pd.read_csv(energy_PATH+file, encoding='latin-1', index_col=False)#, index_col=None, header=None, quoting=csv.QUOTE_ALL)
        if 'mirror' in file:  # For mirrored data replace ego an alter labels
            energy_data_piece.rename({'partnerISO':'reporterISO', 'reporterISO':'partnerISO'}, inplace=True, axis=1)
        if energy_data_piece.shape[0] >= 250000:
            logging.error(f'file to big, some data is most probavly cut ({file}:{energy_data_piece.shape[0]})')
        dfs += energy_data_piece,
    energy_df = pd.concat(dfs)
    energy_df['cmdCode']=energy_df['cmdCode'].replace({'TOTAL':np.nan})
    energy_df=energy_df.dropna(subset='cmdCode')
    
    # removing possible duplicates due to mirror operations, keeping largest
    energy_df.sort_values(by='primaryValue', ascending=False, inplace=True)
    duplicates = energy_df[energy_df.duplicated(subset=['refYear', 'reporterISO', 'partnerISO', 'cmdCode'])]
    logging.info(f"Removing {duplicates.shape[1]} rows from {duplicates['reporterISO'].unique()}, years {duplicates['refYear'].unique()}")
    energy_df.drop_duplicates(subset=['refYear', 'reporterISO', 'partnerISO', 'cmdCode'], inplace=True, keep='first')
    # REMOVING IMPORT DATA
    energy_df = energy_df[energy_df['flowDesc']=='Export']
    
    energy_df=energy_df.groupby(['refYear', 'reporterISO', 'partnerISO']).sum().reset_index()  # Тут может сломаться год
    return energy_df

def load_oil():
    
    energy_files = listdir(oil_PATH)
    try:
        energy_files.remove('.ipynb_checkpoints')
    except Exception:
        pass
    print(energy_files)
    dfs=[]
    for file in energy_files:
        if 'syncthing' in file:
            continue
        if '~' in file:
            continue
        logging.info(f'reading {file}')
        energy_data_piece = pd.read_csv(oil_PATH+file, encoding='latin-1', index_col=False)#, index_col=None, header=None, quoting=csv.QUOTE_ALL)
        if 'mirror' in file:  # For mirrored data replace ego an alter labels
            energy_data_piece.rename({'partnerISO':'reporterISO', 'reporterISO':'partnerISO'}, inplace=True, axis=1)
        if energy_data_piece.shape[0] >= 250000:
            logging.error(f'file to big, some data is most probavly cut ({file}:{energy_data_piece.shape[0]})')
        dfs += energy_data_piece,
    energy_df = pd.concat(dfs)
    energy_df['cmdCode']=energy_df['cmdCode'].replace({'TOTAL':np.nan})
    energy_df=energy_df.dropna(subset='cmdCode')
    
    # removing possible duplicates due to mirror operations, keeping largest
    energy_df.sort_values(by='primaryValue', ascending=False, inplace=True)
    duplicates = energy_df[energy_df.duplicated(subset=['refYear', 'reporterISO', 'partnerISO', 'cmdCode'])]
    logging.info(f"Removing {duplicates.shape[1]} rows from {duplicates['reporterISO'].unique()}, years {duplicates['refYear'].unique()}")
    energy_df.drop_duplicates(subset=['refYear', 'reporterISO', 'partnerISO', 'cmdCode'], inplace=True, keep='first')
    energy_df['flowDesc']='Export'
    #energy_df=energy_df.groupby(['refYear', 'reporterISO', 'partnerISO']).sum().reset_index()
    return energy_df
    
def preprocess_energy(df, year_start=1985, year_end=2022, rolling_window=5, normalize=True, test_data=True):

    source_name = preprocess_energy.__name__.split('_')[1]
    EGO_LABEL = 'reporterISO'
    ALTER_LABEL = 'partnerISO'
    YEAR_LABEL = 'refYear'
    VALUE_LABEL = 'primaryValue'
    
    #def preprocessed_energy(df, year_start, rolling_window=5):
    logging.info("Preprocessing energy data")  # basic preprocessing already done
    #removing old or too contemporary data
    df = df[df[YEAR_LABEL].astype(int) >= year_start]
    df = df[df[YEAR_LABEL].astype(int) <= year_end]
    
    df = convert_country_df(df, ALTER_LABEL, standard_to_convert='STATE_en_UN', purge=True)
    df = convert_country_df(df, EGO_LABEL, standard_to_convert='STATE_en_UN', purge=True)

    countries_all = get_all_countries()
    df_triple = df.groupby([YEAR_LABEL, ALTER_LABEL, EGO_LABEL]).sum()
    #df_triple = df_triple[df_triple[VALUE_LABEL]!=0]
    
    df_triple = pd.DataFrame(df_triple[VALUE_LABEL])
    #return df, df_triple
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
    
    logging.info("Done preprocessing energy Data (COMenergy)")
    return df, df_triple