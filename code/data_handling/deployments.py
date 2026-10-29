import pandas as pd
import numpy as np
import logging


import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


DEPLOYMENTS_PATH = "../data/raw/deployments/IMDT - 12.01.24.xlsx"

def load_deployments(filter_un=False):
    """Loads DEPLOYMENTS data
    filter_un - whether to filter UN-related Data
    """
    df = pd.read_excel(DEPLOYMENTS_PATH)
    if filter_un:
        logging.info('Removing UN-related data')
        df['UN'] = df['UN'].fillna(1).astype(int)
        df = df[df['UN']==0]
    logging.info("Loaded IMDT deployments data")
    return df


def preprocess_deployments(df, nodatatroopfiller = 10, rolling_window = 5, win_type='hann', year_start=1985, year_end=2022, test_data=True, logarithmic=False):
    
    logging.info("Preprocessing IMDT Depoyment Data")

    source_name = preprocess_deployments.__name__.split('_')[1]
    
    EGO_LABEL = 'CountryName1'
    ALTER_LABEL = 'CountryName2'
    YEAR_LABEL = 'year'
    VALUE_LABEL = 'Troops'
    
    # filling no data with troop numbers
    df['Troops'] = df['Troops'].fillna(nodatatroopfiller)
    
    #removing old data
    df = df[df['year'] >= year_start]
    
    #converting to STATE_en_UN
    df = convert_country_df(df, 'CountryName1', standard_to_convert='STATE_en_UN', purge=True)
    df = convert_country_df(df, 'CountryName2', standard_to_convert='STATE_en_UN', purge=True)
    
    countries_all = get_all_countries()

    #---------------
    #d_ = df.reset_index()
    #d_ = d_[[EGO_LABEL, ALTER_LABEL, YEAR_LABEL, VALUE_LABEL]]
    #print(d_)
    #d_=d_.groupby([EGO_LABEL,YEAR_LABEL]).sum().reset_index()
    #print("DFPRINT")
    #print(d_[d_[EGO_LABEL]=='Sweden'])
    #---------------
    
    df_triple = df.groupby(["year", "CountryName1", "CountryName2"]).sum()
    df_triple = df_triple[df_triple['Troops']!=0]

    if test_data:
        test_df(df_triple.reset_index(), source_name, year_start=year_start, year_end=year_end, alter_label=ALTER_LABEL, ego_label=EGO_LABEL, year_label=YEAR_LABEL, value_label=VALUE_LABEL)

    if logarithmic:
        logging.info("Converting troops to log 10 scale")
        df_triple['Troops'] = np.log10(df_triple['Troops'])
        test_df(df_triple.reset_index(), source_name, year_start=year_start, year_end=year_end, alter_label=ALTER_LABEL, ego_label=EGO_LABEL, year_label=YEAR_LABEL, value_label=VALUE_LABEL)
        
    df_triple = pd.DataFrame(df_triple['Troops'])
    # filling an empty year*country*country dataframe with zeroes
    empty_df=get_empty_country_df(years=df['year'].unique(), countries_all=countries_all, names=["year", "CountryName1", "CountryName2"])
    
    # merging the df with zero df
    df_triple = empty_df.merge(df_triple,left_index=True, right_index=True, how='outer').fillna(0)
    
    # counting rolling average
    df_triple = df_triple.reset_index().set_index('year').groupby(['CountryName2', 'CountryName1']).rolling(rolling_window, min_periods=1, win_type=win_type).mean()
    
    df_triple = df_triple.reset_index().set_index(['year', 'CountryName2', 'CountryName1'])
    
    df_triple.index.names = ['year', 'alter', 'ego']
    df_triple.rename(columns={'Troops':'value'}, inplace=True)
    
    # Normalization
    df_triple['value'] = df_triple['value'] / df_triple['value'].max()
    
    logging.info("Done preprocessing IMDT Depoyment Data")

    return df, df_triple