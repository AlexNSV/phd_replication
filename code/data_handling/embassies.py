import pandas as pd
import logging
from os import listdir

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


embassies_PATH = "../data/raw/embassies/Diplometrics_Diplomatic-Representation_1960-2022_20230831.xlsx"


def load_embassies():
    df = pd.read_excel(embassies_PATH)
    raise Exception("Underway")
    # IN THR MAKING
    # visit weighting
    df['value'] = df['LeaderRole'].replace(HOGS_WEIGHTS_DICT)
    df['AttendedMultilatEvent'].replace(MULTILATERAL_DOWNGRADE_DICT, inplace=True)
    df['MetHostHoGS'].replace(MULTILATERAL_DOWNGRADE_DICT, inplace=True)
    df['value'] = df['value'].astype(float) * df['AttendedMultilatEvent'].astype(float) * df['MetHostHoGS'].astype(float)
    
    logging.info("Loaded embassies COLT Database")
    return df

def preprocess_embassies(df, year_start=1985, year_end=2022, rolling_window=5, test_data=True, normalize=True):
    #def preprocessed_embassies(df, year_start, rolling_window=5):
    logging.info("Preprocessing embassies data")  # basic preprocessing already done
    source_name = preprocess_embassies.__name__.split('_')[1]
    
    EGO_LABEL = 'CountryVisitedISO'
    ALTER_LABEL = 'LeaderCountryISO'
    YEAR_LABEL = 'TripYear'
    VALUE_LABEL = 'value'
    #removing old data
    df = df[df[YEAR_LABEL].astype(int) >= year_start]
    
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
    if normalize:
        df_triple['value'] = df_triple['value'] / df_triple['value'].max()
    
    logging.info("Done preprocessing embassies data (COLT)")
    return df, df_triple