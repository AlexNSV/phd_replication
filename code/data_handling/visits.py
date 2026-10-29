import pandas as pd
import logging
from os import listdir

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


visits_PATH = "../data/raw/visits/Diplometrics_COLT_Travel_Dataset_Primary-HOGS-1990-2024_20250317.xlsx"
HOGS_WEIGHTS_DICT = {
        'GS' : 1.0, # Head of state and government
        'G' : 0.9, # head of government
        'S' : 0.9, # head of state
        'De Facto' : 0.9,  # de facto country leader
        'F' : 0.0,  # Foreign Minister
        'E' : 0.0, # Minister of Finance
        'I' : 0.0, # IGO Leader
        'CP' : 0.0  # Crown Prince
}
MULTILATERAL_DOWNGRADE_DICT = {
    'Yes' : 0.25,  # had multilateral meetings
    'No' : 1.0  # no multilasteral meetings
}
HOST_HOG_MEETING_DOWNGRADE_DOCT = {
    'Yes' : 1.0,  # had meetings with host country's HOGs
    'No' : 0.5  # had no meetings with host country's HOGs
}

def load_visits():
    df = pd.read_excel(visits_PATH)

    # visit weighting
    df['value'] = df['LeaderRole'].replace(HOGS_WEIGHTS_DICT)
    df['AttendedMultilatEvent'].replace(MULTILATERAL_DOWNGRADE_DICT, inplace=True)
    df['MetHostHoGS'].replace(MULTILATERAL_DOWNGRADE_DICT, inplace=True)
    df['value'] = df['value'].astype(float) * df['AttendedMultilatEvent'].astype(float) * df['MetHostHoGS'].astype(float)
    
    logging.info("Loaded Visits COLT Database")
    return df

def preprocess_visits(df, year_start=1985, year_end=2022, rolling_window=5, test_data=True, normalize=True):
    #def preprocessed_visits(df, year_start, rolling_window=5):
    logging.info("Preprocessing visits data")  # basic preprocessing already done
    source_name = preprocess_visits.__name__.split('_')[1]
    
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
    
    logging.info("Done preprocessing visits data (COLT)")
    return df, df_triple