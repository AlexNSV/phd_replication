import pandas as pd
import numpy as np
import logging
from os import listdir

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


peacekeep_PATH = "../data/raw/peacekeeping/"

def load_peacekeep():
    #  Data in thousands
    peacekeep_df_post2010 = pd.read_csv(peacekeep_PATH+'DPO-UCHISTORICAL.csv', index_col = 'contribution_id')
    peacekeep_df_pre2010 = pd.read_csv(peacekeep_PATH+'DPPADPO-UCPre2010.csv', index_col = 'ID')
    peacekeep_missions = pd.read_excel(peacekeep_PATH+'mission_countries.xlsx')

    peacekeep_rename = {
    'Contributor':'contributing_country', 
    'Contributor_ISO-3':'isocode3', 
    'Troops':'troops',
    'Date':'last_reporting_date',
    'Mission':'mission_acronym'
    }
    peacekeep_df_pre2010.rename(peacekeep_rename, axis=1, inplace=True)
    
    peacekeep_df_post2010=peacekeep_df_post2010[peacekeep_df_post2010['personnel_type']=='Troops']
    peacekeep_df_post2010['troops'] = peacekeep_df_post2010['male_personnel'] + peacekeep_df_post2010['female_personnel']
    peacekeep_df_post2010 = peacekeep_df_post2010[~(peacekeep_df_post2010['troops']==0)]
    
    # Only multycountry post-2010 mission
    minusca_data = peacekeep_df_post2010.copy()[peacekeep_df_post2010['mission_acronym']=='MINUSCA']
    minusca_data['troops'] = minusca_data['troops']//2
    minusca_data_car = minusca_data.copy()
    minusca_data_chad = minusca_data.copy()
    minusca_data_car['Mission_Country'] = 'Central African Republic'
    minusca_data_chad['Mission_Country'] = 'Chad'
    minusca_data_car['Mission_Country_ISO-3'] = 'CAR'
    minusca_data_chad['Mission_Country_ISO-3'] = 'TCD'
    
    peacekeep_missions_names_dict = peacekeep_missions.set_index('Операция')['Страна (англ.)'].to_dict()
    peacekeep_missions_isos_dict = peacekeep_missions.set_index('Операция')['ISO-код'].to_dict()
    
    peacekeep_df_post2010_proc = peacekeep_df_post2010.copy()[~(peacekeep_df_post2010['mission_acronym']=='MINUSCA')]
    peacekeep_df_post2010_proc['Mission_Country'] = peacekeep_df_post2010_proc['mission_acronym'].apply(lambda x: peacekeep_missions_names_dict[x])
    peacekeep_df_post2010_proc['Mission_Country_ISO-3'] = peacekeep_df_post2010_proc['mission_acronym'].apply(lambda x: peacekeep_missions_isos_dict[x])
    
    peacekeep_df_post2010_done = pd.concat([peacekeep_df_post2010_proc, minusca_data_car, minusca_data_chad])
    
    common_columns = peacekeep_df_pre2010.columns.intersection(peacekeep_df_post2010_done.columns)
    
    peacekeep_df = pd.concat([peacekeep_df_pre2010[common_columns], peacekeep_df_post2010_done[common_columns]])
    peacekeep_df = peacekeep_df.dropna(subset=['troops'])
    peacekeep_df['last_reporting_date'] = pd.to_datetime(peacekeep_df['last_reporting_date'])
    peacekeep_df.sort_values('last_reporting_date', inplace=True)
    
    peacekeep_df['year'] = peacekeep_df['last_reporting_date'].apply(lambda x: x.year)
    
    peacekeep_df.drop_duplicates(subset=['isocode3', 'Mission_Country_ISO-3', 'year'], keep='last', inplace=True)
    
    return peacekeep_df
    

def preprocess_peacekeep(df, year_start, year_end, test_data=True, normalize=True, rolling_window=5):

    source_name = preprocess_peacekeep.__name__.split('_')[1]
    EGO_LABEL = 'isocode3'
    ALTER_LABEL = 'Mission_Country_ISO-3'
    YEAR_LABEL = 'year'
    VALUE_LABEL = 'troops'
    
    logging.info("Preprocessing peacekeep data")  # basic preprocessing already done
    #removing old or too contemporary data
    df = df[df[YEAR_LABEL].astype(int) >= year_start]
    df = df[df[YEAR_LABEL].astype(int) <= year_end]
    
    df = convert_country_df(df, ALTER_LABEL, standard_to_convert='STATE_en_UN', purge=True)
    df = convert_country_df(df, EGO_LABEL, standard_to_convert='STATE_en_UN', purge=True)
    
    countries_all = get_all_countries()
    df_triple = df.groupby([YEAR_LABEL, ALTER_LABEL, EGO_LABEL]).sum()
    df_triple = df_triple[df_triple[VALUE_LABEL]!=0]

    if test_data:
        test_df(df_triple.reset_index(), source_name, year_start=year_start, year_end=year_end, alter_label=ALTER_LABEL, ego_label=EGO_LABEL, year_label=YEAR_LABEL, value_label=VALUE_LABEL) 
    
    df_triple = pd.DataFrame(df_triple[VALUE_LABEL])
    
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
    
    logging.info("Done preprocessing peacekeep Data (UN SC)")
    return df, df_triple