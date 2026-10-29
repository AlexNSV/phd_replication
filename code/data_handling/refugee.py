import pandas as pd
import numpy as np
import logging
from os import listdir

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


refugee_PATH = "../data/raw/refugee/data.csv"

def load_refugee():
    
    refugee_df = pd.read_csv(refugee_PATH, skiprows=14)
    refugee_df['rufugees'] = refugee_df["Refugees under UNHCR's mandate"]+refugee_df["Asylum-seekers"]
    #refugee_df['cmdCode']=refugee_df['cmdCode'].replace({'TOTAL':np.nan})
    #refugee_df=refugee_df.dropna(subset='cmdCode')
    #refugee_df=refugee_df.groupby(['refYear', 'reporterISO', 'partnerISO']).sum().reset_index()
    return refugee_df
    
def preprocess_refugee(df, year_start=1985, year_end=2022, rolling_window=5, normalize=True, test_data=True, logarithmic=False):

    source_name = preprocess_refugee.__name__.split('_')[1]
    EGO_LABEL = 'Country_of_asylum'
    ALTER_LABEL = 'Country_of_origin'
    YEAR_LABEL = 'Year'
    VALUE_LABEL = 'rufugees'
    
    logging.info("Preprocessing refugee data")  # basic preprocessing already done
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

    if logarithmic:
        logging.info(f"Converting {source_name} to log 10 scale")
        #  return df_triple[VALUE_LABEL], np.log10(df_triple[VALUE_LABEL])
        df_triple[VALUE_LABEL] = np.log10(df_triple[VALUE_LABEL])
    #  return df_triple # TESTING ONLY~!
    
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
    
    logging.info("Done preprocessing refugee Data (UNHCR)")
    return df, df_triple