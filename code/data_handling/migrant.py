import pandas as pd
import numpy as np
import logging
from os import listdir

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


migrant_PATH = "../data/raw/migrant/undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx"

def load_migrant(year_start=1900, year_end=2050):
    
    migrant_df = pd.read_excel(migrant_PATH, sheet_name='Table 1', skiprows=10)
    migrant_df = migrant_df.set_index(['Location code of origin','Location code of destination']).stack().reset_index()
    migrant_df = migrant_df.rename(
        {'level_2':'year', 0:'migrants', 'Location code of origin':'origin', 'Location code of destination':'destination'}, 
        axis=1
    )
    migrant_df = migrant_df[migrant_df['year'].isin(list(range(year_start, year_end)))]
    return migrant_df
    
def preprocess_migrant(df, year_start=1985, year_end=2022, rolling_window=5, normalize=True, test_data=True, interpolate=True, logarithmic=False):

    source_name = preprocess_migrant.__name__.split('_')[1]
    EGO_LABEL = 'destination'
    ALTER_LABEL = 'origin'
    YEAR_LABEL = 'year'
    VALUE_LABEL = 'migrants'
    
    logging.info("Preprocessing migrant data")  # basic preprocessing already done
    #removing old or too contemporary data
    df = df[df[YEAR_LABEL].astype(int) >= year_start]
    df = df[df[YEAR_LABEL].astype(int) <= year_end]
    
    df = convert_country_df(df, ALTER_LABEL, standard_to_convert='STATE_en_UN', purge=True, numeric_type='iso')
    df = convert_country_df(df, EGO_LABEL, standard_to_convert='STATE_en_UN', purge=True, numeric_type='iso')
        
    countries_all = get_all_countries()
    df_triple = df.groupby([YEAR_LABEL, ALTER_LABEL, EGO_LABEL]).sum()
    df_triple = df_triple[df_triple[VALUE_LABEL]!=0]
    
    df_triple = pd.DataFrame(df_triple[VALUE_LABEL])

    if test_data:
        test_df(df_triple.reset_index(), source_name, year_start=year_start, year_end=year_end, alter_label=ALTER_LABEL, ego_label=EGO_LABEL, year_label=YEAR_LABEL, value_label=VALUE_LABEL)

    if logarithmic:
        logging.info(f"Converting {source_name} to log 10 scale")
        df_triple[VALUE_LABEL] = np.log10(np.asarray(df_triple[VALUE_LABEL], dtype=float))
        
    # filling an empty year*country*country dataframe with zeroes
    empty_df=get_empty_country_df(years=df[YEAR_LABEL].unique(), countries_all=countries_all, names=[YEAR_LABEL, EGO_LABEL, ALTER_LABEL])
    
    # merging the df with zero df
    df_triple = empty_df.merge(df_triple,left_index=True, right_index=True, how='outer').fillna(0)
    
    # counting rolling average
    df_triple = df_triple.reset_index().set_index(YEAR_LABEL).groupby([ALTER_LABEL, EGO_LABEL]).rolling(rolling_window, min_periods=1).mean()
    
    df_triple = df_triple.reset_index().set_index([YEAR_LABEL, ALTER_LABEL, EGO_LABEL])
    
    df_triple.index.names = ['year', 'alter', 'ego']
    df_triple = df_triple.rename({VALUE_LABEL:'value'}, axis=1)
    
    
    logging.info("Done preprocessing migrant Data (UNHCR)")
    logging.info("Imputing missing years")
    # Imputing data for each dyad
    if interpolate:
        logging.info('Interpolation started')
        df_processed = []
        years = pd.DataFrame(data=list(range(year_start, year_end+1)))
        years.columns = ['year']
        t = df_triple.reset_index()
        for country_a, country_b in list(df_triple.reset_index()[['alter', 'ego']].drop_duplicates().values):
            #years['ego'] = country_a
            #years['alter'] = country_b
            t_n = t[(t['alter']==country_b)&(t['ego']==country_a)]
            t_n = t_n.merge(years, how='outer').sort_values('year')
            t_n['ego'] = country_a
            t_n['alter'] = country_b
            #logging.debug(f"{country_a}, {country_b}, {list(t_n['value'])}")
            t_n['value']=t_n['value'].interpolate(method='linear', limit_direction='both')
            df_processed += t_n,
            #if 'Russian Federation' == country_a:
            #    print(country_a, country_b)
            #    print(t_n.dropna().head())
            #    print(df_processed[-1].tail())
        df_triple=pd.concat(df_processed).set_index(['year', 'ego', 'alter'])
        print(df_triple)
        logging.info("Saving imputed data")
    # to remove 1990 (pre-independence) Croatia etc. Since UNDESA has data for them for some reason
    empty_df=get_empty_country_df(years=list(range(year_start, year_end+1)), countries_all=countries_all, names=['year', 'ego', 'alter'])
    print(empty_df)
    df_triple = empty_df.merge(df_triple,left_index=True, right_index=True, how='left').fillna(0)
    # Normalization
    if normalize:
        df_triple['value'] = df_triple['value'] / df_triple['value'].max()
    df_triple.to_csv("../data/preprocessed/migrant.csv")
    return df, df_triple