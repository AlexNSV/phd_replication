import pandas as pd
import numpy as np
import logging
from os import listdir

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


tourism_PATH = "../data/raw/tourism/unwto-all-data-download_2022.xlsx"

def load_tourism(year_start=1995, year_end=2022):
    #  Data in thousands
    if (year_start < 1995) | (year_end > 2022):
        print(f'Invalid years: {year_start}, {year_end}, only support 1995-2022')
    tourism_df_in = pd.read_excel(tourism_PATH, sheet_name='Inbound Tourism-Regions', skiprows=2, converters={'C.':str})
    tourism_df_in = tourism_df_in.set_index(["C."]).rename({'Unnamed: 5':'Total'}, axis=1).loc[:,['Total', 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]]
    tourism_df_in=tourism_df_in.dropna(subset='Total').drop('Total', axis=1)
    tourism_df_in=tourism_df_in.stack().reset_index().rename({'C.':'ego', 'level_1':'year', 0:'value'}, axis=1)
    tourism_df_in=convert_country_df(tourism_df_in, 'ego', standard_to_convert='STATE_en_UN', purge=True)
    tourism_df_in=tourism_df_in.replace('..', None)

    #tourism_df_out = pd.read_excel(tourism_PATH, sheet_name='Outbound Tourism-Departures', skiprows=2, converters={'C.':str})
    #tourism_df_out = tourism_df_out.set_index(["C."]).rename({'Unnamed: 5':'Total'}, axis=1).loc[:,['Total', 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]]
    #tourism_df_out=tourism_df_out.dropna(subset='Total').drop('Total', axis=1)
    #tourism_df_out=tourism_df_out.stack().reset_index().rename({'C.':'ego', 'level_1':'year', 0:'value'}, axis=1)
    #tourism_df_out=convert_country_df(tourism_df_out, 'ego', standard_to_convert='STATE_en_UN', purge=True)
    #tourism_df_out=tourism_df_out.replace('..', None)
    #tourism_df = tourism_df[tourism_df['year'].isin(list(range(year_start, year_end)))]
    #tourism_df['rufugees and asylum seekers'] = tourism_df["tourisms under UNHCR's mandate"]+tourism_df["Asylum-seekers"]
    #tourism_df['cmdCode']=tourism_df['cmdCode'].replace({'TOTAL':np.nan})
    #tourism_df=tourism_df.dropna(subset='cmdCode')
    #tourism_df=tourism_df.groupby(['refYear', 'reporterISO', 'partnerISO']).sum().reset_index()
    return tourism_df_in#, tourism_df_out

def interpolate_tourism(df, human_intermediate, year_start, year_end, test_data=True):

    source_name = interpolate_tourism.__name__.split('_')[1]

    # normalization
    df['value'] = df['value'] / df['value'].max()
    
    tourism_inferred = human_intermediate.reset_index().merge(df, on = ['ego', 'year'], suffixes=['_human', '_tourism'])
    tourism_inferred = tourism_inferred.dropna(subset=['value_tourism'])
    #tourism_inferred = tourism_inferred.fillna(subset=['value_migrant'])
    tourism_inferred['tourism_new'] = tourism_inferred['value_tourism'] * tourism_inferred['value_human']
    tourism_inferred = tourism_inferred[tourism_inferred['ego']!=tourism_inferred['alter']]
    
    tourism_inferred=tourism_inferred.set_index(['ego', 'alter', 'year']).rename({'tourism_new':'value'}, axis=1)
    
    tourism_inferred=pd.DataFrame(tourism_inferred['value'])

    if test_data:
        test_df(tourism_inferred.reset_index(), source_name, year_start=year_start, year_end=year_end)
    return df, tourism_inferred

"""
Попытка
tourism_df_out.dropna(inplace=True)
for year in tourism_df_out['year'].unique():
    print(year)
    df_year = tourism_df_out[tourism_df_out['year']==year]
    df_year['value'] = (df_year['value'] / df_year['value'].max())
    print(df_year)
"""
"""
def preprocess_tourism(df, year_start=1985, year_end=2022, rolling_window=5, normalize=True):

    EGO_LABEL = 'C.'
    ALTER_LABEL = 'Location code of origin'
    YEAR_LABEL = 'year'
    VALUE_LABEL = 'tourisms'
    
    logging.info("Preprocessing tourism data")  # basic preprocessing already done
    #removing old or too contemporary data
    df = df[df[YEAR_LABEL].astype(int) >= year_start]
    df = df[df[YEAR_LABEL].astype(int) <= year_end]
    
    df = convert_country_df(df, ALTER_LABEL, standard_to_convert='STATE_en_UN', purge=True)
    df = convert_country_df(df, EGO_LABEL, standard_to_convert='STATE_en_UN', purge=True)
    
    countries_all = get_all_countries()
    df_triple = df.groupby([YEAR_LABEL, ALTER_LABEL, EGO_LABEL]).sum()
    df_triple = df_triple[df_triple[VALUE_LABEL]!=0]
    
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
    
    logging.info("Done preprocessing tourism Data (UNHCR)")
    return df, df_triple
"""