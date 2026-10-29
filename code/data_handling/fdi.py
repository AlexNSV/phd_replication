import pandas as pd
import logging


import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df, get_percent_df
from utils.countryconverter import convert_country_df


FDI_PATH = "../data/basic_preprocessed/fdi_total.csv"
HISTORIC_FDI_PATH = '../data/raw/fdi/unctad/US_FdiFlowsStock.csv'

REPLACEMNENTS_HISTORIC = {
    "Indonesia (...2002)":"Indonesia",
    "Ethiopia (...1991)":"Ethiopia"
}

def load_fdi(year_start=1985):
    """Loads FDI data"""
    df = pd.read_csv(FDI_PATH, index_col='Unnamed: 0')
    df = df[df['year'] >= year_start]
    logging.info("Loaded FDI data")
    return df

def load_historic_fdi(rename=True, year_start=1985):
    EGO_LABEL = 'Economy Label'
    YEAR_LABEL = 'Year'
    VALUE_LABEL = 'US$ at current prices in millions'
    
    fdi_df_historic = pd.read_csv(HISTORIC_FDI_PATH)
    fdi_df_historic = fdi_df_historic[(fdi_df_historic['Flow Label']=='Stock')&(fdi_df_historic['Direction Label']=='Outward')]
    new_y_rows = []
    for year in range(year_start, fdi_df_historic[YEAR_LABEL].min()):
        for country in fdi_df_historic[EGO_LABEL].unique():
            row = {
                YEAR_LABEL:year,
                EGO_LABEL:country,
                'Flow Label':'Stock',
                'Direction Label':'Outward',
                VALUE_LABEL:None
            }
            new_y_rows += row,
            
    # Добавляем новые строки в исходный DataFrame
    fdi_df_historic_extended = pd.concat([pd.DataFrame(new_y_rows), fdi_df_historic], ignore_index=True)
    fdi_df_historic_extended = fdi_df_historic_extended.sort_values([EGO_LABEL, YEAR_LABEL])

    fdi_df_historic_extended[EGO_LABEL].replace(REPLACEMNENTS_HISTORIC, inplace=True)
    fdi_df_historic_extended = convert_country_df(fdi_df_historic_extended, EGO_LABEL, standard_to_convert='STATE_en_UN', purge=True)

    # Заполняем пропуски последним известным годом
    fdi_df_historic_extended[VALUE_LABEL] = fdi_df_historic_extended.groupby(EGO_LABEL)[VALUE_LABEL].transform(
        lambda x: x.bfill()
    )
    
    if rename:
        fdi_df_historic_extended.rename({EGO_LABEL: 'ego', YEAR_LABEL:'year', VALUE_LABEL:'value'}, inplace=True, axis=1)
        fdi_df_historic_extended.set_index(['ego', 'year'], inplace=True)
    return fdi_df_historic_extended

def preprocessed_fdi(df, year_start, year_end=2022, rolling_window=5, test_data=True, extrapolate=True):
    logging.info("Preprocessing fdi Depoyment Data")  # basic preprocessing already done

    source_name = preprocessed_fdi.__name__.split('_')[1]
    
    EGO_LABEL = 'Country'
    ALTER_LABEL = 'Partner Country'
    YEAR_LABEL = 'year'
    VALUE_LABEL = 'value'
    #removing old data
    df = df[df['year'] >= year_start]
    
    df = convert_country_df(df, 'Country', standard_to_convert='STATE_en_UN', purge=True)
    df = convert_country_df(df, 'Partner Country', standard_to_convert='STATE_en_UN', purge=True)
    
    countries_all = get_all_countries()
    df_triple = df.groupby(["year", "Country", "Partner Country"]).sum()
    df_triple = df_triple[df_triple['value']!=0]
    
    df_triple = pd.DataFrame(df_triple['value'])
    #  Replacing negative stock FDIs with zero
    #  Note from IMF:
    #  Direct investment positions also could be negative due to net negative positions with fellows and/or due to negative retained earnings
    #      (which may result from the accumulation of negative reinvested earnings).
    df_triple['value']=df_triple['value'].apply(lambda x: x if x >= 0.0 else 0)
    
    if test_data:
        test_df(df_triple.reset_index(), source_name, year_start=year_start, year_end=year_end, alter_label=ALTER_LABEL, ego_label=EGO_LABEL, year_label=YEAR_LABEL, value_label=VALUE_LABEL)    
    # filling an empty year*country*country dataframe with zeroes
    empty_df=get_empty_country_df(years=df['year'].unique(), countries_all=countries_all, names=["year", "Country", "Partner Country"])
    
    # merging the df with zero df
    df_triple = empty_df.merge(df_triple,left_index=True, right_index=True, how='outer').fillna(0)

    df_triple = df_triple.reset_index().drop_duplicates(subset=['year', 'Partner Country', 'Country'])  # lazy fix to duplication problem
    df_triple = df_triple.reset_index().set_index(['year', 'Partner Country', 'Country'])
    df_triple.index.names = ['year', 'alter', 'ego']

    
    
    if extrapolate:
        logging.info("FDI data extrapolation started")
        fdi_df_historic_extended = load_historic_fdi(year_start=year_start)

        def impute_if_none(x):
            if pd.notna(x['value']):
                return x['value']
            return x['value_ego_total'] * x['percent']
        
        percent_df=get_percent_df(df_triple)
        empty_df=get_empty_country_df(years=range(year_start, year_end+1), countries_all=countries_all, names=['year', 'ego', 'alter'])
        empty_df['value'] = df_triple['value']
        empty_df = empty_df.reset_index().set_index(['ego', 'alter'])
        empty_df['percent'] = percent_df['percent']
        empty_df = empty_df.reset_index().set_index(['ego', 'year'])
        empty_df = empty_df.merge(fdi_df_historic_extended[['value']].reset_index(), on=['ego','year'], how='left', suffixes=('', '_ego_total'))
        empty_df['value']=empty_df.apply(impute_if_none,axis=1)
        df_triple = empty_df.reset_index().set_index(['ego', 'alter', 'year'])[['value']]
    
    df_triple = df_triple.reset_index().drop_duplicates(subset=['ego', 'alter', 'year'])
    # counting rolling average
    df_triple = df_triple.reset_index().set_index('year').groupby(['alter', 'ego']).rolling(rolling_window, min_periods=1).mean()

    df_triple = df_triple.reset_index().set_index(['year', 'alter', 'ego'])
    
    # Normalization
    df_triple['value'] = df_triple['value'] / df_triple['value'].max()
    
    logging.info("Done preprocessing FDI Data")
    return df, df_triple