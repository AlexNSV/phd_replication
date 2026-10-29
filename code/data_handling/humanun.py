import pandas as pd
import numpy as np
import logging
from os import listdir

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


humanun_PATH = "../data/raw/unhumanrights/humanun.csv"

def load_humanun():
    #  Data in thousands
    humanun_df_in = pd.read_csv(humanun_PATH, index_col = 'Unnamed: 0')
    #humanun_df_in = humanun_df_in[(humanun_df_in['year'] <= year_end)&(humanun_df_in['year'] >= year_start)]
    humanun_df_in = humanun_df_in.dropna(subset=['sponsors_new', 'affected'])
    humanun_df_in['value'] = 1
    return humanun_df_in
    

def preprocess_humanun(df, year_start, year_end, test_data=True, normalize=True, rolling_window=5, extrapolate=False):

    source_name = preprocess_humanun.__name__.split('_')[1]
    EGO_LABEL = 'sponsors_new'
    ALTER_LABEL = 'affected'
    YEAR_LABEL = 'year'
    VALUE_LABEL = 'value'
    
    logging.info("Preprocessing humanun data")  # basic preprocessing already done
    #removing old or too contemporary data

    df[YEAR_LABEL] = df[YEAR_LABEL].astype(int)
    df = df[df[YEAR_LABEL] >= year_start]
    df = df[df[YEAR_LABEL] <= year_end]
    
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

    if extrapolate:
        # Предположим, df - ваш исходный датафрейм с MultiIndex ['year', 'alter', 'ego']

        # 1. Получаем уникальные пары (alter, ego)
        unique_pairs = df_triple.index.droplevel('year').unique().to_frame(index=False)
        
        # 2. Создаем недостающие года (1985–1991)
        years_to_add = pd.DataFrame({'year': range(year_start, 1992)})
        
        # 3. Декартово произведение (все комбинации пар и годов)
        new_rows = pd.merge(unique_pairs, years_to_add, how='cross')
        new_rows = new_rows.set_index(['year', 'alter', 'ego'])
        
        # 4. Объединяем с исходными данными (используем concat, но аккуратно)
        df_combined = pd.concat([new_rows, df_triple]).sort_index(level=['alter', 'ego', 'year'])
        
        # 5. Применяем bfill только к группам, где есть пропуски (оптимизация)
        #mask = df_combined['value'].isna()  # Находим только пропуски
        df_combined.loc[:, 'value'] = (
            df_combined
            .groupby(['alter', 'ego'])['value']
            .bfill()
        )
        
        # 6. Сортируем по году (опционально)
        df_filled = df_combined.sort_index(level=['year', 'alter', 'ego'])
        
        # Результат:
        df_triple = df_filled.reset_index()
        df_triple = df_triple[df_triple['alter']!=df_triple['ego']]
        df_triple['year'] = df_triple['year'].astype(int)
        df_triple = df_triple.set_index(['year', 'alter', 'ego'])
    else:
        df_triple['year'] = df_triple['year'].astype(int)
    # to remove invalid country-years after extrapolation
    empty_df=get_empty_country_df(years=df_triple.reset_index()['year'].unique(), countries_all=countries_all, names=['year', 'ego', 'alter'])
    df_triple = empty_df.merge(df_triple,left_index=True, right_index=True, how='left').fillna(0)
    
    logging.info("Done preprocessing humanun Data (UNHCR)")
    return df, df_triple