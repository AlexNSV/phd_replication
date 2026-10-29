import pandas as pd
import numpy as np
import logging
from os import listdir
import csv

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df
from scipy.optimize import curve_fit


students_PATH = "../data/basic_preprocessed/students.csv"

def load_students():
    
    students_df = pd.read_csv(students_PATH, dtype = {'COUNTRY_ID':str, 'country_of_origin': str}, index_col = 'Unnamed: 0', low_memory=False)
    return students_df
    

def preprocess_students(df, year_start=1985, year_end=2022, rolling_window=5, normalize=True, test_data=True, interpolate=False, extrapolate=False):

    source_name = preprocess_students.__name__.split('_')[1]
    ALTER_LABEL = 'country_of_origin'
    EGO_LABEL = 'COUNTRY_ID'
    YEAR_LABEL = 'YEAR'
    VALUE_LABEL = 'VALUE'
    
    #def preprocessed_students(df, year_start, rolling_window=5):
    logging.info("Preprocessing students data")  # basic preprocessing already done

    df_china = df[df[EGO_LABEL].str.contains('CHN')].copy()
    df_china = df_china[df_china['INDICATOR_LABEL_EN']=='total inbound internationally mobile students, both sexes (number)']
    # df_china = df_china[df_china['INDICATOR_ID']==26637]
    
    df = convert_country_df(df, ALTER_LABEL, standard_to_convert='STATE_en_UN', purge=True)
    # второй не делаем, так как там Гонконг
    
    #  mask to get Hong-Kong data
    mask = (df[EGO_LABEL].str.contains('HKG', na=False)) & \
           (~df[ALTER_LABEL].str.contains('China', na=False))
    df_ = df[mask].dropna(subset=[ALTER_LABEL]).copy()  # .copy() для явного создания копии
    df = df.merge(df_, how='outer')
    
    df_['share'] = df_.groupby(YEAR_LABEL)[VALUE_LABEL].transform(lambda x: x / x.sum())
    
    df_[EGO_LABEL] = df_[EGO_LABEL].replace('HKG', 'CHN')
    
    logging.info("Extrapolating China data")

    #print("df_china columns:", df_china.columns.tolist())
    #print("df_china shape:", df_china.shape)
    china_dict = {int(v): int(i) for v, i in zip(df_china[YEAR_LABEL].values, df_china[VALUE_LABEL].values)}
    
    # Подготовка данных — только растущий период
    years_growth = [2006, 2007, 2008, 2009, 2010, 2011, 2012,
                             2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    values_growth = [china_dict[y] for y in years_growth]
    
    # Экспоненциальная модель: y = a * exp(b * (x - 2000))
    def exp_model(x, a, b):
        return a * np.exp(b * (x - 2000))
    
    # Подгонка параметров
    popt, _ = curve_fit(exp_model, years_growth, values_growth, p0=(1, 0.1))
    a_fit, b_fit = popt
    
    # Экстраполяция на 2001–2005
    years_extrap = np.arange(2001, 2006)
    values_extrap = []
    for i, year in zip(exp_model(years_extrap, a_fit, b_fit), years_extrap):
        china_dict[int(year)]= int(i)
    
    df_.apply(lambda x: int(china_dict[x[YEAR_LABEL]]*x['share']+x[VALUE_LABEL]), axis=1)
    df_[VALUE_LABEL] = df_.apply(lambda x: int(china_dict[x[YEAR_LABEL]]*x['share']+x[VALUE_LABEL]), axis=1)
    df_=df_.drop(['share'], axis=1)
    # df_.groupby('YEAR').sum()  # Test
    df = df[~df[EGO_LABEL].str.contains('CHN')] # removing China-China error
    df = df.merge(df_, how='outer')
    
    df = convert_country_df(df, EGO_LABEL, standard_to_convert='STATE_en_UN', purge=True)

    #removing old or too contemporary data
    df = df[df[YEAR_LABEL].astype(int) >= year_start]
    df = df[df[YEAR_LABEL].astype(int) <= year_end]

    countries_all = get_all_countries()
    df_triple = df.groupby([YEAR_LABEL, ALTER_LABEL, EGO_LABEL]).sum()
    df_triple = df_triple[df_triple[VALUE_LABEL]!=0]
    
    df_triple = pd.DataFrame(df_triple[VALUE_LABEL])

    if test_data:
        test_df(df_triple.reset_index(), source_name, year_start=year_start, year_end=year_end, alter_label=ALTER_LABEL, ego_label=EGO_LABEL, year_label=YEAR_LABEL, value_label=VALUE_LABEL) 
        
    # filling an empty year*country*country dataframe with zeroes
    empty_df=get_empty_country_df(years=df[YEAR_LABEL].unique(), countries_all=countries_all, names=[YEAR_LABEL, EGO_LABEL, ALTER_LABEL])
    
    # merging the df with zero df
    df_triple = empty_df.merge(df_triple,left_index=True, right_index=True, how='outer')#.fillna(0)

    df_triple = df_triple.reset_index()
    #df_[(df_['COUNTRY_ID']=='United States')]
    #return df, df_triple
    if interpolate:
        logging.info("Interpolating missing US data")
        us_mask = df_triple[EGO_LABEL]=='United States'
        df_us = df_triple[(us_mask)].copy()
        #return df, df_triple
        # Compute total VALUE per year (only US data)
        year_totals = df_us.groupby(YEAR_LABEL)[VALUE_LABEL].transform('sum')
        #return df, df_triple
        #year_totals = year_totals[df_us[YEAR_LABEL]!=1998]  # Seems like 1998 data is wrong
        #return df, df_triple
        # Create a boolean mask: True for years with total < 5000 AND individual value ≈ 0
        need_replace = ((year_totals < 5000) & (df_us[VALUE_LABEL].abs() < 1e-9)) | (df_us[YEAR_LABEL]==1998)
        
        # Replace those cells with None
        df_us.loc[need_replace, VALUE_LABEL] = None
        df_us[VALUE_LABEL]=df_us.groupby(ALTER_LABEL).ffill()[VALUE_LABEL]
        df_us[VALUE_LABEL]=df_us.groupby(ALTER_LABEL).bfill()[VALUE_LABEL]
        df_triple.loc[us_mask, VALUE_LABEL] = df_us[VALUE_LABEL]
    else:
        df_triple = df_triple.fillna(0)

    #return df, df_triple
    if test_data:
        logging.info("Testing after interpolating US data")
        test_df(df_triple.reset_index(), source_name, year_start=year_start, year_end=year_end, alter_label=ALTER_LABEL, ego_label=EGO_LABEL, year_label=YEAR_LABEL, value_label=VALUE_LABEL) 

    
    # counting rolling average
    df_triple = df_triple.reset_index().set_index(YEAR_LABEL).groupby([ALTER_LABEL, EGO_LABEL]).rolling(rolling_window, min_periods=1).mean()
    
    df_triple = df_triple.reset_index().set_index([YEAR_LABEL, ALTER_LABEL, EGO_LABEL])
    
    df_triple.index.names = ['year', 'alter', 'ego']
    df_triple = df_triple.rename({VALUE_LABEL:'value'}, axis=1)
    
    # Normalization
    if normalize:
        df_triple['value'] = df_triple['value'] / df_triple['value'].max()

    # Extrapolating old data
    if extrapolate:
        logging.info("Extrapolating missing years, using simple bfill")
        # Предположим, df - ваш исходный датафрейм с MultiIndex ['year', 'alter', 'ego']

        # 1. Получаем уникальные пары (alter, ego)
        unique_pairs = df_triple.index.droplevel('year').unique().to_frame(index=False)
        
        # 2. Создаем недостающие года (1985–1997)
        years_to_add = pd.DataFrame({'year': range(year_start, 1999)})
        
        # 3. Декартово произведение (все комбинации пар и годов)
        new_rows = pd.merge(unique_pairs, years_to_add, how='cross')
        new_rows = new_rows.set_index(['year', 'alter', 'ego'])
        # print(new_rows)
        
        # 4. Объединяем с исходными данными (используем concat, но аккуратно)
        df_combined = pd.concat([new_rows, df_triple]).sort_index(level=['alter', 'ego', 'year'])
        # print(df_combined)
        
        # 5. Применяем bfill только к группам, где есть пропуски (оптимизация)
        #mask = df_combined['value'].isna()  # Находим только пропуски
        df_combined.loc[:, 'value'] = (
            df_combined
            .groupby(['alter', 'ego'])['value']
            .bfill()
        )
        
        # 6. Сортируем по году (опционально)
        df_filled = df_combined.sort_index(level=['year', 'alter', 'ego']).drop('index',axis=1)
        # print(df_filled)
        
        # Результат:
        df_triple = df_filled.reset_index()
        df_triple = df_triple[df_triple['alter']!=df_triple['ego']]
        df_triple['year'] = df_triple['year'].astype(int)
        df_triple = df_triple.set_index(['year', 'alter', 'ego'])
        # print(df_triple)
        
        # to remove invalid country-years after extrapolation
        empty_df=get_empty_country_df(years=df_triple.reset_index()['year'].unique(), countries_all=countries_all, names=['year', 'ego', 'alter'])
        #return empty_df, empty_df
        df_triple = empty_df.merge(df_triple,left_index=True, right_index=True, how='left').fillna(0)
        # print(df_triple)
        
    if 'index' in df_triple.columns:
        df_triple = df_triple.drop('index', axis=1)
    
    # print(df_triple)
    logging.info("Done preprocessing students Data (COMstudents)")
    return df, df_triple