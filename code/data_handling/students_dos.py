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
    students_df = students_df.dropna(subset=['country_of_origin'])
    return students_df
    

def preprocess_students(df, year_start=1985, year_end=2022, rolling_window=5, normalize=True, test_data=True, interpolate=False):

    source_name = preprocess_students.__name__.split('_')[1]
    ALTER_LABEL = 'country_of_origin'
    EGO_LABEL = 'COUNTRY_ID'
    YEAR_LABEL = 'YEAR'
    VALUE_LABEL = 'VALUE'
    
    #def preprocessed_students(df, year_start, rolling_window=5):
    logging.info("Preprocessing students data")  # basic preprocessing already done

    df_china = df[df['COUNTRY_ID'].str.contains('CHN')].copy()
    df_china = df_china[df_china['INDICATOR_LABEL_EN']=='total inbound internationally mobile students, both sexes (number)']
    # df_china = df_china[df_china['INDICATOR_ID']==26637]
    
    df = convert_country_df(df, ALTER_LABEL, standard_to_convert='STATE_en_UN', purge=True)
    # второй не делаем, так как там Гонконг
    
    #  mask to get Hong-Kong data
    mask = (df['COUNTRY_ID'].str.contains('HKG', na=False)) & \
           (~df['country_of_origin'].str.contains('China', na=False))
    df_ = df[mask].dropna(subset=['country_of_origin']).copy()  # .copy() для явного создания копии
    df = df.merge(df_, how='outer')
    
    df_['share'] = df_.groupby('YEAR')['VALUE'].transform(lambda x: x / x.sum())
    
    df_['COUNTRY_ID'] = df_['COUNTRY_ID'].replace('HKG', 'CHN')
    logging.info("Extrapolating China data")
    china_dict = {int(v): int(i) for v, i in zip(df_china['YEAR'].values, df_china['VALUE'].values)}
    
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
    
    df_.apply(lambda x: int(china_dict[x['YEAR']]*x['share']+x['VALUE']), axis=1)
    df_['VALUE'] = df_.apply(lambda x: int(china_dict[x['YEAR']]*x['share']+x['VALUE']), axis=1)
    df_=df_.drop(['share'], axis=1)
    # df_.groupby('YEAR').sum()  # Test
    df = df[~df['COUNTRY_ID'].str.contains('CHN')] # removing China-China error
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
    df_triple = empty_df.merge(df_triple,left_index=True, right_index=True, how='outer').fillna(0)

    df_triple = df_triple.reset_index()
    #df_[(df_['COUNTRY_ID']=='United States')]
    us_mask = df_triple[EGO_LABEL]=='United States'
    df_us = df_triple[(us_mask)].copy()
    # Compute total VALUE per year (only US data)
    year_totals = df_us.groupby(YEAR_LABEL)[VALUE_LABEL].transform('sum')
    
    # Create a boolean mask: True for years with total < 50000 AND individual value ≈ 0
    need_replace = (year_totals < 5000) & (df_us[VALUE_LABEL].abs() < 1e-9)
    
    # Replace those cells with None
    df_us.loc[need_replace, VALUE_LABEL] = None
    df_us=df_us.groupby(ALTER_LABEL).ffill().bfill()
    df_triple.loc[us_mask, VALUE_LABEL] = df_us[VALUE_LABEL]
    
    # counting rolling average
    df_triple = df_triple.reset_index().set_index(YEAR_LABEL).groupby([ALTER_LABEL, EGO_LABEL]).rolling(rolling_window, min_periods=1).mean()
    
    df_triple = df_triple.reset_index().set_index([YEAR_LABEL, ALTER_LABEL, EGO_LABEL])
    
    df_triple.index.names = ['year', 'alter', 'ego']
    df_triple = df_triple.rename({VALUE_LABEL:'value'}, axis=1)
    
    # Normalization
    if normalize:
        df_triple['value'] = df_triple['value'] / df_triple['value'].max()
    
    logging.info("Done preprocessing students Data (COMstudents)")
    return df, df_triple