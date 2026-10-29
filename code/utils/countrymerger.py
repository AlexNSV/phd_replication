#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Version 1.2
# Early Access

import pandas as pd
from urllib.error import URLError

MIXED_STANDARDS = {
    'en_mixed' : {
        'master' : 'STATE_en_UN',
        'slaves' : ['STATE_en_WorldBank', 'STATE_en_alternative', 'STATE_en_alternative_2','STATE_en_alternative_3','STATE_en_alternative_4', 'STATE_en_alternative_5', 'STATE_en_historic_1', 'STATE_en_historic_2', 'gdelt_state']
    }, 
    'ru_mixed' : {
        'master' : 'STATE_ru',
        'slaves' : ['STATE_ru_alt','STATE_ru_alt_2','STATE_ru_alt_3']
    }, 
    'iso_a3_mixed' : {
        'master' : 'Alpha3_Code',
        'slaves' : ['Alpha3_Code_historic_1', 'Alpha3_Code_historic_2']
    },
    'cow_mixed' : {
        'master' : 'COW_Country_Code',
        'slaves' : ['COW_Country_Code_historic_1']
    }
}

PATH_TO_KEY_DF_LOCAL = '../data_in/compatibility_un.csv'
PATH_TO_EXTRA_DF_LOCAL = '../data_in/compatibility_un - extra.csv'
URL_TO_KEY_DF = 'https://docs.google.com/spreadsheets/d/1ayIh8JK-Io3EvhwctDcIbwelEBoloyEY3Fz1nx0RTyg/export?gid=0&format=csv'
URL_TO_EXTRA_DF = 'https://docs.google.com/spreadsheets/d/1ayIh8JK-Io3EvhwctDcIbwelEBoloyEY3Fz1nx0RTyg/export?gid=211782544&format=csv'
key_columns = ['STATE_ru', 'STATE_ru_alt','STATE_en_UN', 'STATE_en_WorldBank', 
                   'STATE_en_alternative', 'STATE_en_alternative_2','STATE_en_alternative_3','STATE_en_alternative_4', 'STATE_en_historic_1',
                   'gdelt_state', 'ISO_Code', 'ISO_GIS', 
                   'COW_Country_Code','Alpha3_Code']

# Загружает таблицу со значениями разнообразных ключей из гугл таблиц. Если нет интернета, то загружает локально
def loadKeyDf(load_extra=False):
    if load_extra:
        try:
            df = pd.read_csv(URL_TO_KEY_DF, dtype = {'ISO_Code': str, 'ISO_GIS': str})
            extra = pd.read_csv(URL_TO_EXTRA_DF)
        except URLError: #Failed to download online, trying locally
            df = pd.read_csv(PATH_TO_KEY_DF, dtype = {'ISO_Code': str, 'ISO_GIS': str})
            extra = pd.read_csv(PATH_TO_EXTRA_DF)
        df = pd.concat([df, extra])
    else:
        try:
            df = pd.read_csv(URL_TO_KEY_DF, dtype = {'ISO_Code': str, 'ISO_GIS': str})
        except URLError: #Failed to download online, trying locally
            df = pd.read_csv(PATH_TO_KEY_DF, dtype = {'ISO_Code': str, 'ISO_GIS': str})
    return df

# Проверяет, соответствует ли название страны хотя бы одному из стандартов 
def checkCountry(countryname, key_df, return_none = 'wrong'):
    #print(key_df)
    
    #print(key_df.eq(countryname).any())
    
    if any(key_df.eq(countryname).any()):
        #print(f'Removing {countryname}')
        if return_none == 'wrong': return None
        if return_none == 'correct': return countryname
    else:
        if return_none == 'wrong': return countryname # возвращает None для неузнанных и название для правильных названий
        if return_none == 'correct': return None #возвращает названия для неузнанных и None для узнанных
        
# Удаляет страны, которые не входят в key_df
def removeExtraCountriesAndTerritories(df, column_name, inplace=False):
    if inplace:
        df_copy = df
    else:
        df_copy = df.copy()
    key_df = loadKeyDf()
    df_copy[column_name]=df_copy[column_name].apply(lambda x: checkCountry(x, key_df, return_none='wrong'))
    df_copy.dropna(subset = [column_name], inplace=True)
    if not inplace:
        return df_copy

# по dataframe и колонке пишет, каких стран нет в табличке с ключами
def validateCountries(df, column_name):
    key_df = loadKeyDf()
    validated = df[column_name].apply(lambda x: checkCountry(x, key_df, return_none='correct'))
    non_validated = validated.dropna().values
    return non_validated 

# Основаная функция: объединяет два датафрейма
def mergeData (
    master_df_original : pd.DataFrame, 
    slave_df_original : pd.DataFrame, 
    *, 
    master_key_col : str = '_index',
    slave_key_col = None,
    #slave_subset = None,
    return_standard : str = 'master',
    return_key_as_index = True,
    merge_on_year = False,
    master_year_col = None,
    slave_year_col = None
) -> pd.DataFrame:
    
    # Отдельная функция, чтобы проверить страну на соответствие смешанным форматам
    def compareWithMixedStandard(country, key_df, identification, passing, standard):
        for mixed_standard in MIXED_STANDARDS:
            if standard == MIXED_STANDARDS[mixed_standard]['master']:
                for standard_additional in MIXED_STANDARDS[mixed_standard]['slaves']:
                    print(f'{country} is compared with standard {standard_additional}')#, standard, )
                    #print(f'Result is {str(country) in key_df[standard_name_en].values.astype(str)}')
                    if country in key_df[standard_additional].values:
                        passing = True
                        break
                if passing: 
                    identification = mixed_standard
                else:
                    identification = 'unidentified'
        return passing, identification
    
    # Проверяет, соответствует ли series определенному стандарту
    def compareWithStandard(series_test, key_df, standard, skip_mixed=False):
        identification = 'unidentified'
        passing = True
        for country in series_test:
            print(f'{country} is compared with standard {standard}')#, standard, )
            #print(f'Result is {str(country) in key_df[standard].values.astype(str)}')
            if str(country) not in key_df[standard].values.astype(str):
                passing = False
                #English names in different forms
                if not skip_mixed:
                    passing, identification = compareWithMixedStandard(country, key_df, identification, passing, standard)
            if not passing: break
        if passing and identification not in MIXED_STANDARDS:
            identification = standard
        return identification
    
    # определяет стандарт series
    def identifyKey(series_test: pd.Series, key_columns, key_df, show_error=True):
        identification = 'unidentified'
        for standard in key_columns:
            identification_temp = compareWithStandard(series_test, key_df, standard)
            if identification_temp != 'unidentified':
                identification = identification_temp
            if identification != 'unidentified':
                print('identification is ', identification)
                if identification not in MIXED_STANDARDS:
                    return identification
        if show_error:
            assert identification != 'unidentified', f'Failed to match data with any known sets of country ids in:\n {series_test}'
        return identification
    
    # ищет перебором колонку с названиями стран
    def findCountryKey(df, key_columns, key_df, key_col):
        df_id = 'unidentified'
        df_id_col = 'unidentified'
        
        if (key_col is None) or (key_col == '_index'):
            # look for index
            column_id = identifyKey(pd.Series(df.index), key_columns, key_df, show_error=False)
            if column_id != 'unidentified':
                df_id = column_id
                df_id_col = '_index'
                return df_id, df_id_col
        
        # look for columns
        if not key_col is None:
            column_id = identifyKey(df[key_col], key_columns, key_df, show_error=True)
            if column_id != 'unidentified':
                df_id = column_id
                df_id_col = key_col
                return df_id, df_id_col
        else:
            for column in df:
                column_id = identifyKey(df[column], key_columns, key_df, show_error=False)
                if column_id != 'unidentified':
                    df_id = column_id
                    df_id_col = column
                    return df_id, df_id_col
        assert df_id != 'unidentified', 'Failed to match slave data with any known sets of country ids'
        
        return df_id, df_id_col
    
    #переводит колонку из одного чистого стандарта в другой
    def convertStandard(df: pd.DataFrame, id_col, convert_from, convert_to, key_df) -> pd.DataFrame:
        if convert_from == convert_to:
            return df
        converter_dict = key_df.set_index(convert_from)[convert_to].to_dict()
        if id_col != '_index':
            new_id_series = df[id_col].replace(converter_dict)
            df[id_col] = new_id_series
        else:
            df = df.rename(converter_dict)
        return df
    
    #переводит название одной страны из одного стандарта в другой
    def convertCountryname(countryname, convert_to, key_df):#, key_columns, key_df):
        #print('OLD')
        #print(countryname)
        new_countryname = key_df[key_df.eq(countryname).any(1)][convert_to].values[0]
        #print('NEW')
        #print(new_countryname)
        return new_countryname
#        df = pd.DataFrame([countryname], columns = ['column'])
 #       convert_from = identifyKey(df['column'], key_columns, key_df, 'column')
  #      new_countryname = convertStandard(df, 'column', convert_from, convert_to, key_df)
   #     return new_countryname
    
    # переводит смешанные стандарты в единый
    def convertMixedToSingleStandard(standard, df, column_key, key_df):
        assert standard in MIXED_STANDARDS, f'Unsupported mixed standard {standard}'
        single_standard_series = getColumnOrIndex(df, column_key)
        display(single_standard_series)
        single_standard_series = single_standard_series.apply(lambda x: convertCountryname(x, MIXED_STANDARDS[standard]['master'], key_df))
        print('AFTER')
        display(single_standard_series)
        return single_standard_series
    
    # получает колонку или индекс в виде series
    def getColumnOrIndex(df, columnname):
        if columnname == '_index':  # используется для работы с индексом как с остальными колонками
            return pd.Series(df.index)
        else:
            return df[columnname]
    # устанавливает значения колонки или индекса
    def setColumnOrIndex(df, new_series, columnname):
        if columnname == '_index':
            df['_index'] = new_series
            df.set_index('_index', inplace=True, drop=True)
            df.index.name = None
        else:
            df[columnname] = new_series
        return df
    # переводит инедекс или колонку в string
    def convertColumnOrIndex(df, columnname):
        if columnname == '_index':
            df.index = df.index.astype(str)
        else:
            df[columnname] = df[columnname].astype(str)
        return df
    
    # находит колонку с годами
    # нужен handler того, что таких колонок несколько
    def IdentifyYearKey(df):
        df = df.select_dtypes(include=['int32'])
        if df.shape[1] == 1:
            return df.columns[1]
        if df.shape[1] == 0:
            raise Exception("No year column identified in one of the dataframes")
        df_copy = df.mapapply(lambda x: (x > 1500) and (x < 2150)) # try to throw away unlikely number columns
        df_copy = df_copy.replace(False, None).dropna(axis=1)
        if df_copy.shape[1] == 1:
            return df.columns[1]
        if df_copy.shape[1] == 0:
            raise Exception("Failed to identify year in one of the dataframes, pass master_year_column and slave_year_column")
        df_copy = df.mapapply(lambda x: (x > 1914) and (x < 2030)) # try to throw away unlikely number columns
        df_copy = df_copy.replace(False, None).dropna(axis=1)
        if df_copy.shape[1] == 1:
            return df.columns[1]
        raise Exception("Failed to identify year in one of the dataframes, pass master_year_column and slave_year_column")
        

    #проверка стандарта возврата
    assert (return_standard in key_columns) or (return_standard in ['master', 'slave']), 'Unknown standard to return'
    
    # make copies of original dfs to avoid any unfortunate edit
    master_df = master_df_original.copy()
    slave_df = slave_df_original.copy()

    master_on_index = master_key_col == '_index'
    key_df = loadKeyDf()
    
    # Getting master key column and identifying stanrdard
    master_key_series = getColumnOrIndex(master_df, master_key_col)
    master_standard = identifyKey(master_key_series, key_columns, key_df)
    # Getting slave key column and identifying standard
    slave_standard, slave_key_col = findCountryKey(slave_df, key_columns, key_df, slave_key_col)
    
    # Applying single standard for mixed standards
    if slave_standard in MIXED_STANDARDS:
        slave_df[slave_key_col] = convertMixedToSingleStandard(slave_standard, slave_df, slave_key_col, key_df)
        slave_standard = MIXED_STANDARDS[slave_standard]['master']
    
    if master_standard in MIXED_STANDARDS:
        master_df[master_key_col] = convertMixedToSingleStandard(master_standard, master_df, master_key_col, key_df)
        master_standard = MIXED_STANDARDS[master_standard]['master']
    
    #getting to uniform standards
    #if return_standard == 'master':
    if master_standard != slave_standard:
        slave_df = convertStandard(slave_df, slave_key_col, slave_standard, master_standard, key_df)
    #elif return_standard == 'slave':
    #    if master_standard != slave_standard:
    #        master_df = convertStandard(master_df, master_key_col, master_standard, slave_standard, key_df)
    if return_standard not in ['master', 'slave']:
        if slave_standard != return_standard:
            slave_df = convertStandard(slave_df, slave_key_col, slave_standard, return_standard, key_df)
        if master_standard != return_standard:
            master_df = convertStandard(master_df, master_key_col, master_standard, return_standard, key_df)
   # print(master_standard, slave_standard, slave_key_col, master_key_col)
    slave_on_index = slave_key_col == '_index'
    # setting dataframes to single format
    master_df = convertColumnOrIndex(master_df, master_key_col)
    slave_df = convertColumnOrIndex(slave_df, slave_key_col)
    if slave_on_index:
        slave_key_col = None # меняем, чтобы правильно отрабатывали функции слияния
    if master_on_index:
        master_key_col = None
   # display(master_df)
   # display(slave_df)
    
    if not merge_on_year: # merging with to year information
        merged_df = master_df.merge(slave_df, left_on=master_key_col, right_on=slave_key_col,
                                left_index=master_on_index, right_index=slave_on_index, how='left',
                                suffixes=('_master', '_slave'))
    else: # merfing with years
        if master_year_col is None:
            master_year_col = IdentifyYearKey(master_df.astype('int32', errors='ignore'))
        if slave_year_col is None:
            slave_year_col = IdentifyYearKey(slave_df.astype('int32', errors='ignore'))
        if master_on_index:
            master_df.reset_index(drop=False, inplace=True)
        if slave_on_index:
            master_df.reset_index(drop=False, inplace=True)
        merged_df = master_df.merge(slave_df, left_on=[master_key_col, master_year_col], 
                                right_on=[slave_key_col, slave_year_col],
                                how='left', suffixes=('_master', '_slave'))
    # returning back to perform get operation
    if master_on_index:
        master_key_col = '_index'
    if slave_on_index:
        slave_key_col = '_index'
    if return_standard == 'master': # переводим к нужному стандарту аутпут
        key_column_output = getColumnOrIndex(master_df_original, master_key_col).reset_index(drop=True)
    elif return_standard == 'slave':
        key_column_output = getColumnOrIndex(slave_df_original, slave_key_col).reset_index(drop=True)
    if return_standard in ['master', 'slave']:
        merged_df = setColumnOrIndex(merged_df.reset_index(drop=True), key_column_output, master_key_col)
    #ставим в качестве индекса стандарт
    if return_key_as_index and master_key_col != '_index':
        merged_df.set_index(master_key_col, inplace=True)
        if return_standard == 'master':
            return_standard = master_standard
        elif return_standard == 'slave':
            return_standard = slave_standard
        merged_df.index.name = return_standard
    return merged_df
