import pandas as pd
import numpy as np
import logging
from os import listdir

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from utils.countryconverter import convert_country_df


ODA_OECD_PATH = "../data/raw/oda/OECD.DCD.FSD,DSD_DAC2@DF_DAC2A,1.3+all.csv"
ODA_CHINA_PATH = "../data/raw/oda/china/AidDatas_Global_Chinese_Development_Finance_Dataset_Version_3_0/AidDatas_Global_Chinese_Development_Finance_Dataset_Version_3_0/AidDatasGlobalChineseDevelopmentFinanceDataset_v3.0.xlsx"
ODA_RUSSIACHINA_PATH = "../data/raw/oda/oda_ru_china.xlsx"
ODA_INDIA_PATH = "../data/raw/oda/india/ind_aid_global_country_releaseV1.xlsx"
CIA_RENAME_DICT = {'YEAR':'year', 'DONOR':'ego', 'Recipient':'alter', 'Value (millions of dollars)':'value'}
AIDDATA_RENAME_DICT = {'Financier Country':'ego', 'Recipient':'alter', 'Commitment Year':'year','Adjusted Amount (Constant USD 2021)':'value'}
OECD_RENAME_DICT = {'Donor':'ego', 'Recipient':'alter', 'TIME_PERIOD':'year','OBS_VALUE':'value'}
INDIA_RENAME_DICT = {'recipientname':'alter', 'usd_disbursment_con':'value'}


RUSSIA_MULTILATERAL_BREAKDOWN = [  # According to https://hub.unido.org/bilateral-development-partners/russian-federation
    'International Development Association [IDA]',
    'World Bank',
    'UNDP', 
    'UNHCR', 
    'UNICEF',
    'WFP',
    'International Bank for Reconstruction and Development [IBRD]',
    'World Health Organisation [WHO]',
    'Food and Agriculture Organisation [FAO]',
    'United Nations Industrial Development Organization [UNIDO]'
]

def load_oda(add_multilateral=False):
    oecd_oda = load_oecd_oda(add_multilateral=add_multilateral)
    if add_multilateral:
        logging.warning("Only supporting multilateral for oecd data")
    china_oda = load_china_oda()
    russia_oda = load_russia_oda()
    india_oda = load_india_oda()
    oda_df = pd.concat([oecd_oda, china_oda, russia_oda, india_oda])
    return oda_df


def load_oecd_oda(add_multilateral=True):
    logging.info("Reading OECD ODA Data (DAC2)")
    oecd_oda = pd.read_csv(ODA_OECD_PATH)
    oecd_oda = oecd_oda[oecd_oda['PRICE_BASE']=='Q']
    if add_multilateral:
        oecd_oda = oecd_oda[(oecd_oda['Measure']=='Imputed multilateral ODA')|(oecd_oda['Measure']=='Official development assistance (ODA), disbursements')]
    else:
        oecd_oda = oecd_oda[oecd_oda['Measure']=='Official development assistance (ODA), disbursements']
    oecd_oda = pd.DataFrame(oecd_oda.groupby(['TIME_PERIOD','Donor','Recipient'])['OBS_VALUE'].sum(numeric_only=True))
    def replace_zero(x):
        if x >= 0:
            return x
        return 0
    oecd_oda['OBS_VALUE'] = oecd_oda['OBS_VALUE'].apply(replace_zero)
    oecd_oda = oecd_oda.reset_index().rename(OECD_RENAME_DICT, axis=1)
    return oecd_oda

def load_china_oda():
    logging.info("Reading China ODA-like Data (Aiddata)")
    china_oda = pd.read_excel(ODA_CHINA_PATH, sheet_name='GCDF_3.0')
    china_oda = china_oda[china_oda['Flow Class']=='ODA-like']
    china_oda = china_oda[~china_oda['Status'].isin(['Cancelled', 'Suspended'])]
    china_oda = pd.DataFrame(china_oda.groupby(['Financier Country', 'Recipient', 'Commitment Year'])['Adjusted Amount (Constant USD 2021)'].sum())
    china_oda['Adjusted Amount (Constant USD 2021)'] = china_oda['Adjusted Amount (Constant USD 2021)'] / 1000000
    china_oda = china_oda.reset_index().rename(AIDDATA_RENAME_DICT, axis=1)

    logging.info("Reading China historic foreign aid data (CIA)")
    china_oda_cia = pd.read_excel(ODA_RUSSIACHINA_PATH, sheet_name='China') # 1085,1986,1988,1989 data
    china_oda_cia = china_oda_cia.reset_index().rename(CIA_RENAME_DICT, axis=1)

    china_oda_total = pd.concat([china_oda, china_oda_cia])
    
    return china_oda_total

def load_russia_oda():  # No multilateral
    logging.info("Reading Russia ODA data")
    russia_oda = pd.read_excel(ODA_RUSSIACHINA_PATH, sheet_name='Russia')
    russia_oda = russia_oda[(russia_oda['Recipient']!='World')&(russia_oda['Type']!='Multilateral Aid')&(russia_oda['Type']!='Bilateral Aid')]
    russia_oda.set_index(['YEAR','DONOR'], inplace=True)
    russia_oda = russia_oda.reset_index().rename(CIA_RENAME_DICT, axis=1)
    return russia_oda

def load_india_oda():
    logging.info("Reading India ODA data from India development finance dataset")
    india_oda = pd.read_excel(ODA_INDIA_PATH)
    india_oda.rename(INDIA_RENAME_DICT, axis=1, inplace=True)
    india_oda['value'] = india_oda['value'] / 1000000
    india_oda['ego'] = 'India'
    india_oda = pd.DataFrame(india_oda.groupby(['ego', 'alter', 'year'])['value'].sum()).reset_index()
    return india_oda

def preprocess_oda(df, year_start, year_end, test_data=True, normalize=True, rolling_window=10):

    source_name = preprocess_oda.__name__.split('_')[1]
    EGO_LABEL = 'ego'
    ALTER_LABEL = 'alter'
    YEAR_LABEL = 'year'
    VALUE_LABEL = 'value'
    
    logging.info("Preprocessing oda data")  # basic preprocessing already done

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
    
    logging.info("Done preprocessing ODA Data")
    return df, df_triple
    