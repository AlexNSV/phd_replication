import sipri # pip install sipri
import pandas as pd
import logging
from itertools import product, combinations

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, get_system_members, test_df, GREAT_POWERS
from analysis.network_analysis import get_networks
from analysis.community import detect_local_communities
from analysis.hegemony import get_hegemony_scores, get_hegemony_top, visualize_hegemony
from utils.countryconverter import convert_country_df
from data_handling import gsheet_handler


SIPRI_PATH="../data/raw/arms/sipri_arms_transfer_dyad_backup.csv"

def load_sipri():
    """Loads SIPRI Arms transfer data (last: 2022)"""
    #data = sipri.sipri_data(low_year='1985',high_year='2022',seller='',buyer='',armanent_category='any',buyers_or_sellers='',filetype='csv',include_open_deals='on',sum_deliveries='on')
    #df = pd.read_csv(StringIO(data),keep_default_na=False,na_values=['None'])
    #df.to_csv("sipri_arms_transfer_dyad.csv")
    df = pd.read_csv(SIPRI_PATH, index_col = 'Unnamed: 0')
    logging.info("Loaded SIPRI Arms Transfer Data")
    return df


def preprocess_sipri(df, rolling_window=5, win_type='hann', year_start=1992, year_end = 2022, country_df=None, test_data=True):
    """Preprocessing SIPRI Arms Transfer Data:
        - removing rebel groups and international organizations
        - removing unknown recipients and suppliers
        - converting to df in triple format (year*country*country)

    Parameters
    ------------
        df: pd.DataFrame()
            DataFrame with raw unprocessed SIPRI Arms Transfer Data
        rolling_window: int
            Rolling window to use for smoothing the data
        year_start: int
            Year to start with (needed for correct rolling calculations)
    Return
    -----------
        tuple(df , triple_df)
            df : pd.DataFramme() - preprocessed df with SIPRI Arms Transfer Data,
            triple_df  : pd.DataFramme() - DataFrame containing information about Arms Transfer in triple format (year*country*country). 
    """
    source_name = preprocess_sipri.__name__.split('_')[1]
    
    EGO_LABEL = 'seller'
    ALTER_LABEL = 'buyer'
    YEAR_LABEL = 'odat'
    VALUE_LABEL = 'tivorder'
    
    if country_df is None: country_df = gsheet_handler.read_gsheet(tablename='country_data', sheetname='countryids', skiprows=0)['state_en_un'].dropna()
    
    logging.info("Preprocessing SIPRI Arms Transfer Data")
    df.drop(df.shape[0]-1, inplace=True)

    # removing rebel groups and IOs
    df = df[~df[ALTER_LABEL].str.contains('\*')]
    # removing unknown recipients and suppliers
    df = df[~df[ALTER_LABEL].str.contains('unknown')]
    df = df[~df[EGO_LABEL].str.contains('\*')]
    df = df[~df[EGO_LABEL].str.contains('unknown')]

    df[VALUE_LABEL] = df[VALUE_LABEL].astype(float)
    
    #removing old data
    df = df[df[YEAR_LABEL] >= year_start]

    #converting to STATE_en_UN (can do Alpha3_Code)
    df = convert_country_df(df, ALTER_LABEL, standard_to_convert='STATE_en_UN', purge=True, print_convertions=False)
    df = convert_country_df(df, EGO_LABEL, standard_to_convert='STATE_en_UN', purge=True, print_convertions=False)

    df_triple = df.groupby([YEAR_LABEL, ALTER_LABEL, EGO_LABEL]).sum()

    if test_data:
        test_df(df_triple.reset_index(), source_name, year_start=year_start, year_end=year_end, alter_label=ALTER_LABEL, ego_label=EGO_LABEL, year_label=YEAR_LABEL, value_label=VALUE_LABEL)
        
    
    countries_all = set(country_df)
        
    df_triple = df_triple[df_triple[VALUE_LABEL]!=0]
    
    df_triple = pd.DataFrame(df_triple[VALUE_LABEL])
    
  
    # filling an empty year*country*country dataframe with zeroes
    #options = product(df['odat'].unique(),countries_all, countries_all)
    #index = pd.MultiIndex.from_tuples(options, names=["odat", "seller", "buyer"])
    #empty_df = pd.DataFrame(index=index)
    empty_df=get_empty_country_df(years=df[YEAR_LABEL].unique(), countries_all=countries_all, names=[YEAR_LABEL, EGO_LABEL, ALTER_LABEL])

    # merging the df with zero df
    df_triple = empty_df.merge(df_triple,left_index=True, right_index=True, how='outer').fillna(0)
    
    # counting rolling average
    df_triple = df_triple.reset_index().set_index(YEAR_LABEL).groupby([ALTER_LABEL, EGO_LABEL]).rolling(rolling_window, min_periods=1, win_type=win_type).mean()
    
    df_triple = df_triple.reset_index().set_index([YEAR_LABEL, ALTER_LABEL, EGO_LABEL])
    
    df_triple.index.names = ['year', 'alter', 'ego']
    df_triple.rename(columns={VALUE_LABEL:'value'}, inplace=True)
    
    # Normalization
    df_triple['value'] = df_triple['value'] / df_triple['value'].max()
    
    logging.info("Done preprocessing SIPRI Arms Transfer Data")
    return df, df_triple


    
def sipri_main(year_start=1992, rolling_window=5, res_range_start=2, res_range_end=20, one_year_hegemony_threshold=5, min_clients_for_top=3, centrality_threshold=0.5, community_detection='louvian'):
    comm_name = 'weapon_trade'
    
    df = load_sipri()  # Loading the data
    df, df_triple = preprocess_sipri(df, rolling_window=rolling_window, year_start=year_start)  # data preprocessing
    countries_all = get_all_countries(processed_df=df)  # getting set of all countries
    year_end = df_triple.index.get_level_values('year').max()  # getting last year in df
    networks = get_networks(df_triple, countries_all, year_start, year_end)  # getting netowrks
    resolution_range = list(map(lambda x: x/10, list(range(res_range_start, res_range_end))))
    logging.debug(f"Resolution range is {resolution_range}")
    communities = detect_local_communities(networks, df_triple, countries_all, year_start, year_end, resolution_range, centrality_threshold=centrality_threshold, community_detection=community_detection)
    hegemony_df = get_hegemony_scores(communities, resolution_range, year_start, year_end, countries_all, comm_name=comm_name)
    all_time_threshold = (year_end - year_start) * min_clients_for_top
    hegemony_top = get_hegemony_top(hegemony_df, comm_name, one_year_threshold=one_year_hegemony_threshold, all_time_threshold=all_time_threshold)
    visualize_hegemony(hegemony_top, title = f"{comm_name}: top hegemons")
    return communities, hegemony_df, hegemony_top

