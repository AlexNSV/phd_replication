import pandas as pd
import logging
from itertools import product, combinations

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df
from analysis.network_analysis import get_networks
from analysis.community import detect_local_communities
from analysis.hegemony import get_hegemony_scores, get_hegemony_top, visualize_hegemony
from data_handling import gsheet_handler
from utils.countryconverter import convert_country_df



def load_interventions():
    """Loads DoCaNoMI data 1992-2022"""

    df = gsheet_handler.read_gsheet(tablename='interventions', sheetname='i_main', skiprows=1)

    logging.info("Loaded DoCaNoMI data 1992-2022")
    return df

def preprocess_interventions(df, rolling_window=5, year_start=1992, year_end=2022, neighbourhood_value=0.1, neighbourhood_type='region'):
    """Preprocessing DoCaNoMI data:
        - removing non-cases
        - removing cases marked for removal
        - converting to time series format
        - converting to df in triple format (year*country*country)

    Parameters
    ------------
        df: pd.DataFrame()
            DataFrame with raw unprocessed DoCaNoMI Data
        rolling_window: int
            Rolling window to use for smoothing the data
        year_start: int
            Year to start with (needed for correct rolling calculations)
    Return
    -----------
        tuple(df , triple_df)
            df : pd.DataFramme() - preprocessed df with DoCaNoMI Data,
            triple_df  : pd.DataFramme() - DataFrame containing information about interventions in triple format (year*country*country). 
    """
    logging.info("Preprocessing DoCaNoMI Data")

    country_df = gsheet_handler.read_gsheet(tablename='country_data', sheetname='countryids', skiprows=0)
    
    # removing non-cases
    df = df[df['i_case']==1]
    # removing cases marked for removal
    df = df[~df['i_dyad_id'].str.contains('r')]

    df['i_year_end']=df['i_year_end'].replace('ongoing', 2022)
    df = df.dropna(subset=['i_year_end'])
    df['i_year_end'] = df['i_year_end'].astype(int)
    df['i_year_start'] = df['i_year_start'].astype(int)
    
    df['year'] = df.apply(lambda row: list(range(row['i_year_start'], row['i_year_end']+1)), axis=1)
    df = df.explode('year')
    
    df['i_case'] = df['i_case'].astype(float)
    
    #removing old data
    df = df[df['year'] >= year_start]
    df = df[['year', 'refobject_en', 'refsubject_en', 'i_case', 'i_burden_s_share']]
    
    # splitting up refobjects
    df['refobject_en'] = df['refobject_en'].apply(lambda x: x.split("; "))
    df = df.explode('refobject_en')
    df['refobject_en'] = df['refobject_en'].apply(lambda x: x.strip())

    df.replace({'Yugoslavia':'Serbia'}, inplace=True)
    
    #converting to STATE_en_UN (can do Alpha3_Code)
    df = convert_country_df(df, 'refobject_en', standard_to_convert='STATE_en_UN', purge=True)
    df = convert_country_df(df, 'refsubject_en', standard_to_convert='STATE_en_UN', purge=True)

    df['i_burden_s_share'] = df['i_burden_s_share'].replace('None', None).fillna(1)
    
    #setting up triples
    df_triple = df.groupby(["year", "refobject_en", "refsubject_en"]).sum()
    df_triple = df_triple[df_triple['i_case']!=0]
    
    # weighting by participation
    df_triple['i_case'] = df_triple['i_burden_s_share'].fillna(1)
    df_triple = pd.DataFrame(df_triple['i_case'])
    
    #countries_all = set(country_df)
    countries_all = get_all_countries(country_df.dropna(subset='state_en_un'), ego_column = 'state_en_un', alter_column = 'state_en_un')
    
    empty_df=get_empty_country_df(years=df['year'].unique(), countries_all=countries_all, names=["year", "refsubject_en", "refobject_en"])
    
    # multiplying with neighbourhood matrix
    
    zero_df = empty_df.loc[year_start].reset_index()
    if 'state_visual' in country_df.columns: country_df.set_index('state_visual', inplace=True)
    country_df.loc['Yugoslavia'] = country_df.loc['Serbia']  # fix
    if neighbourhood_type == 'region': 
        zero_df['value'] = zero_df.apply(lambda x: neighbourhood_value if country_df.loc[x['refsubject_en'],'region_lowest_level'] == country_df.loc[x['refobject_en'],'region_lowest_level'] else 0, axis=1)
    else:
        zero_df['value'] = 0
    zero_df['year'] = [list(range(year_start, year_end+1)) for i in zero_df.index]
    zero_df = zero_df.explode('year').set_index(['year', 'refsubject_en', 'refobject_en'])

    #print(zero_df)
    #print(df_triple)
    
    df_triple = zero_df.merge(df_triple,left_index=True, right_index=True, how='outer').fillna(0)
    #print(df_triple)
    df_triple['i_case'] = df_triple['i_case'] + df_triple['value']
    df_triple.drop('value', axis=1, inplace=True)

    # counting rolling average
    df_triple = df_triple.reset_index().set_index('year').groupby(['refobject_en', 'refsubject_en']).rolling(rolling_window, min_periods=1).mean()
    
    df_triple = df_triple.reset_index().set_index(['year', 'refobject_en', 'refsubject_en'])
    
    df_triple.index.names = ['year', 'alter', 'ego']
    df_triple.rename(columns={'i_case':'value'}, inplace=True)

    # Normalization
    df_triple['value'] = df_triple['value'] / df_triple['value'].max()
    
    logging.info("Done preprocessing DoCaNoMI Data")
    return df, df_triple


    
def interventions_main(year_start=1992, year_end=2022, rolling_window=5, 
                       res_range_start=2, res_range_end=20, one_year_hegemony_threshold=5, 
                       min_clients_for_top=3, centrality_threshold=0.5,
                       neighbourhood_value=0.1, community_detection='louvian'):
    comm_name = 'interventions'
    
    df = load_interventions() # Loading the data
    df, df_triple = preprocess_interventions(df, rolling_window=rolling_window, year_start=year_start, year_end=year_end, neighbourhood_value=neighbourhood_value, neighbourhood_type=None)  # data preprocessing
    
    countries_all = get_all_countries(processed_df=df, ego_column = 'refsubject_en', alter_column = 'refobject_en')  # getting set of all countries
    year_end = df_triple.index.get_level_values('year').max()  # getting last year in df
    networks = get_networks(df_triple, countries_all, year_start, year_end)  # getting netowrks
    resolution_range = list(map(lambda x: x/10, list(range(res_range_start, res_range_end))))
    logging.debug(f"Resolution range is {resolution_range}")
    communities = detect_local_communities(networks, df_triple, countries_all, year_start, year_end, resolution_range, centrality_threshold=centrality_threshold, community_detection=community_detection)
    hegemony_df = get_hegemony_scores(communities, resolution_range, year_start, year_end, countries_all, comm_name=comm_name)
    all_time_threshold = (year_end - year_start) * min_clients_for_top
    hegemony_top = get_hegemony_top(hegemony_df, comm_name, one_year_threshold=one_year_hegemony_threshold, all_time_threshold=all_time_threshold)
    visualize_hegemony(hegemony_top, title = f"{comm_name}: top hegemons")
    return networks, communities, hegemony_df, hegemony_top