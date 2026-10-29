import sipri # pip install sipri
import pandas as pd
import logging
from itertools import product, combinations

import sys
sys.path.append("..")

from utils import get_all_countries, get_empty_country_df
from network_analysis import get_networks
from community import detect_local_communities
from hegemony import get_hegemony_scores, get_hegemony_top, visualize_hegemony
from countryconverter import convert_country_df

from sipri import load_sipri, preprocess_sipri
from jme import load_jme, preprocess_jme
from interventions import load_interventions, preprocess_interventions
from data_handling import gsheet_handler

def analysis_main(year_start=1992, rolling_window=5, res_range_start=2, res_range_end=20, one_year_hegemony_threshold=5, min_clients_for_top=3, centrality_threshold=0.5, community_detection='louvian', hierarchy_threshold=0, gdp_threshold=0.75):
    comm_name = 'security_combined'
    
    country_df = gsheet_handler.read_gsheet(tablename='country_data', sheetname='countryids', skiprows=0)['state_en_un'].dropna()
    countries_all = set(country_df)
    
    #jme_df = load_jme()
    #jme_df, jme_df_triple = preprocess_jme(jme_df, year_start=1992, rolling_window=5, gdp_threshold=gdp_threshold)
    
    weapon_df = load_sipri()  # Loading the data
    weapon_df, weapon_df_triple = preprocess_sipri(weapon_df, rolling_window=5, year_start=1992)  # data preprocessing

    intervention_df = load_interventions()
    intervention_df, intervention_df_triple = preprocess_interventions(intervention_df, rolling_window=5, year_start=1992, neighbourhood_type=None)  # data preprocessing

    df_triple = intervention_df_triple + weapon_df_triple #+ jme_df_triple
    
    #countries_all = get_all_countries(processed_df=df)  # getting set of all countries
    year_end = df_triple.index.get_level_values('year').max()  # getting last year in df
    
    networks = get_networks(df_triple, countries_all, year_start, year_end)  # getting netowrks
    resolution_range = list(map(lambda x: x/10, list(range(res_range_start, res_range_end))))
    logging.debug(f"Resolution range is {resolution_range}")
    communities = detect_local_communities(networks, df_triple, countries_all, year_start, year_end, resolution_range, centrality_threshold=centrality_threshold, community_detection=community_detection, hierarchy_threshold=hierarchy_threshold)
    hegemony_df = get_hegemony_scores(communities, resolution_range, year_start, year_end, countries_all, comm_name=comm_name)
    all_time_threshold = (year_end - year_start) * min_clients_for_top
    hegemony_top = get_hegemony_top(hegemony_df, comm_name, one_year_threshold=one_year_hegemony_threshold, all_time_threshold=all_time_threshold)
    visualize_hegemony(hegemony_top, title = f"{comm_name}: top hegemons")
    return df_triple, communities, hegemony_df, hegemony_top