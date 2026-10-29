import sipri # pip install sipri
import pandas as pd
import logging
from itertools import product, combinations

import sys
sys.path.append("..")

from utils.utils import get_all_countries, get_empty_country_df, test_df
from analysis.network_analysis import get_networks
from analysis.community import detect_local_communities
from analysis.hegemony import get_hegemony_scores, get_hegemony_top, visualize_hegemony
from utils.countryconverter import convert_country_df
from data_handling import gsheet_handler

JME_PATH="../data/raw/jme/jmeDataPublic.xlsx"
DEFAULT_GDP_THRESHOLD = 0.75  # какую долю от альтер должен составлять эго, чтобы тоже получить баллы

def load_jme():
    """Loads Joint Military Exercises data"""
    df = pd.read_excel(JME_PATH)
    logging.info("Loaded Joint Military Exercises data")
    return df
    
def preprocess_jme(df, year_start, year_end=2022, rolling_window=5, win_type='hann', gdp_threshold=DEFAULT_GDP_THRESHOLD, test_data=True, add_directionality=True):
    data_triple = []

    source_name = preprocess_jme.__name__.split('_')[1]
    
    EGO_LABEL = 'ego'
    ALTER_LABEL = 'alter'
    YEAR_LABEL = 'year'
    VALUE_LABEL = 'value'
    
    def _analyse_dyads(coparticipants_column: pd.Series, data_triple):
        """Add triple "country * country * year" data from a column into a global list
    
        Parameters
        ------------
            coparticipants_column: pd.Series
                A Series containing participants of one exercies
               
        """
        year = coparticipants_column.name[0]
        for dyad in combinations(coparticipants_column['countryName'].values, 2):  # US testing shows to little US, but maybe they will be on network level
        #for dyad in product(coparticipants_column['countryName'].values, repeat=2): # doubts, it might be excessive
            if dyad[0] != dyad[1]:
                #if 'United States' in dyad: print(year, dyad)
                dyad = list(dyad)
                data_triple += ((year, *dyad)),
                dyad.reverse()
                data_triple += ((year, *dyad)),
    # getting year * exerciseId * country data configuration
    dyad_df = pd.DataFrame(df.groupby(['startYear', 'xID', 'countryName'])['startMonth'].count()).reset_index().drop('startMonth', axis=1).set_index(['startYear', 'xID'])
    countries_all = get_all_countries()
    
    dyad_df.reset_index(inplace=True)
    dyad_df=dyad_df[dyad_df['startYear']>=year_start]
    dyad_df.groupby(['startYear','xID']).apply(lambda x: _analyse_dyads(x, data_triple))
    df_triple = pd.DataFrame(data_triple, columns = ['year', 'ego', 'alter'])
    df_triple['value'] = 1
    df_triple = df_triple.groupby(['year', 'ego', 'alter']).count()

    ## ОБЩЕЕ -------------------------------------------------------------------------------------------

    df_triple.reset_index(inplace=True)
    
    #converting to STATE_en_UN (can do Alpha3_Code)
    df_triple = convert_country_df(df_triple, 'alter', standard_to_convert='STATE_en_UN', purge=True)
    df_triple = convert_country_df(df_triple, 'ego', standard_to_convert='STATE_en_UN', purge=True)
    
    df_triple.set_index(['year', 'ego', 'alter'], inplace=True)


    
    empty_df=get_empty_country_df(years=df_triple.reset_index()['year'].unique(), countries_all=countries_all, names=["year", "alter", "ego"])
    # merging the df with zero df
    df_triple = empty_df.merge(df_triple,left_index=True, right_index=True, how='outer').fillna(0)
    
    # Normalization
    df_triple['value'] = df_triple['value'] / df_triple['value'].max()
    ## END OF ОБЩЕЕ -------------------------------------------------------------------------------------------
    # filling missing 2017-2022 with 2016 data
    if test_data:
        test_df(df_triple[df_triple['value']>0].reset_index(), source_name, year_start=year_start, year_end=year_end, alter_label=ALTER_LABEL, ego_label=EGO_LABEL, year_label=YEAR_LABEL, value_label=VALUE_LABEL)
    for year in range(2017, year_end+1):
        df_year = df_triple.reset_index()[df_triple.reset_index()['year']==2016].replace({2016:year})
        df_year.set_index(['year', 'alter', 'ego'], inplace=True, drop=True)
        df_triple=pd.concat([df_triple, df_year])
    
    
    # Добавляем направленности с помощью данных ВВП
    if add_directionality:
        gdp_df = gsheet_handler.read_gsheet(tablename='country_data', sheetname='countryids', skiprows=0).dropna(subset=['state_en_un']).set_index('state_en_un')['gdp2018']
        gdp_df['German Democratic Republic'] = 1049550000000.0
        gdp_df['Czechoslovakia'] = 57600000000.0
        gdp_df['State of Palestine'] = 14498000000.0
        
        df_triple.reset_index(inplace=True)
        df_triple['ego_gdp2018']=df_triple['ego'].apply(lambda x: gdp_df[x])
        df_triple['alter_gdp2018']=df_triple['alter'].apply(lambda x: gdp_df[x])
        df_triple['value'] = df_triple.apply(lambda x: x['value'] if x['ego_gdp2018']/x['alter_gdp2018']>gdp_threshold else 0, axis=1)
        df_triple.set_index(['year', 'ego', 'alter'], inplace=True, drop=True)
        df_triple.drop(['ego_gdp2018', 'alter_gdp2018'], axis=1, inplace=True)
    
    # counting rolling average
    if rolling_window is not None: #  untested
        df_triple = df_triple.reset_index().set_index('year').groupby(['ego', 'alter']).rolling(rolling_window, min_periods=1, win_type=win_type).mean()
        df_triple = df_triple.reset_index().set_index(['year', 'ego', 'alter'])
    
    return df, df_triple

def jme_main(year_start=1992, rolling_window=None, res_range_start=2, res_range_end=20, one_year_hegemony_threshold=5, 
             min_clients_for_top=3, centrality_threshold=0.45, centrality_type='out-degree'):
    comm_name = 'jme'
    
    df = load_jme()
    df, df_triple = preprocess_jme(df, year_start, rolling_window)
    year_end = df_triple.index.get_level_values('year').max()  # getting last year in df
    networks = get_networks(df_triple, countries_all, year_start, year_end, isDigraph=True, removeLessThanZero=False)  # getting netowrks
    resolution_range = list(map(lambda x: x/10, list(range(res_range_start, res_range_end))))
    logging.debug(f"Resolution range is {resolution_range}")
    communities = detect_local_communities(networks, df_triple, countries_all, year_start, year_end, resolution_range, centrality_threshold=centrality_threshold, centrality_type=centrality_type)
    hegemony_df = get_hegemony_scores(communities, resolution_range, year_start, year_end, countries_all, comm_name=comm_name)
    all_time_threshold = (year_end - year_start) * min_clients_for_top
    hegemony_top = get_hegemony_top(hegemony_df, comm_name, one_year_threshold=one_year_hegemony_threshold, all_time_threshold=all_time_threshold)
    visualize_hegemony(hegemony_top, title = f"{comm_name}: top hegemons")
    return communities, hegemony_df, hegemony_top
    #return df_triple