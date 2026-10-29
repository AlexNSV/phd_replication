import logging
from matplotlib import pyplot as plt

import sys
sys.path.append("..")

from utils.utils import get_empty_country_df

def get_hegemony_scores(communities, resolution_range, year_start, year_end, countries_all, comm_name='weapon_trade'):
    """Counts hegemony"""
    df = get_empty_country_df(range(year_start, year_end+1), countries_all, ['year','ego', 'alter'])
    df[comm_name] = 0
    #resolution_range = list(map(lambda x: x/10, list(range(1, 20))))
    for year in range(year_start, year_end + 1):
        for resolution in resolution_range:
            logging.debug(f"counting hegemony for {year}, res={resolution}, there are {len(communities[year][resolution])} communities")
            for community in communities[year][resolution]:
                if community.hegemons is None: continue
                for hegemon in community.hegemons:
                    for client in hegemon.clientele:
                        df.loc[(year, hegemon.name, client), comm_name] += hegemon.strength
    return df

def get_hegemony_top(hegemony_df, comm_name, one_year_threshold=5, all_time_threshold=100): #, MIN_CLIENTS_FOR_GRAPH = 5, )
    """Returns a df of top hegemons for each year
    
    Arguments
    ---------
        hegemony_df : pd.DataFrame
            hegemony scores in a year*ego*alter = value_name DataFrame
        comm_name : str
            name of the value (community) in hegemony_df
        one_year_threshold : int
            minimal hegemony_score to keep analysing a pair of ego - alter as hegemon - client, default is 5
        all_time_threshold : int
            minimal hegemony_score sum of hegemon - client clients across all years for a hegemon to include the hegemon in the top, default is 100 (should be scaled to year number)

    Returns
    --------
        hegemony_df_n_year_top : pd.DataFrame
            DataFrame of hegemon counts hegemon by year
    """

    hegemony_df_t = hegemony_df[hegemony_df[comm_name]>one_year_threshold]
    # считаем клиентов
    hegemony_df_n = hegemony_df_t.reset_index().groupby(['year','ego']).count()
    
    # Отрезаем маленьких
    #hegemony_df_n_top = hegemony_df_n[hegemony_df_n['alter']>=MIN_CLIENTS_FOR_GRAPH]
    # года по колонкам
    hegemony_df_n_year = hegemony_df_n.drop(comm_name, axis=1).reset_index().pivot(index='ego',columns='year').fillna(0)
    hegemony_df_n_year['sum']=hegemony_df_n_year.sum(axis=1)
    hegemony_df_n_year_top = hegemony_df_n_year[hegemony_df_n_year['sum']>all_time_threshold]
    hegemony_df_n_year_top = hegemony_df_n_year_top.drop('sum', axis=1)
    return hegemony_df_n_year_top

def visualize_hegemony(hegemony_df_n_year_top, title, savefile=None):
    
    hegemons = list(hegemony_df_n_year_top.index)
    
    figsize_x = 15
    figsize_y = 8
    dpi = 300
    
    fig, ax = plt.subplots(figsize=(figsize_x,figsize_y), dpi=dpi)
    
    for hegemon in hegemons:
        ax.plot(hegemony_df_n_year_top.columns.get_level_values(1), hegemony_df_n_year_top.loc[hegemon], label=hegemon)
    
    ax.set_title(title)
    ax.set_ylabel = 'Клиентов в зоне гегемонии'
    ax.legend()

    if savefile:
        plt.savefig(savefile)
    
    plt.show()