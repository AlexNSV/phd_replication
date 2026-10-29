import sys
sys.path.insert(1, '..')

import pandas as pd
import networkx as nx
import logging

from analysis.network_analysis import countCentrality, get_network_from_year_df

class Community():
    """
    Object representing a community of states.

    ...

    Attributes
    ----------
    members : list
        List of countries which are members of the community.
    hierarchy_score : float
        Hierarchy score (presumable using Global Reaching Centrality - https://doi.org/10.1371/journal.pone.0033799)
    resolution : float
        Resolution coefficient in Louvian comminity detection modularity formula
    hegemons : list
        List of hegemons in the community
    year : int
        Year the community was observed in

    """
    def __init__(self, members: list(), hierarchy_score: float(), resolution: float(), hegemons: list(), year: int):
        self.members = members
        self.hierarchy_score = hierarchy_score
        self.resolution = resolution
        self.hegemons = hegemons
        self.year = year

    def __repr__(self):
        repr = f'{self.year} Community at resolution {self.resolution} of {self.population} with {self.hegemons} as hegemon (hierarchy score = {self.hierarchy_score}).\nThe members are {self.members}'
        return repr
        
    @property
    def population(self):
        """
        Returns the number of states in the community.
        """
        return len(self.members)

class Hegemon():
    """
    Object representing a hegemonic state.

    ...

    Attributes
    ----------
    name : str
        Name of the hegemonic country
    co_hegemons : list
        Other hegemons in the community
    clientele : list
        Other countries in the community, hegemon's sphere of influence
    centrality : float
        Centrality measure of the hegemon in its local community
    """

    @property
    def strength(self):
        return 1/(len(self.co_hegemons) + 1)
        
    
    def __init__(self, name, co_hegemons, clientele, centrality):
        self.name = name
        self.co_hegemons = co_hegemons
        self.clientele = clientele
        self.centrality = centrality

    def __repr__(self):
        repr = f'{self.name} ({self.centrality:.3f})'
        return repr

def analyse_local_community(network, df_triple, countries_all, year, centrality_threshold, centrality_type, community_detection='louvian', resolution=None, max_size=None, hierarchy_threshold=0):
    """Analyses a single local community using louvian heuristic"""
    logging.debug(f"Resolution is {resolution}, year is {year}")
    unirected_g = nx.Graph(network) # Using undirected graph for the purposes of community detection

    if community_detection == 'louvian':
        communities_generator = nx.community.louvain_communities(unirected_g, weight='value', resolution=resolution, seed=100)
    elif community_detection == 'lukes':
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.lukes.lukes_partitioning.html
        communities_generator = nx.community.lukes_partitioning(unirected_g, max_size=max_size, edge_weight='value', node_weight=None)
    elif community_detection == 'greedy_modularity':
        communities_generator = nx.community.greedy_modularity_communities(unirected_g, weight='value', seed=100, resolution=resolution)
    elif community_detection == 'k_clique':
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.kclique.k_clique_communities.html
        communities_generator = nx.community.k_clique_communities(unirected_g, k = 5)  # k is the size of a smallest clique
    else:
        raise NotImplementedError(community_detection)
    communities = dict()

    for j, country in enumerate(countries_all):
        for i, cluster in enumerate(communities_generator):
            if country in cluster:
                communities[country] = i

    community_df = pd.DataFrame(communities.items(), columns = ['Country', 'Community'])
    # counting population of clusters
    cluster_count = community_df.groupby('Community').count().to_dict()['Country']
    community_df['Community pop'] = community_df.apply(lambda x: cluster_count[x['Community']], axis=1)
    community_df = community_df.set_index('Country')
    # measuring centrality
    centrality_all = countCentrality(network, countries_all, centrality_type)
    res_df = pd.concat([community_df, centrality_all], axis=1)
    # exclude 1-pop clusters from analysis
    res_df = res_df[res_df['Community pop'] > 1]
    # counting within-comminity centrality
    community_count = len(res_df['Community'].unique())
    community_members = list()
    community_infos = list()
    for i, community in enumerate(res_df['Community'].unique()):
        comm_members = list(res_df[res_df['Community'] == community].index)
        comm_network = get_network_from_year_df(df_triple, comm_members, year, ego_indexname = 'ego', alter_indexname = 'alter', forceString=False) # remove forceString if error
        fin_df = countCentrality(comm_network, comm_members, centrality_type, 'local').sort_values('local_centrality', ascending=False)
        community_members += len(comm_members),
        hierarchy_score = nx.global_reaching_centrality(comm_network) #digraph only
        #print(fin_df)
        hegemons=None
        
        #if hierarchy_score > 0.75:
        # C 0 лучше работает почему-то, чем с проверкой иерархии
        if hierarchy_score > hierarchy_threshold:
            hegemons = list()
            hegemons_names = fin_df[fin_df['local_centrality'] >= centrality_threshold].index.to_list()
            #hegemon_counts += [len(hegemons_names)]
            for i, hegemon_name in enumerate(hegemons_names):
                #print(hegemon_name, i, hegemons_names, hegemons_names.copy().remove(hegemon_name))
                co_hegemons = hegemons_names.copy()
                co_hegemons.pop(i)
                comm_members_ = comm_members.copy()
                comm_members_.remove(hegemon_name)
                hegemon = Hegemon(hegemon_name, co_hegemons, comm_members_, fin_df.loc[hegemon_name, 'local_centrality'])
                logging.debug(f"hegemon is {hegemon.name}, co-hegemons are {hegemon.co_hegemons}")
                hegemons += hegemon,
            if len(hegemons_names) == 0: hegemons = None
            #hegemon_list[year] += (hegemons, len(comm_members)),
        #print(resolution/10)
        community_info = Community(comm_members, hierarchy_score, resolution/10, hegemons, year)
        community_infos += community_info,
    #community_infos[f'{i} - {resolution/10}'] = community_info
    logging.debug(f"Число сообществ: {community_count}")
    if (len(community_members)!=0): logging.debug(f"Среднее число стран: {sum(community_members)/len(community_members)}")
    #logging.debug(f"Число гегемонов: {len(hegemon_list)}")
    #logging.debug(hegemon_list)
    #optimization_coefs += [resolution/10]
    #community_counts += [community_count]
    #community_members_all += (sum(community_members)/len(community_members),)
    #hegemon_counts += [len(hegemon_list[year])]
    return community_infos
        

def detect_local_communities(networks, df_triple, countries_all, year_start, year_end, resolution_range, centrality_threshold, centrality_type='out-degree', community_detection='louvian', hierarchy_threshold=0):
    """Analyses multiple local communities"""
    #hegemon_list = dict()
    community_infos = dict()
    for year in range(year_start, year_end + 1):
        #hegemon_list[year] = []
        #optimization_coefs = []
        #community_counts = []
        #community_members_all = []
        #hegemon_counts = []
        community_infos[year] = dict()
        if (community_detection == 'louvian') or (community_detection == 'greedy_modularity'):
            for resolution in resolution_range:
                community_infos[year][resolution] = analyse_local_community(networks[year], 
                                                                            df_triple, countries_all, year,
                                                                            centrality_threshold=centrality_threshold, 
                                                                            centrality_type=centrality_type, community_detection=community_detection,
                                                                            resolution=resolution, hierarchy_threshold=hierarchy_threshold
                                                                           )
        else:
            community_infos[year][0] = analyse_local_community(networks[year], df_triple, countries_all, year, centrality_threshold, centrality_type, community_detection, max_size=100000)
    return community_infos