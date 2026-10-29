# in development

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