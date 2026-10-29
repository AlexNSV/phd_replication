import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging
from itertools import product, permutations
import warnings
import os


from data_handling import gsheet_handler
from utils.countryconverter import convert_country_df

GREAT_POWERS = {"China, People's Republic of", "France", "Germany", "India", "Russian Federation", "United Kingdom of Great Britain and Northern Ireland", "United States"}
YEAR_MAX_SUPPORTED = 2030

def get_all_countries(processed_df=None, ego_column = 'seller', alter_column = 'buyer'):
    """Returns a set of all countries, both ego and alter
    
    Parameters
    ------------
        processed_df: pd.DataFrame()
            DataFrame with raw processed data
        ego_column: str or None
            Name of the column with ego names
        alter_column: str or None
            Name of the column with alter names
    
    Return
    -----------
        countries_all: set
            Set of all countries
    """
    logging.debug("getting all countries for df")
    if processed_df is None:
        country_df = gsheet_handler.read_gsheet(tablename='country_data', sheetname='countryids', skiprows=0)['state_en_un'].dropna()
        return set(country_df)
    if alter_column is not None:
        alter_all = processed_df[alter_column].unique()
    else:
        alter_all = []
    if ego_column is not None:
        ego_all = processed_df[ego_column].unique()
    else:
        ego_column = []
    countries_all = set(list(alter_all) + list(ego_all))
    logging.info(f"there are {len(countries_all)} unique countries, {len(ego_all)} - ego, {len(alter_all)} - alter")
    return countries_all

def _get_empty_country_df(years, countries_all, names):
    # LEGACY
    # filling an empty year*country*country dataframe with no values
    warnings.warn("deprecated version, use get_empty_country_df instead", DeprecationWarning)
    options = product(years,countries_all, countries_all)
    index = pd.MultiIndex.from_tuples(options, names=names)
    empty_df = pd.DataFrame(index=index)
    return empty_df

def get_empty_country_df(years, countries_all=None, names=['year', 'alter', 'ego']):
    # filling an empty year*country*country dataframe with no values
    sm = get_system_members(min(years), max(years))
    sm.set_index(['year', 'alter', 'ego'], drop=True, inplace=True)
    sm.index.names=names
    #index = pd.MultiIndex.from_tuples(sm.values, names=names)
    #empty_df = pd.DataFrame(index=index)
    return sm#empty_df

def _get_system_members_base():
    sm = pd.read_csv('../data/raw/system_membership/system2016.csv')  #  Manually fixed Andorra's inexistance in 1962-1993
    sm = convert_country_df(sm, 'ccode', numeric_type='cow', warning=False, print_convertions=False)
    sm = sm.replace('None', np.nan).dropna()
    sm = sm[['ccode', 'year']]
    sm = sm.rename({'ccode': 'ego'}, axis=1)
    # Adding post_2016 years
    post_2016_dfs = []
    for y in range(2017, YEAR_MAX_SUPPORTED):
        df_y = sm[sm['year']==2016].replace(2016, y)
        post_2016_dfs += df_y,
    sm = pd.concat([sm] + post_2016_dfs)
    return sm

def get_system_members(year_start, year_end, great_only=False):
    if year_end > YEAR_MAX_SUPPORTED: 
        raise NotImplementedError(f"cannot process {year_end}, max is {YEAR_MAX_SUPPORTED}")
    _sm = sm.copy()
    # Adding Palestine
    palestine_values = [['State of Palestine', y] for y in range (year_start, year_end+1)]
    palestine_df = pd.DataFrame(palestine_values, columns=['ego', 'year'])
    _sm = pd.concat([_sm, palestine_df])
    # Adding Germany (German Federal Republic)
    germany_values = [['Germany', y] for y in range (year_start, 1990)]
    germany_df = pd.DataFrame(germany_values, columns=['ego', 'year'])
    _sm = pd.concat([_sm, germany_df])
    
    _sm = _sm[(_sm['year']>=year_start) & (_sm['year']<=year_end)]
    sm_dyads = []
    for y in range(year_start, year_end+1):
        countries_y = set(_sm[_sm['year']==y]['ego'].unique())
        sm_dyad_y = pd.DataFrame(list(permutations(countries_y, 2)), columns = ['ego', 'alter'])
        sm_dyad_y['year'] = y
        sm_dyads += sm_dyad_y,
    sm_dyad = pd.concat(sm_dyads)
    if great_only:
        sm_dyad = sm_dyad[(sm_dyad['ego'].isin(GREAT_POWERS))|sm_dyad['alter'].isin(GREAT_POWERS)]
    #sm.apply(lambda x:get_dyad_dict(x['ego'], x['year'], ),axis=1)
    return sm_dyad

def visualise_test(df_triple, dataname='test', alter_label='alter', ego_label='ego', year_label='year', value_label='value', topnum=3):
    # Выбираем данные за последние 30 лет
    years_to_plot = range(df_triple[year_label].min(), df_triple[year_label].max() + 1)
    
    # Создаем пустой список для хранения топ-10 стран
    top_10_countries = set()
    
    for year in years_to_plot:
      # Фильтруем строки по году
      current_year_df = df_triple.query(f'{year_label} == {year}')
      
      # Сортируем страны по убыванию объема торговли
      sorted_df = current_year_df.sort_values(by=value_label, ascending=False)
      
      # Берем первые 10 строк (топ-10 стран)
      top_10_current_year = set(sorted_df.head(topnum)[ego_label])
      
      # Добавляем эти страны в общий список
      top_10_countries = top_10_countries | top_10_current_year
    
    # Объединяем все данные в один датафрейм
    #top_10_overall = pd.concat(top_10_countries)
    #t[t['ego'].isin(['Afghanistan'])]
    
    # Группируем данные по странам и годам
    grouped_data = df_triple[df_triple[ego_label].isin(top_10_countries)].groupby([ego_label, year_label])[value_label].sum().reset_index()
    
    # Построение графика
    fig, ax = plt.subplots(figsize=(12, 8))
    
    logging.debug(grouped_data[ego_label].unique())
    
    for country in grouped_data[ego_label].unique():
      country_data = grouped_data.query(f'{ego_label} == "{country}"')
      ax.plot(country_data[year_label], country_data[value_label], label=country)
    
    ax.set_xlabel("Год")
    ax.set_ylabel(dataname)
    ax.legend(title=f"Топ-{topnum} стран")
    plt.title(f"Топ-{topnum} стран по {dataname}")
    logging.info("saving figure")
    plt.savefig(f"../output/test_plots/test_{dataname}.png")

def test_df(df_triple, source_name, year_start, year_end, alter_label='alter', ego_label='ego', year_label='year', value_label='value'):
    logging.info("Testing initiated")
    sm = get_system_members(year_start, year_end)
    sm_great = get_system_members(year_start, year_end, great_only=True)
    egos = set(df_triple[ego_label])
    alters = set(df_triple[alter_label])
    years = set(df_triple[year_label])
    sm_set = set(sm['ego'])
    sm_set_great = set(sm_great['ego'])
    sm_years = set(range(year_start, year_end+1))
    missing_egos = sm_set - egos
    extra_alters = alters - sm_set
    extra_egos = egos - sm_set
    missing_great_egos = GREAT_POWERS - egos
    missing_alters = sm_set - alters
    missing_great_alters = GREAT_POWERS - alters
    missing_years = sm_years - years
    testing_results_string = f"Missing {len(missing_egos)}({len(egos)}/{len(sm_set)} = ({len(egos)/len(sm_set)})) egos, {len(missing_alters)}({len(alters)}/{len(sm_set)} = {len(alters)/len(sm_set)}) alters, {len(sm_years)-len(years)} years\n"
    testing_results_string += f"Missing egos: {missing_egos}\n"
    testing_results_string += f"Missing great egos: {missing_great_egos}\n"
    testing_results_string += f"Missing alters: {missing_alters}\n"
    testing_results_string += f"Missing great alters: {missing_great_alters}\n"
    testing_results_string += f"Missing years: {missing_years}\n"
    testing_results_string += f"{sm.shape[0]} - {df_triple.shape[0]} = {sm.shape[0] - df_triple.shape[0]} lines missing ({df_triple.shape[0] / sm.shape[0]})\n"
    if len(extra_egos) > 0: testing_results_string += f"Extra egos: {extra_egos}\n"
    if len(extra_alters) > 0: testing_results_string += f"Extra alters: {extra_alters}\n"
        
    
    for chunk in testing_results_string.split("\n"):
        logging.info(chunk)
    #logging.info(f"Missing {len(missing_egos)}({len(egos)}/{len(sm_set)} = ({len(egos)/len(sm_set)})) egos, {len(missing_alters)}({len(alters)}/{len(sm_set)} = {len(alters)/len(sm_set)}) alters, {len(sm_years)-len(years)} years")
    #logging.info(f"Missing egos: {missing_egos}")
    #logging.info(f"Missing great egos: {missing_great_egos}")
    #logging.info(f"Missing alters: {missing_alters}")
    #logging.info(f"Missing great alters: {missing_great_alters}")
    #logging.info(f"Missing years: {missing_years}")
    #logging.info(f"{sm.shape[0]} - {df_triple.shape[0]} = {sm.shape[0] - df_triple.shape[0]} lines missing ({df_triple.shape[0] / sm.shape[0]})")
    #if len(extra_egos) > 0: logging.info(f"Extra egos: {extra_egos}")
    #if len(extra_alters) > 0: logging.info(f"Extra alters: {extra_alters}")
    df_triple_great = df_triple.reset_index()[(df_triple.reset_index()[ego_label].isin(GREAT_POWERS))|(df_triple.reset_index()[alter_label].isin(GREAT_POWERS))]
    result_dict = {
        'Source name' : source_name,
        'Missing egos' : len(missing_egos),
        'Missing great egos' : len(missing_great_egos),
        'Missing alters' : len(missing_alters),
        'Missing great alters' : len(missing_great_alters),
        'Missing years' : len(missing_years),
        'Missing yearly dyads' : sm.shape[0] - df_triple.shape[0],
        'Missing yearly dyads' : sm_great.shape[0] - df_triple_great.shape[0],
        'Egos percent' : len(egos) / len(sm_set),
        'Alters percent' : len(alters) / len(sm_set),
        'Yearly dyads percent' : df_triple.shape[0] / sm.shape[0],
        'Yearly great dyads percent' : df_triple_great.shape[0] / sm_great.shape[0],
    }
    results_df = pd.DataFrame.from_dict(result_dict, orient='index')
    results_df.T.to_csv(f'../data/testing/{source_name}_test.csv')
    logging.info(results_df)

    visualise_test(df_triple.reset_index(), source_name, alter_label, ego_label, year_label, value_label)
    
    logging.info("Testing Done")

def test_sphere(dfs: dict, years='all'):
    """Возвращает топ-10 в каждой подсфере для быстрого визуального теста"""
    top_10 = dict()
    for sphere in dfs:
        d = dfs[sphere].copy()
        d.reset_index(inplace=True)
        if years != 'all':
            d = d[d['year'].isin(years)]
        top = d.groupby('ego').sum().sort_values('value').tail(10).reset_index()
        top_10[sphere] = top[['ego', 'value']]
    return top_10

def row_geometric_mean(row, subsystem_weights, print_rows=False):
    """геометрическое среднее для N членов ряда row"""
    if print_rows:
        print(row)
    values = row.values
    weights = list(subsystem_weights.values())
    return np.prod(values ** weights) ** (1 / np.sum(weights))

def get_percent_df(processed_df, year=None, EGO_LABEL='ego', YEAR_LABEL='year', VALUE_LABEL='value'):
    """Returns df with information on what percent does each alter has for ego in year year
    """
    df = processed_df.reset_index()
    if year is None: year = df[YEAR_LABEL].min()
    grouped=df.groupby([EGO_LABEL, YEAR_LABEL])[VALUE_LABEL].transform('sum')
    df['percent']=(df[VALUE_LABEL] / grouped)# * 100
    percent_df=df[df[YEAR_LABEL]==year]
    percent_df.set_index(['alter','ego'], inplace=True)
    return percent_df.fillna(0)

def load_csv_files_from_folder(folder_path: str) -> dict:
    """
    Загружает все .csv файлы из папки в словарь DataFrame'ов.

    Параметры:
        folder_path (str): путь к папке с CSV-файлами.

    Возвращает:
        dict: словарь, где ключи — имена файлов без расширения .csv,
              значения — соответствующие pandas DataFrame.

    Игнорируются:
        - файлы, не оканчивающиеся на .csv
        - временные файлы (начинающиеся с '~$')
        - скрытые файлы (начинающиеся с '.')
    """
    dataframes = {}
    for filename in os.listdir(folder_path):
        # Проверяем, что файл имеет расширение .csv и не является временным/скрытым
        if filename.endswith('.csv') and not filename.startswith('~$') and not filename.startswith('.'):
            file_path = os.path.join(folder_path, filename)
            key = os.path.splitext(filename)[0]  # имя без расширения
            dataframes[key] = pd.read_csv(file_path)
    return dataframes

def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '†'
    else:
        return ''


sm = _get_system_members_base()
