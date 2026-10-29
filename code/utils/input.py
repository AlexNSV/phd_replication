import sys
sys.path.insert(1, '..')

import networkx as nx
import pandas as pd
import logging
import io

from itertools import product
from pathlib import Path

def get_categories(directory_path):
    """
    Извлекает уникальные категории из имён файлов форматов:
      - df_category_year.csv   (одинарное подчёркивание)
      - df__category__year.csv (двойное подчёркивание)
    в заданной директории. Логирует путь, список всех файлов и найденные категории.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise ValueError(f"Путь '{directory_path}' не является директорией или не существует.")

    logging.info(f"Сканируем папку: {dir_path.resolve()}")

    # Логируем содержимое папки
    try:
        all_items = list(dir_path.iterdir())
        logging.info(f"Содержимое папки ({len(all_items)} элементов):")
        for item in all_items:
            logging.info(f"  - {item.name}")
    except PermissionError as e:
        logging.warning(f"Не удалось прочитать содержимое папки: {e}")

    categories = set()

    # Ищем все CSV-файлы, начинающиеся с 'df'
    for file_path in dir_path.glob('df*.csv'):
        stem = file_path.stem          # например 'df_economy_2022' или 'df__category__year'
        logging.debug(f"Обрабатываем файл: {file_path.name}, stem='{stem}'")

        # Пробуем распарсить с одинарным подчёркиванием
        parts_single = stem.split('_')   # ['df', 'economy', '2022']
        if len(parts_single) >= 3 and parts_single[0] == 'df':
            # Проверяем, что последняя часть похожа на год (цифры)
            if parts_single[-1].isdigit():
                category = '_'.join(parts_single[1:-1])  # на случай, если в категории есть подчёркивания
                categories.add(category)
                logging.info(f"Добавлена категория: '{category}' из файла {file_path.name}")
                continue

        # Если не подошло, пробуем с двойным подчёркиванием
        parts_double = stem.split('__')  # ['df', 'category', 'year']
        if len(parts_double) == 3 and parts_double[0] == 'df':
            categories.add(parts_double[1])
            logging.info(f"Добавлена категория: '{parts_double[1]}' из файла {file_path.name}")
            continue

        # Если ни один формат не подошёл – пропускаем
        logging.warning(f"Файл {file_path.name} не соответствует ожидаемым форматам и пропущен")

    logging.info(f"Найдено уникальных категорий: {len(categories)}")
    result = sorted(categories)
    logging.info(f"Итоговый список категорий: {result}")
    return result



def read_subsystems(year_start=1992, year_end=2022, subsystem_cats=None, isDigraph = True):
    if subsystem_cats is None:
        subsystem_cats=get_categories('../data/preprocessed/subsystem_level/')
    logging.info(f'reading subsystems {subsystem_cats}')
    subsystems = dict()
    for subsystem, y in product(subsystem_cats, range(year_start, year_end+1)):
        if not subsystem in subsystems.keys(): subsystems[subsystem] = dict()
        if not 'networks' in subsystems[subsystem].keys(): subsystems[subsystem]['networks'] = dict()
        if isDigraph:
            create_using = nx.DiGraph
        else:
            create_using = nx.Graph
        subsystems[subsystem]['networks'][y] = nx.read_weighted_edgelist(f'../data/networks/n_{subsystem}_{y}.nx', comments='#', delimiter='|', create_using=create_using, nodetype=None, encoding='utf-8')
    for subsystem in subsystem_cats:
        subsystems[subsystem]['data'] = pd.read_csv(f'../data/preprocessed/subsystem_level/df_{subsystem}_{y}.csv', index_col=['year','alter','ego'])
    return subsystems