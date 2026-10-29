#!/usr/bin/env python
# coding: utf-8

# In[7]:
import sys
sys.path.append("..")


import pandas as pd
import geopandas
import matplotlib.colors as clr
import matplotlib.patches as ptch
import numpy as np

from matplotlib import cm
from pathlib import Path
from matplotlib import pyplot as plt

#загрузка базовых данных
gdf = geopandas.read_file("../geoshapes/v22_2608.shp")
gdf = gdf.set_index('STATE_en_U')

# Loads a column into a world shapefile, keeping non-existant values empty
def importShapefile(df, columnname = None, *, columnnames = None, country_subset = None, savepathname = None):
    
    # Can both handle lists of columnnames or single a single columnname
    if columnnames is None:
        assert columnname is not None, 'Must specify one of "columnname" or "columnnames"'
        columnnames = [columnname]
    else:
        assert columnname is None, 'Cannot specify both "columnname" and "columnnames"'
    
    if country_subset is None:
        country_subset = gdf.index.values
   
    if savepathname is None:
        savepathname = f'{columnname}_shapefile/{columnname}'
    Path(savepathname.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)

    
    
    df_indexed = df #.set_index('STATE_en_UN')
    for columnname in columnnames:
        gdf[columnname] = np.nan
        for country in country_subset:
            # copying value by value from a df to a gdf
            gdf.loc[country, columnname] = df_indexed.loc[country, columnname]
        print(f'Writing {columnname} to shapefile')

    gdf.to_file(f"{savepathname}.shp")
    print(f'Saved {savepathname}.shp, {savepathname}.dbf, {savepathname}.cpg, {savepathname}.prj, {savepathname}.shx')
    
def drawMap(df, stat = 'cluster', savename = 'basic', addlegend = False, addclusterlegend = False, colormap = 'viridis', lang = 'en'):
    
    if savename == 'basic':
        savename = stat + df_working_column + 'map.png'
    
    df_indexed = df #.set_index('STATE_en_UN')

    gdf[stat] = np.nan
    for country in gdf.index.values:
        gdf.loc[country, stat] = df_indexed.loc[country, stat]
    #присоединение новых данных
    #gdf['df_working_column'] = np.nan
    #for country in gdf.index.values:
    #    #gdf[df_working_column][country] = getCountryStat(country, df_working, stat)
    #    gdf.loc[country,'df_working_column'] = getCountryStat(country, df_working, stat)
    
    #from matplotlib import cm
    #viridis = cm.get_cmap('plasma', 8)

    if colormap == 'viridis':
        cmap = 'viridis'
    elif colormap == '5 clusters':
        #colors=['#E3C8E8', '#3B4CC0', '#B30423', '#F39879', '#8CB1FF']
        colors=['#EEDAF2', '#3B4CC0', '#B30423', '#F2AAB4', '#8CB1FF']
        #colors=['#85004B', '#3B4CC0', '#B30423', '#F2AAB4', '#8CB1FF']
        cmap = clr.LinearSegmentedColormap.from_list(name = 'customcmap', colors=colors)
    elif colormap == 'voting':
        colors=['#CD5C5C', '#FFEF5A', '#60A384']
        cmap = clr.LinearSegmentedColormap.from_list(name = 'customcmap', colors=colors)
    elif colormap == 'voting_clusters':
        #colors=['#F2AAB4', '#EEDAF2', '#8CB1FF']
        colors=['#B30423', '#EEDAF2', '#3B4CC0']
        cmap = clr.LinearSegmentedColormap.from_list(name = 'customcmap', colors=colors)
    elif colormap == 'voting_clusters_reverse':
        #colors=['#8CB1FF', '#EEDAF2', '#F2AAB4']
        colors=['#3B4CC0', '#EEDAF2', '#B30423']
        cmap = clr.LinearSegmentedColormap.from_list(name = 'customcmap', colors=colors)
    elif colormap == 'sanctions':
        #colors=['#a0a0a4', '#ace5ee', '#306070']
        colors=['#a0a0a4', '#60A384', '#CD5C5C']
        #cmap = "RdYlGn"
        cmap = clr.LinearSegmentedColormap.from_list(name = 'customcmap', colors=colors)
    elif colormap == 'sanctions_clusters':
        #colors=['#a0a0a4', '#F2AAB4', '#8CB1FF']
        colors=['#a0a0a4', '#B30423', '#3B4CC0']
        cmap = clr.LinearSegmentedColormap.from_list(name = 'customcmap', colors=colors)
    else:
        colors = colormap # ['#EEDAF2', '#3B4CC0', '#F2AAB4', '#B30423', '#8CB1FF']
        cmap = clr.LinearSegmentedColormap.from_list(name = 'customcmap', colors=colors)
        
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
  
    #рисование карты
    fig, axs = plt.subplots(1, 2, dpi=400, figsize=(10, 4), gridspec_kw = {'width_ratios':[3,1]})


    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    
    geomap = gdf.plot(
        column=stat,
        missing_kwds={'color': 'lightgrey'}, 
        ax = axs[0],
        cmap = cmap,
      #  color = ['72FF51', 'FFBD32', '636FFC', 'FC303B', 'FFFF3B'],
      #  cmap = 'Set1',
        legend=addlegend,
        legend_kwds={'label': "Shareof Population",
                        'orientation': "horizontal"}
    )
    
    
    if colormap == '5 clusters':
        from matplotlib import cm
        #viridis = cm.get_cmap(cmap, n_clusters)
        custom_patches =  [
            ptch.Patch(color=colors[1]),
            ptch.Patch(color=colors[4]),
            ptch.Patch(color=colors[0]),
            ptch.Patch(color=colors[3]),
            ptch.Patch(color=colors[2]),
        ]
        if lang == 'ru':
            custom_labels = [
                    'Воинствующий Запад',
                    'Умеренный Запад',
                    'Нейтральные страны',
                    'Сочувствующие России',
                    'Решительные сторонники России',
            ]
        if lang == 'en':
            custom_labels = [
                    'Hardline West',
                    'Moderate West',
                    'Neutral countries',
                    'Sympathetic with Russia',
                    'Aligned with Russia',
            ]
        axs[1].legend(custom_patches, custom_labels)
    if colormap == 'voting' or colormap == 'voting_clusters' or colormap == 'voting_clusters_reverse':
        from matplotlib import cm
        #viridis = cm.get_cmap(cmap, n_clusters)
        custom_patches =  [
            ptch.Patch(color=colors[0]),
            ptch.Patch(color=colors[1]),
            ptch.Patch(color=colors[2])
        ]
        if lang == 'ru':
            custom_labels = [
                'Проголосовали против резолюции',
                'Воздержались или не голосовали',
                'Проголосовали в поддержку резолюции',
            ]
        if lang == 'en':
            custom_labels = [
                'Against',
                'Abstention or no vote',
                'In favour',
            ]
        axs[1].legend(custom_patches, custom_labels)
    if colormap == 'sanctions' or colormap == 'sanctions_clusters':
        from matplotlib import cm
        #viridis = cm.get_cmap(cmap, n_clusters)
        custom_patches =  [
            ptch.Patch(color=colors[0]),
            ptch.Patch(color=colors[1]),
            ptch.Patch(color=colors[2])
        ]
        if lang == 'ru':
            custom_labels = [
                'Россия',
                'Санкции не введены',
                'Ввод санкций',
            ]
            title = "Введение санкций против России"
        if lang == 'en':
            custom_labels = [
                'Russia',
                'No sanctions',
                'Sanctions',
            ]
            title = "Imposition of Sanctions on Russia"
        axs[1].legend(custom_patches, custom_labels)

    
    #axs[0].set_title(savename)
    geomap.set_axis_off()
    axs[1].set_axis_off()
    
    plt.savefig(savename, facecolor='#FFFFFF', dpi=300)
    plt.show()

def plot_lisa_map(gdf, year, horizon, k_neighbors, weight_type, figsize=(12, 10), save_path=None):
    """
    Визуализирует карту LISA-кластеров.

    Параметры:
        gdf : GeoDataFrame с колонкой 'cluster' (категории).
        year : int, год (для заголовка).
        horizon : int, горизонт.
        k_neighbors : int, число соседей.
        weight_type : str, тип весов.
        figsize : tuple, размер фигуры.
        save_path : str или None, путь для сохранения.
    """
    # Фиксированный порядок категорий и соответствующие цвета
    categories = ['High-High', 'Low-Low', 'High-Low', 'Low-High', 'Нет значимости', 'Нет данных']
    colors = ['#d7191c', '#2c7bb6', '#f4a582', '#abd9e9', '#e0e0e0', '#f7f7f7']
    cmap = clr.ListedColormap(colors)

    # Преобразуем в категориальный тип, чтобы легенда содержала все категории
    gdf['cluster'] = pd.Categorical(gdf['cluster'], categories=categories, ordered=False)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf.plot(
        column='cluster',
        ax=ax,
        legend=True,
        categorical=True,
        cmap=cmap,
        edgecolor='black',
        linewidth=0.3,
        legend_kwds={'loc': 'lower left', 'title': 'LISA Cluster'}
    )
    ax.set_title(f'LISA clusters for {year} (h={horizon}, k={k_neighbors}, weight={weight_type})')
    ax.set_axis_off()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()