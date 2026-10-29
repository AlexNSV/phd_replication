import pandas as pd
import logging

import numpy as np
import re
from scipy import stats
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin



def compute_deltas(df_y, horizons=(1, 2, 3), year=None, return_yearly_mean=False):
    """
    Вычисляет изменения value для каждой страны за заданные горизонты.
     
    Параметры:
        df_y : DataFrame с колонками ['year', 'value', 'country']
               (могут быть и другие, но эти обязательны).
        horizons : tuple/list горизонтов (в годах) для расчёта изменений,
        year : вернуть конкретный год или все (default - None).
        return_yearly_mean : вернуть усредненное по годам. Только если year is None.
    
    Возвращает:
        DataFrame с колонками ['country', 'year'] и для каждого горизонта h
        колонкой 'delta_{h}', содержащей value(year) - value(year - h).
        Если данных за год year-h нет, значение будет NaN.
    """
    # Оставляем только нужные колонки и сортируем
    if 'country' in df_y.columns:
        fullcolumnset = ['country', 'year', 'value']
        novaluecolumnset = ['country', 'year']
    else:
        logging.info(f'co country data found in {df_y.columns}')
        fullcolumnset = ['year', 'value']
        novaluecolumnset = ['year']    
    df = df_y[fullcolumnset].sort_values(novaluecolumnset).copy()
    
    
    # Для каждого горизонта строим сдвинутые значения и вычисляем дельту
    delta_frames = []
    for h in horizons:
        # Создаём копию с year+h, чтобы при merge сопоставить текущему году
        shifted = df[fullcolumnset].copy()
        shifted['year'] = shifted['year'] + h
        shifted = shifted.rename(columns={'value': f'value_lag_{h}'})
        # Merge по (country, year) – теперь year в shifted = исходный year + h
        merged = df.merge(shifted, on=novaluecolumnset, how='left')
        # Дельта = текущее значение - значение h лет назад
        merged[f'delta_{h}'] = merged['value'] - merged[f'value_lag_{h}']
        if 'country' in df_y.columns:
            delta_frames.append(merged[['country', 'year', f'delta_{h}']])
        else:
            delta_frames.append(merged[['year', f'delta_{h}']])
    
    # Объединяем все дельты в один DataFrame
    deltas = delta_frames[0]
    for frame in delta_frames[1:]:
        deltas = deltas.merge(frame, on=novaluecolumnset, how='outer')

    #print(horizons, year)
    if year:
        if len(horizons) > 1:
            raise NotImplementedError('Unsupported for multiple horizons')
        # Выбор конкретного года
        #print(deltas[deltas['year'] == year][['country', f'delta_{horizons[0]}']])
        #delta_year = deltas[deltas['year'] == year][['country', f'delta_{horizons[0]}'[0]]].dropna()
        delta_year = deltas[deltas['year'] == year][['country', f'delta_{horizons[0]}']].dropna()
        delta_year.columns = ['country', 'delta_mean']
        
        deltas = delta_year
    elif return_yearly_mean:
        if len(horizons) > 1:
            raise NotImplementedError('Unsupported for multiple horizons')
        # Среднее изменение по всем годам для каждой страны
        delta_mean = deltas.groupby('country')[f'delta_{horizons[0]}'].mean().reset_index()
        delta_mean.columns = ['country', 'delta_mean']
        deltas = delta_mean
    logging.info(f"Размер таблицы дельт: {deltas.shape}")
    return deltas

def build_neighbor_features(df_x, deltas,
                            top_k_list=(5, 10, 15, 20),
                            horizons=(1, 2, 3),
                            include_own_change=True,
                            own_change_horizon=1,
                            own_change_col='country_a_change_1y',
                            threshold=None,
                            return_neighbor_lists=False):  # новый параметр
    """
    ... (прежняя документация) ...
    
    Параметры:
        ...
        return_neighbor_lists : bool, по умолчанию False
            Если True, добавляет колонки со списками стран-соседей:
            - в дистанционном режиме: top{K}_neighbors для каждого K
            - в пороговом режиме: neighbors_list
            Эти колонки не используются в регрессионном анализе.
    """
    has_year = 'year' in df_x.columns

    # ---- Пороговый режим (сила близости) ----
    if threshold is not None:
        if top_k_list is not None:
            print("Пороговый режим активен: параметр top_k_list игнорируется.")

        if has_year:
            df_x_filtered = df_x[df_x['value'] >= threshold].copy()
            # Для списков сохраним упорядоченные country_b (по убыванию value)
            if return_neighbor_lists:
                # Сортируем по убыванию value,2022 чтобы в списке сначала шли более близкие
                df_x_filtered_sorted = df_x_filtered.sort_values(['country_a', 'year', 'value'], ascending=False)
                # Группируем и собираем списки
                neighbor_lists = df_x_filtered_sorted.groupby(['country_a', 'year'])['country_b'].apply(list).reset_index()
                neighbor_lists.rename(columns={'country_b': 'neighbors_list'}, inplace=True)
            # Объединяем с дельтами
            merged = df_x_filtered.merge(
                deltas,
                left_on=['country_b', 'year'],
                right_on=['country', 'year'],
                how='left'
            ).drop(columns=['country'])
        else:
            df_x_agg = df_x.groupby(['country_a', 'country_b'], as_index=False)['value'].mean()
            df_x_filtered = df_x_agg[df_x_agg['value'] >= threshold]
            if return_neighbor_lists:
                # Сортируем по убыванию value (без года) и собираем списки по country_a
                df_x_filtered_sorted = df_x_filtered.sort_values(['country_a', 'value'], ascending=False)
                neighbor_lists = df_x_filtered_sorted.groupby('country_a')['country_b'].apply(list).reset_index()
                neighbor_lists.rename(columns={'country_b': 'neighbors_list'}, inplace=True)
                # В режиме без года списки будут одинаковы для всех лет; позже размножим
            merged = df_x_filtered.merge(
                deltas,
                left_on='country_b',
                right_on='country',
                how='left'
            )

        # Вычисляем средние дельты
        agg_dict = {f'neighbor_change_{h}y': (f'delta_{h}', 'mean') for h in horizons}
        final_df = merged.groupby(['country_a', 'year']).agg(**agg_dict).reset_index()

        # Добавляем список соседей, если нужно
        if return_neighbor_lists:
            if has_year:
                final_df = final_df.merge(neighbor_lists, on=['country_a', 'year'], how='left')
            else:
                # В статическом режиме neighbour_lists содержит только country_a, без года
                # Нужно сделать cross join со всеми годами, присутствующими в final_df
                years = final_df['year'].unique()
                # Создаём DataFrame с комбинациями (country_a, year) и списком
                expanded_lists = []
                for _, row in neighbor_lists.iterrows():
                    ca = row['country_a']
                    nlist = row['neighbors_list']
                    for y in years:
                        expanded_lists.append({'country_a': ca, 'year': y, 'neighbors_list': nlist})
                expanded_df = pd.DataFrame(expanded_lists)
                final_df = final_df.merge(expanded_df, on=['country_a', 'year'], how='left')

    # ---- Дистанционный режим (топ-K по наименьшему value) ----
    else:
        if has_year:
            df_x_sorted = df_x.sort_values(['country_a', 'year', 'value']).copy()
            df_x_sorted['rank'] = df_x_sorted.groupby(['country_a', 'year']).cumcount()

            merged_all = df_x_sorted.merge(
                deltas,
                left_on=['country_b', 'year'],
                right_on=['country', 'year'],
                how='left'
            ).drop(columns=['country'])

            results = []
            for K in top_k_list:
                topK = merged_all[merged_all['rank'] < K]
                # Агрегация средних дельт
                agg_dict = {f'top{K}_change_{h}y': (f'delta_{h}', 'mean') for h in horizons}
                grouped = topK.groupby(['country_a', 'year']).agg(**agg_dict).reset_index()
                # Если нужны списки соседей
                if return_neighbor_lists:
                    # Берем topK и собираем country_b в список (по порядку rank)
                    # Для этого отсортируем по рангу (уже отсортировано) и сгруппируем
                    neighbor_groups = topK.sort_values(['country_a', 'year', 'rank']).groupby(['country_a', 'year'])['country_b'].apply(list).reset_index()
                    neighbor_groups.rename(columns={'country_b': f'top{K}_neighbors'}, inplace=True)
                    grouped = grouped.merge(neighbor_groups, on=['country_a', 'year'], how='left')
                results.append(grouped)

            if results:
                final_df = results[0]
                for df in results[1:]:
                    # При слиянии нужно объединять по (country_a, year)
                    # Для колонок с дельтами и со списками - они разные
                    # merge с outer по ключам
                    final_df = final_df.merge(df, on=['country_a', 'year'], how='outer', suffixes=('', '_dup'))
                    # Если появились дублирующиеся колонки, можно оставить только первые (но обычно нет)
                    # Просто удаляем суффиксы (если они возникли, но по идее колонки уникальны)
                    # Лучше явно указать, что merge только по ключам, а остальные колонки уникальны
                    # Оставим как есть, но очистим колонки с суффиксом _dup, если такие появились
                    dup_cols = [c for c in final_df.columns if c.endswith('_dup')]
                    if dup_cols:
                        final_df.drop(columns=dup_cols, inplace=True)
            else:
                final_df = pd.DataFrame(columns=['country_a', 'year'])

        else:
            # Без года
            df_x_agg = df_x.groupby(['country_a', 'country_b'], as_index=False)['value'].mean()
            df_x_sorted = df_x_agg.sort_values(['country_a', 'value']).copy()
            df_x_sorted['rank'] = df_x_sorted.groupby('country_a').cumcount()

            results = []
            for K in top_k_list:
                topK_pairs = df_x_sorted[df_x_sorted['rank'] < K][['country_a', 'country_b']]
                merged = topK_pairs.merge(deltas, left_on='country_b', right_on='country', how='left')
                agg_dict = {f'top{K}_change_{h}y': (f'delta_{h}', 'mean') for h in horizons}
                grouped = merged.groupby(['country_a', 'year']).agg(**agg_dict).reset_index()
                if return_neighbor_lists:
                    # В статическом режиме список стран одинаков для всех лет, возьмём из топK_pairs
                    # Сортируем по рангу (уже отсортировано) и собираем список по country_a
                    neighbor_groups = df_x_sorted[df_x_sorted['rank'] < K].sort_values(['country_a', 'rank']).groupby('country_a')['country_b'].apply(list).reset_index()
                    neighbor_groups.rename(columns={'country_b': f'top{K}_neighbors'}, inplace=True)
                    # Размножаем для каждого года
                    years = deltas['year'].unique()
                    expanded = []
                    for _, row in neighbor_groups.iterrows():
                        ca = row['country_a']
                        nlist = row[f'top{K}_neighbors']
                        for y in years:
                            expanded.append({'country_a': ca, 'year': y, f'top{K}_neighbors': nlist})
                    expanded_df = pd.DataFrame(expanded)
                    grouped = grouped.merge(expanded_df, on=['country_a', 'year'], how='left')
                results.append(grouped)

            if results:
                final_df = results[0]
                for df in results[1:]:
                    final_df = final_df.merge(df, on=['country_a', 'year'], how='outer')
            else:
                final_df = pd.DataFrame(columns=['country_a', 'year'])

    # ---- Добавляем изменение самой country_a (общее для обоих режимов) ----
    if include_own_change:
        delta_col = f'delta_{own_change_horizon}'
        if delta_col not in deltas.columns:
            raise ValueError(f"В deltas отсутствует колонка {delta_col}. "
                             f"Убедитесь, что горизонт {own_change_horizon} был рассчитан в compute_deltas.")
        own = deltas[['country', 'year', delta_col]].rename(
            columns={'country': 'country_a', delta_col: own_change_col}
        )
        final_df = final_df.merge(own, on=['country_a', 'year'], how='left')

    return final_df

def build_monadic_features(df_monad,
                           deltas,
                           value_col='value',
                    
       horizons=(1, 2, 3),
                           include_value=True,
                           include_lags=False,
                           lag_horizons=(1, 2, 3)):
    """
    Создаёт признаки на основе монадных данных (страна, год, значение) для каждой страны.

    Функция объединяет исходное значение (опционально) и заранее рассчитанные изменения
    (deltas) за указанные горизонты. Никакого поиска соседей не выполняется.

    Параметры:
        df_monad : DataFrame с колонками ['country', 'year', value_col].
                   Каждая строка соответствует стране в определённый год.
        deltas : результат compute_deltas – DataFrame с колонками
                 ['country', 'year', 'delta_1', 'delta_2', ...].
        value_col : str, название колонки со значением в df_monad.
        horizons : tuple/list горизонтов, для которых нужно добавить изменения.
        include_value : bool, включать ли в итоговую таблицу текущее значение value_col.
        include_lags : bool, добавлять ли лаги значения (value_col за предыдущие годы).
        lag_horizons : tuple/list горизонтов для лагов (используется, если include_lags=True).

    Возвращает:
        DataFrame с колонками:
            country, year,
            (если include_value=True) value_col,
            delta_1, delta_2, ... (для каждого горизонта из horizons),
            (если include_lags=True) lag_{h} (value_col за h лет до текущего года).
    """
    # Проверка наличия необходимых колонок
    required = ['country', 'year', value_col]
    missing = [c for c in required if c not in df_monad.columns]
    if missing:
        raise ValueError(f"В df_monad отсутствуют колонки: {missing}")

    for h in horizons:
        if f'delta_{h}' not in deltas.columns:
            raise ValueError(f"В deltas отсутствует колонка delta_{h}. "
                             f"Сначала вызовите compute_deltas с горизонтами {horizons}.")

    # Начинаем с дельт по стране и году
    result = deltas[['country', 'year'] + [f'delta_{h}' for h in horizons]].copy()

    # Добавляем текущее значение value_col
    if include_value:
        value_df = df_monad[['country', 'year', value_col]].copy()
        result = result.merge(value_df, on=['country', 'year'], how='left')

    # Добавляем лаги значений (опционально)
    if include_lags:
        # Создаём копию с year+h, чтобы присоединить значение из предыдущего года
        for h in lag_horizons:
            lag_df = df_monad[['country', 'year', value_col]].copy()
            lag_df['year'] = lag_df['year'] + h
            lag_df = lag_df.rename(columns={value_col: f'lag_{h}'})
            result = result.merge(lag_df, on=['country', 'year'], how='left')

    return result

def _needs_scaling(estimator):
    """True, если модель чувствительна к масштабу признаков."""
    if isinstance(estimator, Pipeline):
        estimator = estimator.steps[-1][1]
    return isinstance(estimator, (LinearRegression, Ridge, Lasso))


def _extract_feature_weights(pipeline, feature_names):
    """
    Извлекает веса/важности признаков из обученного пайплайна.

    Возвращает:
        weights : dict {имя_признака: вес}
        intercept : float или None (для линейных моделей)
        weight_kind : str — 'coefficient' | 'importance' | 'unknown'
    """
    # Достаём финальный estimator из пайплайна
    estimator = pipeline
    if isinstance(pipeline, Pipeline):
        estimator = pipeline.steps[-1][1]

    weights = {}
    intercept = None
    weight_kind = 'unknown'

    if hasattr(estimator, 'coef_'):
        coefs = np.asarray(estimator.coef_).ravel()
        if coefs.shape[0] == len(feature_names):
            weights = dict(zip(feature_names, coefs.tolist()))
            weight_kind = 'coefficient'
        else:
            # несогласованная длина — не падаем, просто оставляем пустым
            weights = {name: None for name in feature_names}
        if hasattr(estimator, 'intercept_'):
            try:
                intercept = float(np.asarray(estimator.intercept_).ravel()[0])
            except Exception:
                intercept = None

    elif hasattr(estimator, 'feature_importances_'):
        importances = np.asarray(estimator.feature_importances_).ravel()
        if importances.shape[0] == len(feature_names):
            weights = dict(zip(feature_names, importances.tolist()))
            weight_kind = 'importance'
        else:
            weights = {name: None for name in feature_names}

    else:
        # модель без явных весов (например, KNN, SVR) — фиксируем это
        weights = {name: None for name in feature_names}

    return weights, intercept, weight_kind


# ---------- Основная функция ----------

def regression_analysis(features,
                        target='country_a_change_1y',
                        K_list=None,          # для режима topk (None – авто)
                        horizons=None,        # для режимов topk/threshold (None – авто)
                        models=None,
                        cv_folds=5,
                        random_state=42,
                        feature_columns=None  # для режима 'generic': список признаков
                        ):
    """
    Проводит регрессионный анализ для предсказания целевой переменной.
    Автоматически определяет режим работы по наличию колонок в features:

    - Если есть колонки вида 'top{К}_change_{h}y'  → режим 'topk'
      (используются K_list и horizons, если они переданы; иначе извлекаются из данных)
    - Если таких колонок нет, но есть 'neighbor_change_{h}y' → режим 'threshold'
      (используются horizons из данных; K игнорируется)
    - Если нет ни тех, ни других → режим 'generic': используются произвольные
      числовые признаки. Если задан feature_columns – берётся только указанное
      подмножество; иначе — все числовые колонки, кроме целевой.

    Возвращает:
        results_df : DataFrame с результатами кросс-валидации.
        best_model_info : словарь с информацией о лучшей модели.
            В том числе:
                'weights'       : dict {имя_признака: вес}
                'intercept'     : свободный член (для линейных) или None
                'weight_kind'   : 'coefficient' | 'importance' | 'unknown'
    """
    logger.info("=== Начало регрессионного анализа ===")
    logger.info(f"target='{target}', cv_folds={cv_folds}, random_state={random_state}")

    # ---------- Проверка целевой переменной ----------
    if target not in features.columns:
        logger.error(f"Целевая переменная '{target}' отсутствует в DataFrame.")
        raise ValueError(f"Целевая переменная '{target}' отсутствует в DataFrame.")

    data = features.dropna(subset=[target]).copy()
    logger.info(f"Исходное количество строк: {len(features)}, после удаления пропусков в целевой: {len(data)}")
    if len(data) == 0:
        raise ValueError("После удаления пропусков в целевой переменной не осталось строк.")
    if len(data) < 30:
        raise ValueError("После удаления пропусков в целевой переменной осталось слишком мало строк.")

    # ---------- Автоматическое определение режима и извлечение параметров ----------
    topk_cols = [c for c in data.columns if re.match(r'^top\d+_change_\d+y$', c)]
    neighbor_cols = [c for c in data.columns
                     if c.startswith('neighbor_change_') and c.endswith('y')]

    feature_columns_used = None  # для generic-режима

    if topk_cols:
        mode = 'topk'
        logger.info("Обнаружены колонки вида 'top{K}_change{h}y' → выбран режим 'topk'.")
        all_K = sorted({int(re.search(r'top(\d+)_', c).group(1)) for c in topk_cols})
        all_h = sorted({int(re.search(r'_change_(\d+)y$', c).group(1)) for c in topk_cols})

        if K_list is None:
            K_list = all_K
        else:
            missing_K = [k for k in K_list if k not in all_K]
            if missing_K:
                logger.warning(f"Запрошенные K {missing_K} отсутствуют в колонках. "
                               f"Используются только доступные.")
                K_list = [k for k in K_list if k in all_K]

        if horizons is None:
            horizons = all_h
        else:
            missing_h = [h for h in horizons if h not in all_h]
            if missing_h:
                logger.warning(f"Запрошенные горизонты {missing_h} отсутствуют в колонках. "
                               f"Используются только доступные.")
                horizons = [h for h in horizons if h in all_h]

        if not K_list or not horizons:
            raise ValueError("Не найдено ни одной допустимой комбинации K и h для режима topk.")
        combinations = [(K, h) for K in K_list for h in horizons]

    elif neighbor_cols:
        mode = 'threshold'
        logger.info("Обнаружены колонки 'neighbor_change_{h}y' → выбран режим 'threshold'.")
        all_h = sorted({int(re.search(r'neighbor_change_(\d+)y$', c).group(1))
                        for c in neighbor_cols})
        if horizons is None:
            horizons = all_h
        else:
            missing_h = [h for h in horizons if h not in all_h]
            if missing_h:
                logger.warning(f"Запрошенные горизонты {missing_h} отсутствуют. "
                               f"Используются только доступные.")
                horizons = [h for h in horizons if h in all_h]
        if not horizons:
            raise ValueError("Не найдено ни одного горизонта для режима threshold.")
        combinations = [(-1, h) for h in horizons]

    else:
        # ---------- РЕЖИМ: произвольные признаки ----------
        mode = 'generic'
        logger.info("Специализированные колонки не найдены → режим 'generic' "
                    "(произвольные признаки).")

        if feature_columns is not None:
            missing = [c for c in feature_columns if c not in data.columns]
            if missing:
                logger.warning(f"Указанные признаки отсутствуют в данных и будут пропущены: {missing}")
            feature_columns_used = [c for c in feature_columns
                                    if c in data.columns and c != target]
        else:
            feature_columns_used = [
                c for c in data.columns
                if c != target and pd.api.types.is_numeric_dtype(data[c])
            ]

        # убираем колонки, где нет ни одного значения
        feature_columns_used = [c for c in feature_columns_used if data[c].notna().any()]

        if not feature_columns_used:
            logger.error("Не найдено ни одного числового признака для режима 'generic'.")
            raise ValueError("Не найдено ни одного числового признака для режима 'generic'.")

        logger.info(f"Режим 'generic': используется {len(feature_columns_used)} признаков: "
                    f"{feature_columns_used}")
        # одна «комбинация» — все признаки вместе
        combinations = [(None, None)]

    # ---------- Логирование выбранных параметров ----------
    if mode == 'topk':
        logger.info(f"Итоговые K_list: {K_list}")
        logger.info(f"Итоговые horizons: {horizons}")
    elif mode == 'threshold':
        logger.info(f"Итоговые horizons: {horizons}")
        logger.info("K игнорируется (режим threshold).")
    else:
        logger.info(f"Режим generic: признаков = {len(feature_columns_used)}")

    # ---------- Модели по умолчанию / нормализация ----------
    if models is None:
        logger.debug("Используются модели по умолчанию.")
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=random_state)
        }
    elif isinstance(models, dict):
        if not models:
            raise ValueError("Передан пустой словарь моделей.")
        logger.debug(f"Переданы пользовательские модели: {list(models.keys())}")
    elif isinstance(models, (list, tuple)):
        normalized = {}
        for i, m in enumerate(models):
            name = type(m).__name__
            if name in normalized:
                name = f"{name}_{i}"
            normalized[name] = m
        models = normalized
        logger.debug(f"Модели нормализованы в словарь: {list(models.keys())}")
    else:
        models = {type(models).__name__: models}
        logger.debug(f"Одиночная модель нормализована в словарь: {list(models.keys())}")

    # ---------- Кросс-валидация ----------
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        'R2': 'r2',
        'RMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': make_scorer(mean_absolute_error)
    }

    results = []
    total_combinations = len(models) * len(combinations)
    logger.info(f"Всего комбинаций для перебора: {total_combinations}")

    comb_counter = 0
    for model_name, model in models.items():
        for K, h in combinations:
            comb_counter += 1

            # ---- Формируем набор признаков для этой итерации ----
            if mode == 'topk':
                col = f'top{K}_change_{h}y'
                if col not in data.columns:
                    logger.warning(f"Колонка {col} отсутствует, пропускаем.")
                    continue
                feature_cols_this = [col]
                combo_label = f"K={K}, h={h}"

            elif mode == 'threshold':
                col = f'neighbor_change_{h}y'
                if col not in data.columns:
                    logger.warning(f"Колонка {col} отсутствует, пропускаем.")
                    continue
                feature_cols_this = [col]
                combo_label = f"h={h}"

            else:  # generic
                feature_cols_this = feature_columns_used
                combo_label = f"features={len(feature_cols_this)}"

            logger.debug(f"Комбинация {comb_counter}/{total_combinations}: "
                         f"model={model_name}, {combo_label}")

            df_subset = data[feature_cols_this + [target]].dropna()
            if len(df_subset) < cv_folds:
                logger.warning(f"Пропуск: {model_name}, {combo_label} – "
                               f"недостаточно данных ({len(df_subset)} строк).")
                continue

            if _needs_scaling(model):
                pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
            else:
                pipeline = Pipeline([('model', model)])

            cv_results = cross_validate(
                pipeline,
                df_subset[feature_cols_this],
                df_subset[target],
                cv=cv,
                scoring=scoring,
                return_train_score=False
            )

            mean_r2 = np.mean(cv_results['test_R2'])
            std_r2 = np.std(cv_results['test_R2'])
            mean_rmse = np.mean(cv_results['test_RMSE'])
            mean_mae = np.mean(cv_results['test_MAE'])

            logger.info(f"Результаты: {model_name}, {combo_label} -> "
                        f"R²={mean_r2:.4f} (±{std_r2:.4f})")

            results.append({
                'Model': model_name,
                'K': K,
                'Horizon': h,
                'Mean_R2': mean_r2,
                'Std_R2': std_r2,
                'Mean_RMSE': mean_rmse,
                'Mean_MAE': mean_mae
            })

    if not results:
        raise RuntimeError("Не удалось построить ни одной модели (проверьте данные).")

    # ---------- Сводная таблица ----------
    results_df = (pd.DataFrame(results)
                  .sort_values('Mean_R2', ascending=False)
                  .reset_index(drop=True))
    best_row = results_df.iloc[0]
    best_model_name = best_row['Model']
    best_K = best_row['K']
    best_h = best_row['Horizon']

    # ---- Определяем лучший набор признаков ----
    if mode == 'topk':
        best_col = f'top{best_K}_change_{best_h}y'
        best_features = [best_col]
    elif mode == 'threshold':
        best_col = f'neighbor_change_{best_h}y'
        best_features = [best_col]
    else:  # generic
        best_col = None
        best_features = feature_columns_used

    logger.info(f"Лучшая комбинация: модель={best_model_name}, R²={best_row['Mean_R2']:.4f}")

    # ---------- Обучение финальной модели ----------
    best_model = models[best_model_name]
    if _needs_scaling(best_model):
        final_pipeline = Pipeline([('scaler', StandardScaler()), ('model', best_model)])
    else:
        final_pipeline = Pipeline([('model', best_model)])

    final_df = data[best_features + [target]].dropna()
    final_pipeline.fit(final_df[best_features], final_df[target])

    # ---------- Извлечение весов признаков ----------
    weights, intercept, weight_kind = _extract_feature_weights(final_pipeline, best_features)
    logger.info(f"Веса финальной модели ({weight_kind}): {weights}")
    if intercept is not None:
        logger.info(f"Intercept: {intercept:.4f}")

    best_model_info = {
        'model_name': best_model_name,
        'K': best_K if mode == 'topk' else None,
        'horizon': best_h if mode != 'generic' else None,
        'feature': best_col if best_col is not None else best_features,
        'features': best_features,
        'pipeline': final_pipeline,
        'train_size': len(final_df),
        'cv_mean_R2': best_row['Mean_R2'],
        'cv_mean_RMSE': best_row['Mean_RMSE'],
        'cv_mean_MAE': best_row['Mean_MAE'],
        'mode': mode,
        # --- НОВОЕ ---
        'weights': weights,           # {имя_признака: вес}
        'intercept': intercept,       # свободный член (только для линейных)
        'weight_kind': weight_kind,   # 'coefficient' | 'importance' | 'unknown'
    }

    logger.info("=== Сводная таблица результатов (топ-10) ===")
    logger.info("\n" + results_df.head(10).to_string(index=False))

    logger.info("=== Лучшая конфигурация ===")
    logger.info(f"Модель: {best_model_name}")
    if mode == 'topk':
        logger.info(f"K: {best_K}")
    elif mode == 'threshold':
        logger.info("K: пороговый режим")
    else:
        logger.info("K: generic-режим")
    logger.info(f"Горизонт: {best_h if mode != 'generic' else 'generic'}")
    logger.info(f"Признаки: {best_features}")
    logger.info(f"Средний R²: {best_row['Mean_R2']:.4f} ± {best_row['Std_R2']:.4f}")

    logger.info("=== Регрессионный анализ завершён ===")
    return results_df, best_model_info

def ols_inference(X, y, add_constant=True, feature_names=None, cluster=None):
    """
    OLS с полным инференсом: coef, SE, t, p, 95% CI, R², adj R², F.

    Параметры
    ---------
    X : DataFrame | ndarray (n, k)
    y : Series | ndarray (n,)
    add_constant : bool — добавлять ли столбец единиц
    feature_names : list[str] — имена (если X не DataFrame)
    cluster : ndarray (n,) — метки кластеров для cluster-robust SE (напр. страна).
              Если None — обычные гомоскедастичные SE.

    Возвращает dict с params, se, t, p, ci_low, ci_high, r2, adj_r2,
    f_stat, f_pvalue, n, k, df_resid, feature_names, se_kind.
    """
    from scipy import stats
    
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = list(X.columns)
        X = X.to_numpy(dtype=float)
    else:
        X = np.asarray(X, dtype=float)
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(X.shape[1])]

    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]

    if add_constant:
        X = np.column_stack([np.ones(n), X])
        feature_names = ['const'] + list(feature_names)

    k = X.shape[1]
    df_resid = n - k
    if df_resid <= 0:
        raise ValueError(f"Мало наблюдений: n={n}, k={k}, df={df_resid}")

    # --- коэффициенты (lstsq устойчив к коллинеарности) ---
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid

    # --- (X'X)^-1 ---
    XtX_inv = np.linalg.pinv(X.T @ X)

    # --- стандартные ошибки ---
    if cluster is None:
        sigma2 = rss / df_resid
        cov = sigma2 * XtX_inv
        se_kind = 'homoskedastic'
    else:
        cluster = np.asarray(cluster)
        if len(cluster) != n:
            raise ValueError("cluster должен быть длины n")
        meat = np.zeros((k, k))
        for g in np.unique(cluster):
            mask = cluster == g
            sg = X[mask].T @ resid[mask]
            meat += np.outer(sg, sg)
        G = len(np.unique(cluster))
        c = (G / (G - 1)) * ((n - 1) / df_resid) if G > 1 else 1.0
        cov = c * XtX_inv @ meat @ XtX_inv
        se_kind = f'cluster-robust (G={G})'

    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    with np.errstate(divide='ignore', invalid='ignore'):
        t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_resid))

    t_crit = stats.t.ppf(0.975, df=df_resid)
    ci_low  = beta - t_crit * se
    ci_high = beta + t_crit * se

    # --- F-тест (все регрессоры кроме константы = 0) ---
    if add_constant and k > 1:
        rss_r = tss  # RSS ограниченной модели = TSS
        q = k - 1
        if q > 0 and rss > 0:
            f_stat = ((rss_r - rss) / q) / (rss / df_resid)
            f_pvalue = 1 - stats.f.cdf(f_stat, q, df_resid)
        else:
            f_stat, f_pvalue = np.nan, np.nan
    else:
        f_stat, f_pvalue = np.nan, np.nan

    return {
        'params':     dict(zip(feature_names, beta.tolist())),
        'se':         dict(zip(feature_names, se.tolist())),
        't':          dict(zip(feature_names, t_stats.tolist())),
        'p':          dict(zip(feature_names, p_values.tolist())),
        'ci_low':     dict(zip(feature_names, ci_low.tolist())),
        'ci_high':    dict(zip(feature_names, ci_high.tolist())),
        'r2':         float(r2),
        'adj_r2':     float(adj_r2),
        'f_stat':     float(f_stat) if np.isfinite(f_stat) else None,
        'f_pvalue':   float(f_pvalue) if np.isfinite(f_pvalue) else None,
        'n':          int(n),
        'k':          int(k),
        'df_resid':   int(df_resid),
        'feature_names': list(feature_names),
        'se_kind':    se_kind,
    }

def summary_table(inf):
    """DataFrame с результатами инференса + звёздочки значимости."""
    rows = [{
        'term':    name,
        'coef':    inf['params'][name],
        'se':      inf['se'][name],
        't':       inf['t'][name],
        'p':       inf['p'][name],
        'ci_low':  inf['ci_low'][name],
        'ci_high': inf['ci_high'][name],
    } for name in inf['feature_names']]
    df = pd.DataFrame(rows)
    df['signif'] = df['p'].apply(
        lambda p: '***' if p < 0.001 else
                  '**'  if p < 0.01  else
                  '*'   if p < 0.05  else
                  '.'   if p < 0.1   else '')
    return df


def benjamini_hochberg(p_values, alpha=0.05):
    """FDR-поправка Бенджамини-Хохберга. Возвращает (rejected, p_adj)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    p_adj_sorted = p_sorted * n / np.arange(1, n + 1)
    p_adj_sorted = np.minimum.accumulate(p_adj_sorted[::-1])[::-1]
    p_adj = np.empty(n)
    p_adj[order] = np.clip(p_adj_sorted, 0, 1)
    return p_adj <= alpha, p_adj

def run_singular_regression(predictor_df, power_dfs, gdp_df=None, horizons=(3,), models=None, expermient_label=None,
                            rank_x=False
                           ):
    logging.info(f"Running singular regression ({expermient_label}), gdp_ppp control is {not (gdp_df is None)}")
    res = {}
    df_x=predictor_df.copy()
    if rank_x:
        df_x['value']=df_x.groupby(['year']).rank()['value']
    if models is None:
        models=(LinearRegression(),)
    if not gdp_df is None:
        # для контроля
        gdp_ppp = gdp_df[gdp_df['indicator']=='GDP, PPP (constant 2021 international $)']
        gdp_ppp = compute_deltas(gdp_ppp, horizons=horizons,
                                    year=None, return_yearly_mean=False).rename({f'delta_{horizons[0]}':'value'}, axis=1)
        df_x=df_x.merge(gdp_ppp, left_on=('country','year'), right_on=('country','year'), how='left', suffixes=('','_gdp'))
    for power_cat in power_dfs:
        if ('gdp' in power_cat) and (not gdp_df is None):
            continue
        elif ('beckley' in power_cat) and (not gdp_df is None):
            continue
        elif 'subsystem' in power_cat:
            continue
        logging.info(f'Checking {expermient_label} for {power_cat}')
        deltas = compute_deltas(power_dfs[power_cat], horizons=horizons,
                                year=None, return_yearly_mean=False)
        
        deltas_ranked = deltas.copy()
        deltas_ranked[f'delta_{horizons[0]}'] = deltas_ranked.groupby(['year']).rank(ascending=False)[f'delta_{horizons[0]}']
        
        features = df_x.merge(deltas_ranked, left_on=('country','year'), right_on=('country','year'), how='inner').dropna()
        logging.info(power_cat)
        logging.info(features[['value',f'delta_{horizons[0]}', 'value_gdp']].corr())
        
        #print(features.head())
        try:
            res[power_cat]=regression_analysis(features, target=f'delta_{horizons[0]}', models=models, feature_columns=['value','value_gdp'])
        except ValueError as E:
            res[power_cat]=({'ValueError':E},None)
    return res

logger = logging.getLogger()
