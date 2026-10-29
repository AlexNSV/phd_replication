from utils.utils import test_sphere, visualise_test

def show_top(df, x=20):
    """Shows top 20 egos in a formatted df"""
    return df.reset_index().groupby(['ego', 'year']).sum(numeric_only=True).sort_values('value', ascending=False).head(x)

def to_mean(df):
    """gets normalised df (value col) by dividing by mean"""
    return df['value'] / df['value'].mean()

def normalize(df):
    """Normalizing a df by std deviation"""
    df_n = (df-df.mean())/df.std()
    df_n = df_n - df_n.min()
    return df_n

def remove_self_loops(df):
    """removes self loops e.g. refugees from Ukraine to Ukraine"""
    temp = df.reset_index()
    temp = temp[temp['ego']!=temp['alter']]
    temp = temp.set_index(['year', 'alter', 'ego'])
    return temp

def merge_sphere(dfs, weights, test=False, normalize=True, label='undefined'):
    """merges subnets of a single sphere with weights (recieves a dict of dfs and a dict of weights)"""
    sphere_triple = None
    for indicator in dfs:
        if normalize:
            dfs[indicator]['value'] = to_mean(dfs[indicator])  # normalize(dfs[indicator])
        if test:
            visualise_test(dfs[indicator].reset_index(), dataname=str(indicator)+'_processed', alter_label='alter', ego_label='ego', year_label='year', value_label='value', topnum=5)
    sphere_triple = sum(dfs[df_sphere] * weights[df_sphere] for df_sphere in dfs)
    sphere_triple = remove_self_loops(sphere_triple)
    if test:
        dfs['overall_sphere'] = sphere_triple
        test_res = test_sphere(dfs)
        print(test_res)
        visualise_test(dfs['overall_sphere'].reset_index(), dataname=f'overall_sphere_{label}', alter_label='alter', ego_label='ego', year_label='year', value_label='value', topnum=5)
    return sphere_triple.fillna(0)

def pairwise_deltas(df, include_reverse=False):
    """
    Создаёт DataFrame со всеми попарными разницами значений между странами по годам.

    Параметры:
        df : pandas.DataFrame
            Должен содержать колонки 'country', 'value', 'year'.
        include_reverse : bool, default=False
            Если False (по умолчанию) – для каждой пары стран (A, B) создаётся только
            одна строка, где country_a < country_b (лексикографически).
            Если True – добавляются обе направленности: (A, B) и (B, A).

    Возвращает:
        pandas.DataFrame с колонками ['country_a', 'country_b', 'delta', 'year'],
        где delta = value(country_a) - value(country_b).
    """
    # Проверка наличия необходимых колонок
    required = ['country', 'value', 'year']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"DataFrame должен содержать колонку '{col}'")

    # Убираем дубликаты (country, year) – если есть, берём среднее значение
    df_unique = df.groupby(['country', 'year'], as_index=False)['value'].mean()

    # Самослияние по году
    merged = pd.merge(
        df_unique.rename(columns={'country': 'country_a', 'value': 'value_a'}),
        df_unique.rename(columns={'country': 'country_b', 'value': 'value_b'}),
        on='year'
    )

    # Убираем пары с одинаковой страной
    merged = merged[merged['country_a'] != merged['country_b']]

    # Если не нужны обе направленности – оставляем только лексикографический порядок
    if not include_reverse:
        merged = merged[merged['country_a'] < merged['country_b']]

    # Вычисляем разность
    merged['delta'] = merged['value_a'] - merged['value_b']

    # Возвращаем только нужные колонки
    return merged[['country_a', 'country_b', 'delta', 'year']].reset_index(drop=True)

def get_value_diff(df):
    """Возвращает df с дельтой по значению value между парами стран на основе монадного датасета"""
    merged = df.merge(df, on='year', suffixes=('_a', '_b'))
    
    # Убираем строки, где страна_a == страна_b
    pairs = merged[merged['country_a'] != merged['country_b']].copy()

    # Вычисляем delta
    pairs['delta'] = abs(pairs['value_a'] - pairs['value_b'])
    return pairs.drop(['value_a','value_b'], axis=1).rename({'delta':'value'}, axis=1)
