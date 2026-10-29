import sys
sys.path.insert(1, '..')

from utils.countrymerger import *
import pandas as pd
import numpy as np
import warnings

def swap_country_id(country_tag, id_to_replace, id_to_be_replaced_with):
    if str(country_tag) != str(id_to_replace):
        #if str(id_to_replace) in str(country_tag):  # For checking myself
        #    print(f"Not replacing {country_tag} with {id_to_replace} -> {id_to_be_replaced_with} convertion")
        return country_tag
    return str(id_to_be_replaced_with)

def convert_country_df(data: pd.DataFrame, data_id_col: str, separator: str = None, standard_to_convert: str = "STATE_en_UN", warning = True, replace_missing=None, purge=False, numeric_type=None, print_convertions=True):
    """ Function to convert a series of country ids to a different standard

    ---
    Parameters:
        data (Pandas.DataFrame): Data with country identifiers
        data_id_col (str) : Country id column name
        separator (str): separator of country identifiers in a cell (can be string or None)
        standard_to_convert (str): standard to convert to from the list of supported standrds, to see the list consult countrymerger.KEY_COLUMNS. Default is STATE_EN_UN for UN standard country names (http://unterm.un.org).
        warning (bool): issue warnings when unable to idntify country id.
        replace_missing (str or None): what to replace missing values with.
        purge (bool): whether to remove unidentified country ids. Only works if replace_missing is None.

    ---
    Returns:
        converted_country_series (Pandas.Series): country_series in a new format
    """
    replacement_dict = get_id_dict(data[data_id_col], separator, standard_to_convert, warning, replace_missing=replace_missing, numeric_type=numeric_type)
    data_replaced = data.copy()
    for i in replacement_dict.items():
        if print_convertions: print(i[0], i[1])
        data_replaced[data_id_col] = data_replaced[data_id_col].apply(lambda x: swap_country_id(x, i[0], i[1]))#str(x).replace(str(i[0]),str(i[1])))
    if purge:
        data_replaced[data_id_col] = data_replaced[data_id_col].replace('None', np.nan)
        if replace_missing is not None:
            warnings.warn('replace_missing is not None, purging not supported')
        data_replaced=data_replaced.dropna(subset=[data_id_col])
    return data_replaced


def get_id_set(country_series: pd.Series, separator: str = None):
    """Returns a set of countries from a series of country ids (possibly with separators)
    """
    def _add_to_country_list(country_list, country_ids, separator):
        if pd.isna(country_ids): return None
        country_ids = str(country_ids)
        #print(country_ids)
        if separator is None:
            country_list += [country_ids]#.strip()]
        elif separator in country_ids:
            #print([country_id.strip() for country_id in country_ids.split(separator)])
            #country_list += [country_id.strip() for country_id in country_ids.split(separator)]
            country_list += [country_id for country_id in country_ids.split(separator)]
        else:
            #print(country_ids.strip())
            country_list += [country_ids]#.strip()]
    country_list = list()
    country_series.apply(lambda x: _add_to_country_list(country_list, x, separator))
    #print(country_list)
    country_set = set(country_list)
    return country_set

def get_id_dict(country_series: pd.Series, separator: str, standard_to_convert: str = "STATE_en_UN", warning = True, replace_missing='keep', numeric_type=None):
    """Returns a dict of old country ids and new ones
    from a series of country ids (possibly with separators)
    """
    country_set = get_id_set(country_series, separator)

    keys_df = loadKeyDf(load_extra=True)
    if numeric_type == 'iso':
        keys_df = keys_df[[standard_to_convert, 'ISO_Code', 'ISO_GIS']]
    elif numeric_type == 'cow':
        keys_df = keys_df[[standard_to_convert] + [MIXED_STANDARDS['cow_mixed']['master']] + MIXED_STANDARDS['cow_mixed']['slaves']]
        keys_df['COW_Country_Code'] = keys_df['COW_Country_Code'].astype(str)  # unsure if moving astype(str) to whole df will break anything
    elif numeric_type is None:
        pass
    else:
        raise ValueError(f"Expected numeric_type to be None, 'cow' or 'iso', got {numeric_type} instead")
    converting_x = keys_df.columns.get_loc(standard_to_convert)
    keys = keys_df.values
    #  keys_lower = keys_df.applymap(lambda s: s.lower() if type(s) == str else s)  Pandas<2.10
    keys_lower = keys_df.map(lambda s: s.lower() if type(s) == str else s)
    conversion_dict = {}
    
    for country in country_set:
        country_lower = country.lower()
        try:
            i = np.where(keys_lower == country_lower.strip())[0][0]
            country_converted = keys[i, converting_x]
            #print(country, country_converted)
            conversion_dict[country] = country_converted
        except IndexError:
            if replace_missing == 'keep':
                if warning: warnings.warn(f"Unknown identifier {country}, keeping as is")
                conversion_dict[country] = country  # keeping country as is
            else:
                if warning: warnings.warn(f"Unknown identifier {country}, replacing with {replace_missing}")
                conversion_dict[country] = replace_missing
    return conversion_dict