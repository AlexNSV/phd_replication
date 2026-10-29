import sys
sys.path.append("..")

import pandas as pd


def pretest_dataset(df):
    df = df.reset_index()
    df.sort_values('value')
    df.groupby('year').sum()
    print('Russia/USSR')
    print(df[df['ego']=='Russian Federation'].groupby('year').sum())
    print('USA')
    print(df[df['ego']=='United States'].groupby('year').sum())
    print('China')
    print(df[df['ego']=="China, People's Republic of"].groupby('year').sum())

def test_triple_df(df):
    system_df = pd.read_csv('../data/raw/system_membership/system2016.csv')
    return system_df