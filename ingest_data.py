import pandas as pd
from minsearch import Index

def load_data():

    df = pd.read_csv('dataset/coffee_analysis.csv')
    cols = df.columns.tolist()
    df.dropna(inplace=True)
    df[['100g_USD', 'rating']] = df[['100g_USD', 'rating']].astype(str)
    docs = df.to_dict(orient='records')

    return docs

def build_index(documents):
    index = Index(
        text_fields=['name','roaster','100g_USD', 'rating', 'review_date', 'desc_1', 'desc_2', 'desc_3'],
        keyword_fields=['origin_1', 'origin_2', 'roast', 'loc_country']
    )
    index.fit(documents)
    return index


