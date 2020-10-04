

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def main():
    data_dir = r'C:\Users\CHAN\Documents\Datathon'



    movie_df = pd.read_csv(data_dir + '/movie_lense/movies.csv')
    movie_df = movie_df[movie_df.genres != '(no genres listed)']
    movie_genres = list(movie_df.genres.str.split("|").explode().unique())
    # list of genres
    movie_df = pd.concat([movie_df.drop('genres', axis=1), movie_df.genres.str.get_dummies(sep='|')], axis=1)
    movie_df['title'] = movie_df['title'].str.split('(').str[0].str.strip()

    dt_parser = lambda x: datetime.utcfromtimestamp(int(x)).strftime("%d/%m/%Y %H:%M:%S")
    tag_df = pd.read_csv(data_dir + '/movie_lense/tags.csv', parse_dates=['timestamp'],  date_parser=dt_parser)
    ntag_df = tag_df.groupby('movieId').agg(lambda x: x.tolist()).drop(['userId', 'timestamp'],axis=1)


    industry_df = pd.read_csv(data_dir + '/movie_industry.csv', encoding='latin-1',engine='python')
    industry_df.released = pd.to_datetime(industry_df.released, infer_datetime_format=True)
    industry_df = industry_df[industry_df.budget != 0.0]
    industry_df = industry_df[industry_df.votes > 1000]

    cpi_df = pd.read_csv(data_dir + '/cpi.csv')
    cpi_df['Label'] = pd.to_datetime(cpi_df.Label, infer_datetime_format=True)
    cpi_df.set_index('Label', inplace=True)

    cpi_aug_2020 = cpi_df.loc[pd.to_datetime('2020-08-01')]['Value']

    def correct_for_inflation(timestamp, amount):
        #convert to 'todays's money'
        return cpi_aug_2020/cpi_df.loc[timestamp.replace(day=1)]['Value']*amount

    def adjust_gross(movie):
        return correct_for_inflation(movie.released, movie.gross)/1000000

    def adjust_budget(movie):
        return correct_for_inflation(movie.released, movie.budget)/1000000

    industry_df['inflation_adjusted_gross'] = industry_df.apply(adjust_gross, axis=1)
    industry_df['inflation_adjusted_budget'] = industry_df.apply(adjust_budget, axis=1)

    nindustry_df = industry_df.drop(['gross', 'budget','star','writer','company','country','director','released'],axis=1)
    nindustry_df = nindustry_df.query("rating not in ['Not specified', 'NOT RATED', 'UNRATED']")
    nindustry_df = pd.get_dummies(nindustry_df, columns = ['rating'])
    nindustry_df = nindustry_df.add_prefix('input_')

    nntag_df = ntag_df.merge(movie_df, how="left", on="movieId")
    ftag_df = nindustry_df.merge(nntag_df, how="left", left_on="input_name", right_on="title")
    ftag_df = ftag_df.drop('input_genre',axis=1)


    ftag_df_adv = ftag_df[ftag_df.Adventure == 1.0]
    tag_list= []
    for x in ftag_df_adv['tag']:
        tag_list.extend(x)

    from collections import Counter
    ctag = Counter(tag_list)
    top_tags= [k for k, v in sorted(ctag.items(), key=lambda item: item[1], reverse=True)][:10]

    tagged_movies = pd.get_dummies(tag_df[tag_df['tag'].isin(top_tags)][['movieId', 'tag']], columns=['tag']).groupby('movieId').max()
    movies_with_tags = ftag_df_adv.merge(tagged_movies, how="left", on='movieId')
    movies_with_tags = movies_with_tags.drop('input_name',axis=1)
    movies_with_tags.replace([np.inf, -np.inf], np.nan, inplace=True)
    movies_with_tags = movies_with_tags.dropna(axis=0)
    y = movies_with_tags.filter(regex='tag_')
    X = movies_with_tags.filter(regex='input_')
    y = y.astype(int)

    print(X,y)
    return(X,y)

if __name__ == "__main__":
    main()




