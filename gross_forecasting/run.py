import pandas as pd
from gross_forecasting.inflation import adjust_monetization


def prepare_data(file_path: str):
    """ pre-processing of data to make it capable of accessing by models"""
    movie_industry_df = pd.read_csv(file_path, encoding='latin-1')
    return movie_industry_df


def linear_regression_training(df: pd.DataFrame):
    """ Train linear regression model for movies w/ or w/o budgets"""
    pass


def run():
    movie_industry_file_path = '../movie_industry.csv'
    cpi_df = pd.read_csv('../cpi.csv')
    cpi_df['Label'] = pd.to_datetime(cpi_df['Label'], infer_datetime_format=True)
    cpi_df.set_index('Label', inplace=True)

    df = prepare_data(movie_industry_file_path)
    gross = df['gross']
    budget = df['budget']

    released = df['released'].apply(lambda time: pd.Timestamp(time).replace(day=1))

    df['inflated_gross'] = adjust_monetization(released, gross, cpi_df)
    df['inflated_budget'] = adjust_monetization(released, budget, cpi_df)
    print(df[['budget', 'gross', 'inflated_budget', 'inflated_gross']])


if __name__ == '__main__':
    run()