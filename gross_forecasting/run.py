import pandas as pd
from gross_forecasting.inflation import adjust_monetization


def prepare_data(file_path):
    """ pre-processing of data to make it capable of accessing by models"""
    movie_industry_df = pd.read_csv(file_path, encoding='latin-1')


def run():
    cpi_df = pd.read_csv('../cpi.csv')
    cpi_df['Label'] = pd.to_datetime(cpi_df['Label'], infer_datetime_format=True)
    cpi_df.set_index('Label', inplace=True)

    df = pd.read_csv('../movie_industry.csv', encoding='latin-1')
    gross = df['gross']
    budget = df['budget']

    released = df['released'].apply(lambda time: pd.Timestamp(time).replace(day=1))

    print(type(gross), type(released))
    df['inflated_gross'] = adjust_monetization(released, gross, cpi_df)
    df['inflated_budget'] = adjust_monetization(released, budget, cpi_df)
    print(df[['budget', 'gross', 'inflated_budget', 'inflated_gross']])


if __name__ == '__main__':
    run()