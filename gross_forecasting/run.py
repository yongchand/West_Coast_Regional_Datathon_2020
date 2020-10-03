import pandas as pd
from gross_forecasting.inflation import adjust_monetization
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def prepare_data(file_path: str):
    """ pre-processing of data to make it capable of accessing by models"""
    movie_industry_df = pd.read_csv(file_path, encoding='latin-1')
    return movie_industry_df


def pre_processing(df: pd.DataFrame, include_budget: bool):
    df = df.drop(['budget', 'gross', 'released', 'year', 'name'], axis=1)
    df['inflated_budget'] = df['inflated_budget'] / 1e+6
    df['inflated_gross'] = df['inflated_gross'] / 1e+6

    y = df['inflated_gross']

    if include_budget:
        x = df.drop(['inflated_gross'], axis=1)
        n_components = 20
    else:
        x = df.drop(['inflated_gross', 'inflated_budget'], axis=1)
        n_components = 4

    x_dummy = pd.get_dummies(x, drop_first=True)

    pca = PCA(n_components=n_components)
    pca.fit(x_dummy)

    x_pca_dummy = pca.transform(x_dummy)
    x_train, x_test, y_train, y_test = train_test_split(x_pca_dummy, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def linear_regression_training(x_train: pd.DataFrame, y_train: pd.Series):
    """ Linear regression model training for movies w/ or w/o budget."""
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    return lr


def linear_regression_inference(lr: LinearRegression, y_test: pd.Series, x_test: pd.Series):
    y_pred = lr.predict(x_test)
    print("Linear regression R-squared score is:", r2_score(y_true=y_test, y_pred=y_pred))
    return y_pred


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

    df_w_budget = df[df['budget'] > 0]
    df_wo_budget = df[df['budget'] == 0]

    x_budget_train, x_budget_test, y_budget_train, y_budget_test = pre_processing(df_w_budget, include_budget=True)
    x_wo_budget_train, x_wo_budget_test, y_wo_budget_train, y_wo_budget_test = pre_processing(df_wo_budget,
                                                                                              include_budget=False)

    budget_model = linear_regression_training(x_budget_train, y_train=y_budget_train)
    wo_budget_model = linear_regression_training(x_wo_budget_train, y_train=y_wo_budget_train)

    linear_regression_inference(budget_model, x_test=x_budget_test, y_test=y_budget_test)
    linear_regression_inference(wo_budget_model, y_test=y_wo_budget_test, x_test=x_wo_budget_test)


if __name__ == '__main__':
    run()