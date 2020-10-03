import pandas as pd


def adjust_monetization(time: pd.Series, monetization: pd.Series, cpi_df: pd.DataFrame):
    """ Adjust based on Aug 2020 data."""
    cpi_aug = cpi_df.loc[pd.to_datetime('2020-08-01')]['Value']
    combined_df = pd.concat([time, monetization], axis=1)

    return combined_df.apply(lambda t: int(cpi_aug / cpi_df.loc[t[0]]['Value'] * t[1]), axis=1)