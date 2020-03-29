import time
from typing import List

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine, Unit

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

DATA_TAXI = r'C:\Users\alewz\Google Drive\programming\projects_al\first_pytourch\PYTORCH_NOTEBOOKS\Data\NYCTaxiFares.csv'

# variables ############################################################################################################

cols_number: List[str] = [
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'passenger_count',
    'distance',
]
cols_categories: List[str] = [
    'hours',
    'am_or_pm',
    'day_of_week',
]

# features engineering #################################################################################################

if __name__ == '__main__':

    taxi = pd.read_csv(DATA_TAXI)
    print(taxi.head())
    print(taxi.dtypes)

    # haversine distance from log and lat

    taxi['pickup'] = list(zip(taxi['pickup_longitude'], taxi['pickup_latitude']))
    taxi['dropoff'] = list(zip(taxi['dropoff_longitude'], taxi['dropoff_latitude']))

    vhaversine = np.vectorize(haversine)
    taxi['distance'] = vhaversine(taxi.pickup.values, taxi.dropoff.values, unit=Unit.MILES)

    # parsing datetime
    taxi['pickup_datetime'] = pd.to_datetime(taxi['pickup_datetime'])
    taxi['pickup_datetime'] = taxi['pickup_datetime'] - pd.Timedelta(hours=4)
    taxi['hours'] = taxi['pickup_datetime'].dt.hour
    taxi['am_or_pm'] = np.where(taxi['hours'] > 12, 'pm', 'am')
    taxi['day_of_week'] = taxi['pickup_datetime'].dt.strftime('%a')

    for col in cols_categories:
        taxi[col] = taxi[col].astype('category')
    y_col = ['fare_amount']

    features = taxi[cols_categories + cols_number + y_col]
    target: pd.Series = taxi[y_col]

    print(features.dtypes)
    print(features.head())

    taxi.to_pickle('taxi_cleaned')

    sns.distplot(target)
    plt.show()
    print(target.describe())
