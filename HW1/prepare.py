
def prepare_data(training_data, new_data):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import sklearn.datasets as ds
    import matplotlib.pyplot as plt
    import math
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from pandas.core.frame import DataFrame
    from sklearn.model_selection import train_test_split

    copied_training_data = training_data.copy()
    copied_new_data = new_data.copy()

    copied_new_data.household_income.fillna(copied_training_data.household_income.median(), inplace=True)

    copied_training_data['SpecialProperty'] = copied_training_data['blood_type'].isin(['O+', 'B+'])
    copied_training_data.drop(columns=['blood_type'], inplace=True)

    copied_new_data['SpecialProperty'] = copied_new_data['blood_type'].isin(['O+', 'B+'])
    copied_new_data.drop(columns=['blood_type'], inplace=True)

    scaler_standard = StandardScaler()
    scaler_standard.fit(copied_training_data[['PCR_01', 'PCR_02', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_08']])
    copied_new_data[['PCR_01', 'PCR_02', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_08']] = scaler_standard.transform(copied_new_data[['PCR_01', 'PCR_02', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_08']])

    scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
    scaler_minmax.fit(copied_training_data[['PCR_03', 'PCR_04', 'PCR_09', 'PCR_10']])
    copied_new_data[['PCR_03', 'PCR_04', 'PCR_09', 'PCR_10']] = scaler_minmax.transform(copied_new_data[['PCR_03', 'PCR_04', 'PCR_09', 'PCR_10']])

    return copied_new_data